"""Core graph primitives for a small audio DSP effect graph.

A graph is a plain mapping with two keys::

    {
        "nodes": ["input", "delay", "mix"],
        "edges": [
            {"from": "input", "to": "delay"},
            {"from": "input", "to": "mix"},
            {"from": "delay", "to": "mix"},
        ],
    }

Node ids are strings and every edge points from an upstream producer to a
downstream consumer.

Floating-point results are compared with an absolute tolerance of ``1e-6``.
"""

from __future__ import annotations

import heapq
from collections import defaultdict

TOLERANCE = 1e-6


class DspError(Exception):
    """Raised when a DSP graph or a processing request is invalid."""


def _is_positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _check_nodes(nodes: list[str]) -> list[str]:
    """Return ``nodes`` as a list, raising on duplicate ids."""
    nodes = list(nodes)
    seen = set()
    for node in nodes:
        if node in seen:
            raise DspError(f"duplicate node id: {node}")
        seen.add(node)
    return nodes


def _check_endpoints(node_set: set[str], edges: list[dict]) -> None:
    for edge in edges:
        for key in ("from", "to"):
            endpoint = edge[key]
            if endpoint not in node_set:
                raise DspError(f"unknown endpoint: {endpoint}")


def _adjacency(nodes: list[str], edges: list[dict]):
    """Build (successors, predecessors, in_degree) after validation."""
    node_set = set(nodes)
    _check_endpoints(node_set, edges)
    succ = {node: set() for node in nodes}
    pred = {node: set() for node in nodes}
    for edge in edges:
        source, target = edge["from"], edge["to"]
        if source == target:
            raise DspError(f"self loop: {source}")
        succ[source].add(target)
        pred[target].add(source)
    in_degree = {node: len(pred[node]) for node in nodes}
    return succ, pred, in_degree


def toposort(nodes: list[str], edges: list[dict]) -> list[str]:
    """Return a deterministic topological order of ``nodes``.

    Contract:
    - Each edge is ``{"from": source, "to": target}`` and means that
      ``source`` must be emitted before ``target``.  Insertion order in
      ``nodes`` carries no ordering guarantee and must not be used as the
      dependency order.
    - Every edge endpoint must be present in ``nodes``.  Unknown endpoints
      raise :class:`DspError`.
    - A self-loop or any longer directed cycle raises :class:`DspError`.
    - When multiple nodes are ready, emit the lexicographically smallest node
      id first, so the result is deterministic.  An empty graph returns ``[]``.
    - Duplicate node ids in ``nodes`` raise :class:`DspError`.
    """
    nodes = _check_nodes(nodes)
    succ, _, in_degree = _adjacency(nodes, edges)

    ready = [node for node in nodes if in_degree[node] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for nxt in sorted(succ[node]):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                heapq.heappush(ready, nxt)

    if len(order) != len(nodes):
        remaining = sorted(node for node in nodes if in_degree[node] > 0)
        raise DspError(f"cycle detected involving: {', '.join(remaining)}")
    return order


def cycle_nodes(nodes: list[str], edges: list[dict]) -> list[str]:
    """Return every node that participates in at least one directed cycle.

    Contract:
    - The search is graph-wide, not limited to a single level; a back edge
      from a deep node to a much earlier node must be detected.
    - The returned ids are de-duplicated and sorted lexicographically; an
      acyclic or empty graph returns ``[]``.  A self-loop puts that node in the
      result.
    - Unknown endpoints or duplicate node ids raise :class:`DspError`.
    - Edges are evaluated by dependency topology; duplicate edges do not create
      extra cycle nodes.
    """
    nodes = _check_nodes(nodes)
    node_set = set(nodes)
    _check_endpoints(node_set, edges)

    succ = {node: set() for node in nodes}
    in_cycle = set()
    for edge in edges:
        source, target = edge["from"], edge["to"]
        if source == target:
            in_cycle.add(source)
        succ[source].add(target)

    # Tarjan's strongly-connected-components, iterative so deep graphs are
    # safe.  Any SCC with more than one node is a cycle; self-loops were
    # collected above.
    index_of = {}
    lowlink = {}
    on_stack = set()
    stack = []
    counter = [0]

    for root in nodes:
        if root in index_of:
            continue
        work = [(root, iter(sorted(succ[root])))]
        index_of[root] = lowlink[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, it = work[-1]
            advanced = False
            for nxt in it:
                if nxt not in index_of:
                    index_of[nxt] = lowlink[nxt] = counter[0]
                    counter[0] += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, iter(sorted(succ[nxt]))))
                    advanced = True
                    break
                if nxt in on_stack:
                    lowlink[node] = min(lowlink[node], index_of[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == index_of[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    in_cycle.update(component)

    return sorted(in_cycle)


def buffer_plan(order: list[str], edges: list[dict], block_size: int) -> dict:
    """Plan the per-node sample buffers needed to process one block.

    Contract:
    - ``order`` is a complete topological order of the graph's nodes.
    - ``block_size`` is the number of samples (frames) handled per block.  It
      must be a positive integer; ``0``, negative values, and non-integers
      raise :class:`DspError` instead of causing a division by zero later.
    - Buffers are allocated from the block size, never from the node count, so
      a large block cannot overflow a buffer sized for one sample per node.
    - A producer's output is read by all of its consumers.  To keep shared
      fan-out from aliasing, each node owns one output block plus one
      independent queue per outgoing edge.  The returned mapping is
      ``{node: block_size * (1 + out_degree(node))}`` in samples.
    - Every edge endpoint must appear exactly once in ``order``; otherwise
      raise :class:`DspError`.
    """
    if not _is_positive_int(block_size):
        raise DspError(f"block_size must be a positive integer, got {block_size!r}")

    order = list(order)
    if len(set(order)) != len(order):
        raise DspError("order contains duplicate node ids")
    node_set = set(order)
    _check_endpoints(node_set, edges)

    out_degree = {node: 0 for node in order}
    seen_edges = set()
    for edge in edges:
        pair = (edge["from"], edge["to"])
        if pair in seen_edges:
            continue
        seen_edges.add(pair)
        out_degree[edge["from"]] += 1

    return {node: block_size * (1 + out_degree[node]) for node in order}


def latency_compensate(order: list[str], delays: dict) -> dict:
    """Return the extra delay, in samples, needed to align parallel branches.

    Contract:
    - Each entry of ``delays`` is one parallel branch's measured total latency
      in samples (a non-negative integer).  Missing nodes default to ``0``.
    - The slowest branch sets the target: every branch is padded to
      ``max(delays.values())``.  The returned mapping is
      ``{node: max_latency - own_latency}`` for each node in ``order``.
    - Consequently ``delays[node] + result[node]`` equals the maximum branch
      latency for every node (within ``1e-6`` for numeric callers).
    - Unknown node ids, negative delays, and non-integer delays raise
      :class:`DspError`.  An empty graph returns ``{}``.
    """
    order = list(order)
    node_set = set(order)
    own = {}
    for node, delay in delays.items():
        if node not in node_set:
            raise DspError(f"unknown endpoint: {node}")
        if not isinstance(delay, int) or isinstance(delay, bool):
            raise DspError(f"delay for {node} must be an integer, got {delay!r}")
        if delay < 0:
            raise DspError(f"delay for {node} must be non-negative, got {delay}")
        own[node] = delay

    if not order:
        return {}
    target = max(own.values(), default=0)
    return {node: target - own.get(node, 0) for node in order}


def process_block(graph: dict, inputs: dict, state: dict) -> dict:
    """Process one block through ``graph`` while carrying state forward.

    Contract:
    - ``graph`` uses the ``{"nodes": [...], "edges": [...]}`` shape.
    - ``inputs`` maps node id to a sequence of freshly injected samples for
      this block.  Unknown nodes raise :class:`DspError`.
    - The block size comes from a positive integer ``graph["block_size"]``
      when present; otherwise it is inferred from the longest input sequence.
      All non-empty inputs must have exactly that length.  Zero (including an
      inferred zero), negative, or non-integer block sizes raise
      :class:`DspError` rather than dividing by zero.
    - Nodes are evaluated in topological order.  A node's output at sample
      index ``i`` is its injected input at ``i`` plus the sample-wise sum of
      all direct predecessor outputs at ``i``.  A predecessor output is
      read-only and may feed several successors without being consumed.
      Every node emits exactly ``block_size`` samples, so output sample counts
      match the input block and no frames are dropped.
    - ``state`` maps a node to its accumulated output history from earlier
      blocks.  The returned mapping has the same shape plus this block's
      outputs appended; prior entries are preserved, never reset, so adjacent
      blocks remain continuous.  Callers must not share one state mapping
      between independent graphs.
    - Returns ``{"outputs": {node: [float, ...]},
      "state": {node: [float, ...]}}``.  Comparison tolerance is ``1e-6``.
    """
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    order = toposort(nodes, edges)
    node_set = set(nodes)

    inputs = inputs or {}
    for node in inputs:
        if node not in node_set:
            raise DspError(f"unknown endpoint: {node}")
    injected = {node: [float(s) for s in seq] for node, seq in inputs.items()}

    if "block_size" in graph:
        block_size = graph["block_size"]
        if not _is_positive_int(block_size):
            raise DspError(
                f"block_size must be a positive integer, got {block_size!r}"
            )
    else:
        block_size = max((len(seq) for seq in injected.values()), default=0)
        if block_size <= 0:
            raise DspError("inferred block size is zero; nothing to process")

    for node, seq in injected.items():
        if len(seq) not in (0, block_size):
            raise DspError(
                f"input for {node} has {len(seq)} samples, expected {block_size}"
            )

    _, pred, _ = _adjacency(_check_nodes(nodes), edges)

    outputs = {}
    for node in order:
        source = injected.get(node) or [0.0] * block_size
        preds = sorted(pred[node])
        if not preds:
            outputs[node] = list(source)
            continue
        block = list(source)
        for upstream in preds:
            upstream_out = outputs[upstream]
            for i in range(block_size):
                block[i] += upstream_out[i]
        outputs[node] = block

    new_state = {node: list(history) for node, history in (state or {}).items()}
    for node in order:
        new_state.setdefault(node, []).extend(outputs[node])

    return {"outputs": outputs, "state": new_state}


def rebuild(graph: dict, changed: set[str]) -> dict:
    """Incrementally rebuild a graph after ``changed`` nodes are modified.

    Contract:
    - ``changed`` is an iterable of node ids that changed.  Unknown ids raise
      :class:`DspError`.
    - The affected set is ``changed`` plus every node reachable downstream
      from it.  Only that subgraph is re-toposorted; unaffected nodes keep
      their previous relative order and positions, so an unrelated graph is
      not rebuilt in full.
    - A previous order is read from ``graph["order"]`` when present, otherwise
      from ``graph["nodes"]``.  The input graph must be acyclic; an invalid
      graph raises :class:`DspError`.
    - Returns ``{"nodes": [...], "edges": [...], "order": [...],
      "dirty": [...]}`` where ``dirty`` is the sorted affected set and
      ``order`` is a valid topological order of the entire graph.
    """
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    node_set = set(nodes)

    # Validates duplicates, unknown endpoints, self-loops and cycles.
    full_order = toposort(nodes, edges)

    changed = set(changed)
    for node in changed:
        if node not in node_set:
            raise DspError(f"unknown endpoint: {node}")

    succ, _, _ = _adjacency(nodes, edges)
    affected = set(changed)
    stack = list(changed)
    while stack:
        node = stack.pop()
        for nxt in succ[node]:
            if nxt not in affected:
                affected.add(nxt)
                stack.append(nxt)

    previous = list(graph.get("order", nodes))
    if set(previous) != node_set or len(previous) != len(nodes):
        previous = full_order

    # Re-toposort only the affected subgraph, then splice it back into the
    # slots the affected nodes occupied before, leaving every unaffected node
    # at its previous position.
    sub_edges = [
        edge for edge in edges
        if edge["from"] in affected and edge["to"] in affected
    ]
    affected_order = toposort(sorted(affected), sub_edges)

    new_order = []
    affected_iter = iter(affected_order)
    for node in previous:
        if node in affected:
            new_order.append(next(affected_iter))
        else:
            new_order.append(node)

    # The splice is valid whenever unaffected predecessors/successors still
    # bracket the affected slots; if an unusual layout breaks that, fall back
    # to a full deterministic sort so the result is always a valid order.
    position = {node: index for index, node in enumerate(new_order)}
    for edge in edges:
        if position[edge["from"]] >= position[edge["to"]]:
            new_order = full_order
            break

    return {
        "nodes": nodes,
        "edges": edges,
        "order": new_order,
        "dirty": sorted(affected),
    }


def validate_graph(nodes: list[str], edges: list[dict]) -> list[str]:
    """Return a deterministic list of structural graph issues.

    Contract:
    - This function reports problems instead of raising.  ``[]`` means the
      graph is structurally valid.
    - Unknown endpoints are reported once per missing id as
      ``"unknown endpoint: <id>"``.
    - Self-loops are reported once per node as ``"self loop: <id>"``.
    - Repeated directed edges after their first occurrence are reported as
      ``"duplicate edge: <from>-><to>"``.
    - All three groups are sorted lexicographically and concatenated in the
      order above, so callers can compare issue lists deterministically.
    """
    node_set = set(nodes)
    unknown = set()
    self_loops = set()
    duplicates = []
    seen_edges = set()

    for edge in edges:
        source, target = edge["from"], edge["to"]
        if source not in node_set:
            unknown.add(source)
        if target not in node_set:
            unknown.add(target)
        if source == target:
            self_loops.add(source)
        pair = (source, target)
        if pair in seen_edges:
            duplicates.append(f"duplicate edge: {source}->{target}")
        else:
            seen_edges.add(pair)

    issues = [f"unknown endpoint: {node}" for node in sorted(unknown)]
    issues += [f"self loop: {node}" for node in sorted(self_loops)]
    issues += sorted(duplicates)
    return issues


def graph_batch(graphs: list[dict], config: dict) -> dict:
    """Process a batch of graphs atomically and preserve their input order.

    Contract:
    - ``config`` requires a positive integer ``block_size``.  Optional
      ``inputs`` and ``state`` sequences supply one mapping per graph; omitted
      entries default to ``{}``.  Length mismatches, a zero/negative block
      size, or invalid configuration raise :class:`DspError`.
    - Every graph is validated before any graph is processed.  If any graph is
      invalid (for example unknown endpoints, a self-loop, a cycle, or a bad
      block size), the whole batch raises :class:`DspError` and no partial
      results are returned.  This is the rollback boundary.
    - Each graph receives its own state; state is never shared or reused
      between graphs, even when two graphs have identical node ids.  Results
      are returned in exactly the same order as ``graphs``.
    - Returns ``{"results": [{"order": [...], "outputs": {...},
      "state": {...}}, ...], "block_size": int}``.  Floating-point values are
      compared with tolerance ``1e-6``.
    """
    if not isinstance(config, dict):
        raise DspError("config must be a mapping")
    block_size = config.get("block_size")
    if not _is_positive_int(block_size):
        raise DspError(f"block_size must be a positive integer, got {block_size!r}")

    graphs = list(graphs)
    count = len(graphs)

    def _per_graph(key):
        seq = config.get(key)
        if seq is None:
            return [{} for _ in graphs]
        seq = list(seq)
        if len(seq) != count:
            raise DspError(
                f"config[{key!r}] has {len(seq)} entries for {count} graphs"
            )
        return seq

    batch_inputs = _per_graph("inputs")
    batch_states = _per_graph("state")

    # Validate every graph before any processing: this is the rollback
    # boundary, so one bad graph rejects the whole batch.
    orders = []
    for graph in graphs:
        nodes = list(graph.get("nodes", []))
        edges = list(graph.get("edges", []))
        issues = validate_graph(nodes, edges)
        if issues:
            raise DspError(f"invalid graph: {'; '.join(issues)}")
        orders.append(toposort(nodes, edges))

    results = []
    for graph, order, inputs, graph_state in zip(
        graphs, orders, batch_inputs, batch_states
    ):
        # Each graph gets a private copy of its state so nothing is shared
        # between graphs, even when node ids collide.
        private_state = {
            node: list(history) for node, history in (graph_state or {}).items()
        }
        run_graph = {
            "nodes": list(graph.get("nodes", [])),
            "edges": list(graph.get("edges", [])),
            "block_size": block_size,
        }
        processed = process_block(run_graph, inputs or {}, private_state)
        results.append(
            {
                "order": order,
                "outputs": processed["outputs"],
                "state": processed["state"],
            }
        )

    return {"results": results, "block_size": block_size}
