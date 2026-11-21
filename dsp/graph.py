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
downstream consumer.  All functions in this module are intentionally
unimplemented: the exercise is to implement the documented contracts without
changing the public signatures or the tests.

Floating-point results are compared with an absolute tolerance of ``1e-6``.
"""

from __future__ import annotations

import heapq
from collections import defaultdict, deque

TOLERANCE = 1e-6


class DspError(Exception):
    """Raised when a DSP graph or a processing request is invalid."""


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
