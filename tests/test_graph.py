"""Contract tests for the audio DSP graph.

There are exactly 15 tests: one for each known defect plus five focused
edge-case combinations.  Every test calls a public graph stub; the suite is
expected to report ``15 failed`` until ``dsp/graph.py`` is implemented.
"""

import pytest

from dsp.graph import (
    DspError,
    buffer_plan,
    cycle_nodes,
    graph_batch,
    latency_compensate,
    process_block,
    rebuild,
    toposort,
    validate_graph,
)


def test_toposort_respects_dependencies():
    nodes = ["mix", "delay", "input", "reverb"]
    edges = [
        {"from": "input", "to": "delay"},
        {"from": "input", "to": "reverb"},
        {"from": "delay", "to": "mix"},
        {"from": "reverb", "to": "mix"},
    ]

    order = toposort(nodes, edges)
    position = {node: index for index, node in enumerate(order)}

    assert set(order) == set(nodes)
    assert position["input"] < position["delay"] < position["mix"]
    assert position["input"] < position["reverb"] < position["mix"]


def test_cycle_detection_finds_cross_level_cycle():
    nodes = ["input", "eq", "comp", "limit", "mix", "send"]
    edges = [
        {"from": "input", "to": "eq"},
        {"from": "eq", "to": "comp"},
        {"from": "comp", "to": "limit"},
        {"from": "limit", "to": "mix"},
        {"from": "mix", "to": "send"},
        {"from": "send", "to": "eq"},
    ]

    cycle = cycle_nodes(nodes, edges)

    assert set(cycle) == {"eq", "comp", "limit", "mix", "send"}


def test_buffer_plan_covers_large_blocks_with_fan_out():
    order = ["source", "left", "right", "mix"]
    edges = [
        {"from": "source", "to": "left"},
        {"from": "source", "to": "right"},
        {"from": "left", "to": "mix"},
        {"from": "right", "to": "mix"},
    ]

    plan = buffer_plan(order, edges, 1024)

    assert all(plan[node] >= 1024 for node in order)
    assert plan["source"] == 3 * 1024
    assert plan["left"] == 2 * 1024
    assert plan["mix"] == 1024


def test_latency_compensation_aligns_parallel_branches():
    delays = {"dry": 2, "short": 6, "long": 11}

    compensation = latency_compensate(["dry", "short", "long"], delays)
    target = max(delays.values())

    assert set(compensation) == set(delays)
    for node, own_delay in delays.items():
        assert abs((own_delay + compensation[node]) - target) <= 1e-6
    assert compensation["long"] == 0


def test_process_block_carries_state_between_blocks():
    graph = {
        "nodes": ["input", "delay", "mix"],
        "edges": [
            {"from": "input", "to": "delay"},
            {"from": "input", "to": "mix"},
            {"from": "delay", "to": "mix"},
        ],
        "block_size": 4,
    }

    first = process_block(graph, {"input": [0.25, 0.25, 0.25, 0.25]}, {})
    second = process_block(
        graph,
        {"input": [0.5, 0.5, 0.5, 0.5]},
        first["state"],
    )

    assert first["state"] != {}
    assert len(second["state"]["input"]) == 8
    assert len(second["outputs"]["mix"]) == 4


def test_rebuild_reorders_only_affected_subgraph():
    graph = {
        "nodes": ["a", "b", "c", "d", "noise"],
        "edges": [
            {"from": "a", "to": "b"},
            {"from": "b", "to": "c"},
            {"from": "c", "to": "d"},
        ],
        "order": ["a", "b", "c", "d", "noise"],
    }

    rebuilt = rebuild(graph, {"b"})

    assert rebuilt["dirty"] == ["b", "c", "d"]
    assert rebuilt["order"] == ["a", "b", "c", "d", "noise"]


def test_validate_graph_reports_unknown_endpoint():
    issues = validate_graph(
        ["input", "output"],
        [
            {"from": "input", "to": "output"},
            {"from": "output", "to": "ghost"},
        ],
    )

    assert issues == ["unknown endpoint: ghost"]


def test_validate_graph_reports_self_loop():
    issues = validate_graph(["a", "b"], [{"from": "a", "to": "a"}])

    assert issues == ["self loop: a"]


def test_graph_batch_keeps_state_isolated():
    graphs = [
        {"nodes": ["a"], "edges": []},
        {"nodes": ["a"], "edges": []},
    ]
    config = {
        "block_size": 4,
        "inputs": [
            {"a": [1.0, 2.0, 3.0, 4.0]},
            {"a": [5.0, 6.0, 7.0, 8.0]},
        ],
    }

    result = graph_batch(graphs, config)

    assert result["results"][0]["state"] is not result["results"][1]["state"]
    assert (
        result["results"][0]["outputs"]["a"]
        != result["results"][1]["outputs"]["a"]
    )


def test_buffer_plan_rejects_zero_block_size():
    with pytest.raises(DspError):
        buffer_plan(["a"], [], 0)


def test_single_node_graph_returns_node():
    assert toposort(["only"], []) == ["only"]


def test_chain_graph_order_is_correct():
    nodes = ["d", "b", "a", "c"]
    edges = [
        {"from": "a", "to": "b"},
        {"from": "b", "to": "c"},
        {"from": "c", "to": "d"},
    ]

    assert toposort(nodes, edges) == ["a", "b", "c", "d"]


def test_empty_graph_returns_empty():
    assert toposort([], []) == []


def test_process_block_rejects_zero_block_size():
    graph = {"nodes": ["a"], "edges": []}

    with pytest.raises(DspError):
        process_block(graph, {"a": []}, {})


def test_batch_all_success_preserves_order():
    graphs = [
        {"nodes": ["z"], "edges": []},
        {"nodes": ["a", "b"], "edges": [{"from": "a", "to": "b"}]},
        {"nodes": ["m"], "edges": []},
    ]

    result = graph_batch(graphs, {"block_size": 4})

    assert [entry["order"] for entry in result["results"]] == [
        ["z"],
        ["a", "b"],
        ["m"],
    ]
    assert result["block_size"] == 4
