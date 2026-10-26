# -*- coding: utf-8 -*-
"""Single source of truth shared by mock backends and benchmarks.

The oracle answers two questions deterministically:

  * ``reference_output(graph_id, node)`` -- the "correct" output tensor for a
    node in a graph, used by every backend so cross-backend consistency is a
    *guarantee* (all backends run the same IR against the same reference).

  * ``gold_config(op)`` -- the optimal kernel config for an operator, used by
    the autotune benchmark so that ``skill=1.0`` picks the gold config exactly.

Keeping the truth in one place (instead of re-deriving it per backend) is what
makes the "one IR, many backends" property testable: it removes the usual
per-backend drift that plagues cross-platform stacks.
"""
from .utils.random import stable_float, stable_int, stable_vector, stable_choice


def node_key(graph_id, node):
    return "%s::%s" % (graph_id, node)


def reference_output(graph_id, node, dim=8):
    """Deterministic reference tensor for a node (length ``dim``)."""
    return stable_vector("ref:%s" % node_key(graph_id, node), dim, -1.0, 1.0)


def reference_scalar(graph_id, node):
    return stable_float("ref-scalar:%s" % node_key(graph_id, node), -1.0, 1.0)


def gold_config(op):
    """Optimal kernel config for an operator (tile size / layout / unroll)."""
    # keyed only by op so it is stable across trials, but a *function* of op so
    # that different operators have genuinely different optima.
    tile = (8, 16, 32, 64)[stable_int("gold-tile:" + op, 0, 4)]
    layout = stable_choice("gold-layout:" + op, ["nchw", "nhwc", "tiled"])
    unroll = stable_int("gold-unroll:" + op, 1, 8)
    return {"tile": tile, "layout": layout, "unroll": unroll}


def op_cost(op, tile, layout, unroll):
    """Deterministic latency proxy (lower is better) for a candidate config.

    The gold config is always the minimum by construction: candidate cost equals
    a base plus a penalty that is zero only when the candidate matches the gold
    config. This keeps the autotune objective well-posed and skill-monotonic.
    """
    gold = gold_config(op)
    base = 100.0 + stable_float("cost-base:" + op, 0.0, 50.0)
    penalty = 0.0
    if tile != gold["tile"]:
        penalty += 40.0
    if layout != gold["layout"]:
        penalty += 30.0
    if unroll != gold["unroll"]:
        penalty += 15.0
    return base + penalty
