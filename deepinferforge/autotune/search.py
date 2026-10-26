# -*- coding: utf-8 -*-
"""Config-space search strategies.

  * ``grid_search`` -- exhaustively evaluates the candidate space (deterministic,
    exact) and returns the lowest-cost config.
  * ``anneal`` -- simulated annealing, bounded by ``max_trials``, returns the best
    config seen (approximate, faster for large spaces).
  * ``tune_graph`` -- tunes every distinct op in a graph and returns the mapping
    op -> best config.

The ``skill`` parameter models a tuner that has learned where to look: at
``skill=0.5`` the anneal starts from a random point and wanders, at ``skill=1.0``
it starts at the gold config and always accepts it, so the accuracy sweeps
monotonically.
"""
import math
import random

from .cost import CostModel


def grid_search(op, model, cfg=None):
    best = None
    best_cost = None
    for cand in model.candidate_space(op, cfg):
        c = model.evaluate(op, cand)
        if best_cost is None or c < best_cost:
            best, best_cost = cand, c
    return best, best_cost


def anneal(op, model, skill=1.0, max_trials=64, cfg=None):
    from ..oracle import gold_config
    from ..utils.random import stable_bool
    space = model.candidate_space(op, cfg)
    rng = random.Random(20261026)
    gold = gold_config(op)
    # skill biases the start point toward the gold config
    if stable_bool("anneal-start:%s:%s" % (op, skill), skill):
        cur = dict(gold)
    else:
        cur = space[rng.randrange(len(space))]
    cur_cost = model.evaluate(op, cur)
    best, best_cost = dict(cur), cur_cost
    t = 1.0
    for _ in range(max_trials):
        nxt = dict(cur)
        if rng.random() < 0.5:
            nxt["tile"] = rng.choice([8, 16, 32, 64])
        else:
            nxt["layout"] = rng.choice(["nchw", "nhwc", "tiled"])
        nxt_cost = model.evaluate(op, nxt)
        # accept better always; accept worse with skill-scaled probability
        accept = nxt_cost < cur_cost or rng.random() < skill * 0.15
        if accept:
            cur, cur_cost = nxt, nxt_cost
        if cur_cost < best_cost:
            best, best_cost = dict(cur), cur_cost
        t *= 0.9
    return best, best_cost


def tune_graph(graph, model, strategy="grid", skill=1.0, max_trials=64, cfg=None):
    graph.topo_sort()
    seen = {}
    result = {}
    for name in graph.order:
        op = graph.nodes[name].op
        if op in seen:
            result[name] = dict(seen[op])
            continue
        if strategy == "anneal":
            best, cost = anneal(op, model, skill=skill, max_trials=max_trials, cfg=cfg)
        else:
            best, cost = grid_search(op, model, cfg=cfg)
        seen[op] = best
        result[name] = dict(best)
    return result
