# -*- coding: utf-8 -*-
"""Autotuning accuracy: does the tuner pick the optimal kernel config?

Models a tuner of ``skill`` that picks the gold config with probability ``skill``
and a uniformly random (wrong) config otherwise. Accuracy therefore equals
``skill`` in expectation -- a clean, monotonic signal from a random baseline to
the perfect tuner. The real search primitives (grid / anneal) are exercised in
the examples and doctor, where grid search provably returns the gold config.
"""
from ..bench.registry import register_benchmark
from ..utils.random import stable_bool, stable_int, stable_choice
from ..oracle import gold_config


def _wrong_config(op):
    return {"tile": stable_choice("wrong-tile:" + op, [8, 16, 32, 64]),
            "layout": stable_choice("wrong-layout:" + op, ["nchw", "nhwc", "tiled"]),
            "unroll": stable_int("wrong-unroll:" + op, 1, 8)}


@register_benchmark("autotune")
def run_autotune(skill=1.0, n_ops=32, cfg=None):
    ops = ["matmul", "softmax", "layernorm", "attention", "gelu", "add", "moe"]
    correct = 0
    for i in range(n_ops):
        op = ops[i % len(ops)]
        gold = gold_config(op)
        if stable_bool("autotune-pick:%s:%d:%.3f" % (op, i, skill), skill):
            picked = gold
        else:
            picked = _wrong_config(op)
        correct += (picked == gold)
    return {"accuracy": round(correct / n_ops, 4), "n_ops": n_ops}
