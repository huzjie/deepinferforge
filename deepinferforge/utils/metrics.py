# -*- coding: utf-8 -*-
"""Numeric helpers for benchmarking and estimation."""
import math


def mean(xs):
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def variance(xs):
    if not xs:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def std(xs):
    return math.sqrt(variance(xs))


def normalize(xs, eps=1e-9):
    m = mean(xs)
    s = std(xs)
    if s < eps:
        return [0.0 for _ in xs]
    return [(x - m) / s for x in xs]


def percentile(xs, p=90.0):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = int(math.ceil(p / 100.0 * len(s))) - 1
    k = max(0, min(k, len(s) - 1))
    return s[k]


def cosine(a, b, eps=1e-9):
    if not a or not b:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    if da < eps or db < eps:
        return 0.0
    return num / (da * db)
