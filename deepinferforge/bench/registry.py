# -*- coding: utf-8 -*-
"""Benchmark registry."""
_BENCHMARKS = {}


def register_benchmark(name):
    def deco(fn):
        _BENCHMARKS[name] = fn
        fn.benchmark_name = name
        return fn
    return deco


def list_benchmarks():
    return sorted(_BENCHMARKS)


def get_benchmark(name):
    return _BENCHMARKS[name]
