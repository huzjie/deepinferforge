# -*- coding: utf-8 -*-
"""Benchmarks: measure the properties a unified inference stack must have.

Explicit imports are required so the ``@register_benchmark`` decorators run.
"""
from .registry import register_benchmark, list_benchmarks, get_benchmark
from .runner import run_all, summarize
from . import consistency, fusion, quant, speculative, autotune

__all__ = [
    "register_benchmark", "list_benchmarks", "get_benchmark",
    "run_all", "summarize",
]
