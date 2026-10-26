# -*- coding: utf-8 -*-
"""Operator fusion speedup: fewer kernels => less launch overhead.

Measures the kernel-count reduction the fusion pass achieves on a graph full of
the canonical ``matmul -> bias -> act`` pattern. Deterministic (a compile-time
rewrite), so the speedup is a fixed, reproducible number.
"""
from ..bench.registry import register_benchmark
from ..ir.builder import GraphBuilder
from ..passes.fusion import FusionPass
from ..passes.pipeline import default_pipeline


def _fusion_heavy_graph(n_patterns=8):
    b = GraphBuilder("fusion_heavy")
    x = b.input("x", shape=[1, 16])
    h = x
    for i in range(n_patterns):
        w = b.const(i + 1.0)
        m = b.matmul(h, w)
        s = b.add(m, w)
        h = b.relu(s)
    b.output(h)
    return b.build()


@register_benchmark("fusion")
def run_fusion(cfg=None):
    graph = _fusion_heavy_graph()
    before = graph.op_count()
    FusionPass().run(graph)
    after = graph.op_count()
    return {
        "ops_before": before,
        "ops_after": after,
        "speedup": round(before / max(after, 1), 3),
        "kernels_removed": before - after,
    }
