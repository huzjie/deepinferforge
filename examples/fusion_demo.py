# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Show operator fusion shrinking a graph (kernel-count reduction)."""
from deepinferforge.ir.builder import GraphBuilder
from deepinferforge.passes.fusion import FusionPass
from deepinferforge.passes.pipeline import default_pipeline

b = GraphBuilder("fusion_demo")
x = b.input("x")
h = x
for i in range(8):
    w = b.const(i + 1.0)
    h = b.relu(b.add(b.matmul(h, w), w))
b.output(h)
g = b.build()

before = g.op_count()
report = default_pipeline().run(g)
after = g.op_count()
print("ops before=%d  after=%d  fusion_speedup=%.2fx" % (before, after, before / after))
print("pass report:", report)
