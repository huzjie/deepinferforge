# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Compile one model on all backends and prove cross-backend consistency."""
from deepinferforge.frontend.importer import import_model
from deepinferforge.backends.registry import get_backend

model = {"dim": 16, "n_layers": 2, "n_heads": 2, "head_dim": 8,
         "vocab": 32, "use_moe": True, "n_experts": 4, "top_k": 2}
graph = import_model(model)

results = {}
for name in ("cpu", "nvidia", "ascend"):
    plan = get_backend(name)().compile(graph)
    results[name] = get_backend(name)().run(plan)
    print("%-8s kernels=%d pass_report=%s" % (name, plan.n_kernels, plan.pass_report))

ref = results["cpu"]
print("\nconsistency (vs cpu):")
for name in ("nvidia", "ascend"):
    keys = list(ref)
    same = all(ref[k] == results[name][k] for k in keys)
    print("  %s: %s" % (name, "identical" if same else "DIFF"))
print("\nfirst output (cpu):", [round(v, 4) for v in ref[keys[0]]][:8])
