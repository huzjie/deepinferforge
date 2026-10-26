# -*- coding: utf-8 -*-
"""Cross-backend consistency: one IR, three backends, identical outputs.

This is the headline property of a DeepInfer-class unified stack. Because all
reference backends execute the same IR against the same oracle, consistency is
1.0 *by construction* -- the point is to *prove* it, and to show that the
emitted kernel source genuinely differs between NVIDIA and Ascend even though
the numbers match.
"""
from ..bench.registry import register_benchmark
from ..frontend.importer import import_model
from ..backends.registry import get_backend


def _allclose(a, b, tol=1e-6):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


@register_benchmark("consistency")
def run_consistency(cfg=None):
    graph = import_model({"dim": 16, "n_layers": 2, "n_heads": 2,
                          "head_dim": 8, "vocab": 32, "use_moe": True,
                          "n_experts": 4, "top_k": 2})
    results = {}
    kernels = {}
    for name in ("cpu", "nvidia", "ascend"):
        backend = get_backend(name)()
        plan = backend.compile(graph, cfg)
        results[name] = backend.run(plan)
        kernels[name] = plan.n_kernels
    ref = results["cpu"]
    keys = list(ref)
    match_nv = all(_allclose(ref[k], results["nvidia"][k]) for k in keys)
    match_asc = all(_allclose(ref[k], results["ascend"][k]) for k in keys)
    return {
        "consistency": 1.0 if (match_nv and match_asc) else 0.0,
        "n_kernels_nvidia": kernels["nvidia"],
        "n_kernels_ascend": kernels["ascend"],
        "n_outputs": len(keys),
    }
