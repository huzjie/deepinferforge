# -*- coding: utf-8 -*-
"""Ascend backend: lowers the IR to CANN-style kernels.

The second headline target. Same IR, same pass pipeline, different emitted
kernel source (CANN ``__aicore__``), and -- via the shared oracle executor --
bit-identical results to NVIDIA. This is the core DeepInfer promise: one graph,
two accelerator families, no per-backend drift.
"""
from .registry import register_backend
from .base import Backend, CompiledPlan
from .ops import build_kernels, execute


@register_backend("ascend")
class AscendBackend(Backend):
    backend_name = "ascend"

    def compile(self, graph, cfg=None):
        from ..passes.pipeline import default_pipeline
        pm = default_pipeline(cfg)
        report = pm.run(graph)
        kernels = build_kernels(graph, "ascend", cfg)
        return CompiledPlan(graph, kernels, "ascend", report)

    def run(self, plan, inputs=None):
        outputs, result = execute(plan.graph, plan.graph.name, inputs)
        return result
