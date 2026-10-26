# -*- coding: utf-8 -*-
"""CPU reference backend: the golden executor with no kernel emission.

The CPU backend is the ground truth the GPU/Ascend backends are validated
against -- it runs the same IR and returns the same oracle outputs, but skips
kernel code generation entirely.
"""
from .registry import register_backend
from .base import Backend, CompiledPlan
from .ops import execute


@register_backend("cpu")
class CpuBackend(Backend):
    backend_name = "cpu"

    def compile(self, graph, cfg=None):
        from ..passes.pipeline import default_pipeline
        pm = default_pipeline(cfg)
        report = pm.run(graph)
        return CompiledPlan(graph, [], "cpu", report)

    def run(self, plan, inputs=None):
        outputs, result = execute(plan.graph, plan.graph.name, inputs)
        return result
