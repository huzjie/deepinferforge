# -*- coding: utf-8 -*-
"""NVIDIA backend: lowers the IR to CUDA-style kernels.

One of the two headline targets of a DeepInfer-style stack. Compilation runs
the shared pass pipeline, then emits a ``__global__`` kernel per node; execution
goes through the shared oracle executor so results are bit-identical to Ascend.
"""
from .registry import register_backend
from .base import Backend, CompiledPlan
from .ops import build_kernels, execute


@register_backend("nvidia")
class NvidiaBackend(Backend):
    backend_name = "nvidia"

    def compile(self, graph, cfg=None):
        from ..passes.pipeline import default_pipeline
        pm = default_pipeline(cfg)
        report = pm.run(graph)
        kernels = build_kernels(graph, "nvidia", cfg)
        return CompiledPlan(graph, kernels, "nvidia", report)

    def run(self, plan, inputs=None):
        outputs, result = execute(plan.graph, plan.graph.name, inputs)
        return result
