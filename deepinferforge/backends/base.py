# -*- coding: utf-8 -*-
"""Backend abstraction + the compiled plan it returns."""


class CompiledPlan(object):
    """A lowered graph: an ordered list of kernels ready to launch."""

    def __init__(self, graph, kernels, backend, pass_report=None):
        self.graph = graph
        self.kernels = kernels       # list of KernelSpec in launch order
        self.backend = backend
        self.pass_report = pass_report or {}

    @property
    def n_kernels(self):
        return len(self.kernels)

    def to_dict(self):
        return {
            "backend": self.backend,
            "n_kernels": self.n_kernels,
            "kernels": [k.to_dict() for k in self.kernels],
            "pass_report": dict(self.pass_report),
        }


class Backend(object):
    backend_name = "base"

    def compile(self, graph, cfg=None):
        raise NotImplementedError

    def run(self, plan, inputs=None):
        raise NotImplementedError
