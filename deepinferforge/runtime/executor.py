# -*- coding: utf-8 -*-
"""Execute a :class:`CompiledPlan` and collect the graph outputs.

The executor walks the plan's kernels in launch order and runs each through the
backend's execution path. For the reference backends that means the shared
oracle executor; for a production engine this is where the real kernels would
be dispatched to streams / command queues.
"""


class Executor(object):
    def __init__(self, backend):
        self.backend = backend

    def run(self, plan, inputs=None):
        return self.backend.run(plan, inputs)

    def run_graph(self, graph, backend_name, cfg=None):
        from ..backends.registry import get_backend
        cls = get_backend(backend_name)
        backend = cls()
        plan = backend.compile(graph, cfg)
        return plan, backend.run(plan)
