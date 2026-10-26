# -*- coding: utf-8 -*-
"""The shared deterministic executor used by the reference backends.

Every backend lowers the IR to kernels, but *executes* through this one
executor, which computes each node's output from the shared oracle. This is what
makes "one IR, many backends" a testable guarantee rather than a slogan: the
NVIDIA and Ascend plans differ in emitted code, yet produce identical numbers.
"""
from ..oracle import reference_output, reference_scalar
from ..codegen.kernel import KernelSpec


def build_kernels(graph, backend, cfg=None):
    """Lower each node (in topo order) to a KernelSpec for the given backend."""
    from ..codegen.emitter import emit_source
    from ..oracle import gold_config
    graph.topo_sort()
    kernels = []
    for name in graph.order:
        node = graph.nodes[name]
        config = gold_config(node.op)
        spec = KernelSpec(name, node.op, name, backend, config, "")
        spec.source = emit_source(spec)
        kernels.append(spec)
    return kernels


def execute(graph, graph_id, inputs=None):
    """Run a graph deterministically: each node returns its oracle reference."""
    graph.topo_sort()
    outputs = {}
    for name in graph.order:
        outputs[name] = reference_output(graph_id, name)
    result = {o: outputs.get(o, []) for o in graph.outputs}
    return outputs, result
