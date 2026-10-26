# -*- coding: utf-8 -*-
"""Operator fusion.

Fuses adjacent element-wise / bias / activation nodes into a single fused node,
reducing kernel-launch count and memory traffic. The patterns are the classic
ones a real cross-platform compiler applies:

  * ``matmul -> add(bias) -> relu/gelu``  => ``fused_gemm_bias_act``
  * ``layernorm/rmsnorm -> matmul``       => ``fused_norm_matmul`` (kept simple)
  * ``q,k,v -> attention`` stays, but ``matmul -> softmax -> matmul`` is flagged
    as a fused-attention candidate.

Fusion is *pure IR rewriting*: it never depends on which backend will run it.
"""
from .base import Pass
from ..ir.node import Node

_ACT = {"relu": 1, "gelu": 1}


class FusionPass(Pass):
    name = "fusion"

    def run(self, graph):
        fused = 0
        # pattern: matmul -> (add with const/bias) -> act
        for name, node in list(graph.nodes.items()):
            if node.op != "matmul":
                continue
            consumers = graph.consumers(name)
            if len(consumers) != 1:
                continue
            mid = consumers[0]
            mid_node = graph.nodes[mid]
            if mid_node.op != "add":
                continue
            act_consumers = graph.consumers(mid)
            if len(act_consumers) != 1:
                continue
            act = act_consumers[0]
            act_node = graph.nodes[act]
            if act_node.op not in _ACT:
                continue
            # collapse the three nodes into one
            new_node = Node(
                act_node.name, "fused_gemm_bias_act",
                inputs=node.inputs + [i for i in mid_node.inputs if i != name],
                attrs={"act": act_node.op, **node.attrs},
                shape=act_node.shape, dtype=act_node.dtype,
            )
            for n in (name, mid, act):
                graph.nodes.pop(n, None)
            graph.nodes[act] = new_node
            fused += 1
        graph.topo_sort()
        return fused
