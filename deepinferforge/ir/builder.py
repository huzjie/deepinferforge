# -*- coding: utf-8 -*-
"""A convenient builder that lowers tensor ops into IR nodes.

The builder auto-names nodes, wires dataflow edges, and returns node names so
the model frontend can express a Transformer as a chain of ``matmul`` /
``layernorm`` / ``softmax`` / ``gelu`` calls instead of hand-building nodes.
"""
import itertools

from .node import Node
from .graph import Graph
from .dtypes import from_name, float32


class GraphBuilder(object):
    def __init__(self, name="graph", dtype="float32"):
        self.graph = Graph(name)
        self.dtype = from_name(dtype)
        self._counter = itertools.count()

    def _new_name(self, op):
        return "%s_%d" % (op, next(self._counter))

    def input(self, name, shape=None, dtype=None):
        self.graph.mark_input(name)
        return name

    def output(self, name):
        self.graph.mark_output(name)
        return name

    def op(self, op_name, inputs, attrs=None, shape=None, dtype=None):
        name = self._new_name(op_name)
        node = Node(name, op_name, inputs, attrs, shape,
                    from_name(dtype) if dtype else self.dtype)
        self.graph.add_node(node)
        return name

    # -- common tensor ops -------------------------------------------------
    def const(self, value, dtype=None):
        return self.op("const", [], {"value": value}, dtype=dtype)

    def matmul(self, a, b):
        return self.op("matmul", [a, b])

    def add(self, a, b):
        return self.op("add", [a, b])

    def mul(self, a, b):
        return self.op("mul", [a, b])

    def relu(self, a):
        return self.op("relu", [a])

    def gelu(self, a):
        return self.op("gelu", [a])

    def softmax(self, a, axis=-1):
        return self.op("softmax", [a], {"axis": axis})

    def layernorm(self, a, weight=None, bias=None):
        ins = [a] + ([weight] if weight else []) + ([bias] if bias else [])
        return self.op("layernorm", ins)

    def reshape(self, a, shape):
        return self.op("reshape", [a], {"shape": list(shape)})

    def transpose(self, a, perm):
        return self.op("transpose", [a], {"perm": list(perm)})

    def attention(self, q, k, v, mask=None):
        ins = [q, k, v] + ([mask] if mask else [])
        return self.op("attention", ins)

    def moe(self, x, n_experts, top_k):
        return self.op("moe", [x], {"n_experts": n_experts, "top_k": top_k})

    def rmsnorm(self, a, weight=None):
        ins = [a] + ([weight] if weight else [])
        return self.op("rmsnorm", ins)

    def rope(self, a):
        return self.op("rope", [a])

    def swiglu(self, a, b):
        return self.op("swiglu", [a, b])

    def cast(self, a, dtype):
        return self.op("cast", [a], {"dtype": dtype}, dtype=dtype)

    def build(self):
        self.graph.topo_sort()
        return self.graph
