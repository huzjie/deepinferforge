# -*- coding: utf-8 -*-
"""A single IR node: an operator applied to named inputs, producing one output."""


class Node(object):
    __slots__ = ("name", "op", "inputs", "attrs", "shape", "dtype")

    def __init__(self, name, op, inputs=None, attrs=None, shape=None, dtype=None):
        self.name = name
        self.op = op
        self.inputs = list(inputs or [])
        self.attrs = dict(attrs or {})
        self.shape = list(shape or [])
        self.dtype = dtype

    def __repr__(self):
        return "Node(%s, op=%s, inputs=%s)" % (self.name, self.op, self.inputs)

    def to_dict(self):
        return {
            "name": self.name, "op": self.op, "inputs": list(self.inputs),
            "attrs": dict(self.attrs), "shape": list(self.shape),
            "dtype": getattr(self.dtype, "name", None),
        }
