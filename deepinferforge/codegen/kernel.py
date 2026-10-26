# -*- coding: utf-8 -*-
"""A kernel spec: one IR node lowered to one launchable kernel."""


class KernelSpec(object):
    __slots__ = ("name", "op", "node", "backend", "config", "source")

    def __init__(self, name, op, node, backend, config, source):
        self.name = name
        self.op = op
        self.node = node
        self.backend = backend
        self.config = config
        self.source = source

    def __repr__(self):
        return "KernelSpec(%s, op=%s, backend=%s)" % (self.name, self.op, self.backend)

    def to_dict(self):
        return {
            "name": self.name, "op": self.op, "node": self.node,
            "backend": self.backend, "config": dict(self.config),
        }
