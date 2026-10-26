# -*- coding: utf-8 -*-
"""Constant folding.

Evaluates pure arithmetic nodes whose inputs are all ``const`` at compile time
and replaces them with a single ``const`` node. This shrinks the graph and, more
importantly, lets later fusion / memory passes see through constant subtrees.
"""
from .base import Pass


class ConstFoldPass(Pass):
    name = "constfold"

    def _eval(self, op, args):
        try:
            if op == "add":
                return args[0] + args[1]
            if op == "mul":
                return args[0] * args[1]
            if op == "sub":
                return args[0] - args[1]
            if op == "div":
                return args[0] / args[1]
        except (TypeError, ZeroDivisionError):
            return None
        return None

    def run(self, graph):
        folded = 0
        changed = True
        while changed:
            changed = False
            for name, node in list(graph.nodes.items()):
                if node.op not in ("add", "mul", "sub", "div"):
                    continue
                if not node.inputs:
                    continue
                vals = []
                ok = True
                for inp in node.inputs:
                    if inp in graph.nodes and graph.nodes[inp].op == "const":
                        vals.append(graph.nodes[inp].attrs.get("value"))
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                v = self._eval(node.op, vals)
                if v is None:
                    continue
                node.op = "const"
                node.attrs = {"value": v}
                node.inputs = []
                folded += 1
                changed = True
        graph.topo_sort()
        return folded
