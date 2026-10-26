# -*- coding: utf-8 -*-
"""Dead code elimination.

Removes nodes that are not (transitively) needed to produce the graph outputs.
This is the pass that lets a full 671B-parameter graph ship with only the
subgraph the current request actually needs.
"""
from .base import Pass


class DeadCodeEliminationPass(Pass):
    name = "dce"

    def run(self, graph):
        live = set()
        stack = list(graph.outputs)
        while stack:
            n = stack.pop()
            if n in live or n not in graph.nodes:
                continue
            live.add(n)
            stack.extend(graph.nodes[n].inputs)
        removed = 0
        for name in list(graph.nodes):
            if name not in live:
                graph.nodes.pop(name, None)
                removed += 1
        graph.topo_sort()
        return removed
