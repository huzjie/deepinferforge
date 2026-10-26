# -*- coding: utf-8 -*-
"""Memory planning.

Uses liveness analysis to compute how many buffers are live at once, and assigns
each intermediate a *reusable* buffer id. Two tensors whose lifetimes do not
overlap share a buffer, which is exactly how real engines cut peak memory. The
pass reports the buffer count before vs. after.
"""
from .base import Pass


class MemoryPlanPass(Pass):
    name = "memory_plan"

    def run(self, graph):
        graph.topo_sort()
        # last use of each value along the topo order
        last_use = {}
        for idx, name in enumerate(graph.order):
            for inp in graph.nodes[name].inputs:
                if inp in graph.nodes:
                    last_use[inp] = idx
        # greedy buffer reuse
        free = []
        buffers = {}
        peak = 0
        next_free = 0
        for idx, name in enumerate(graph.order):
            for inp in graph.nodes[name].inputs:
                if inp in graph.nodes and last_use.get(inp) == idx:
                    free.append(buffers[inp])
            if free:
                buffers[name] = free.pop()
            else:
                buffers[name] = next_free
                next_free += 1
            peak = max(peak, next_free - len(free))
        self._peak = peak
        return next_free - peak  # number of buffers saved vs. naive one-per-node
