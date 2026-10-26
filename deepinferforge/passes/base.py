# -*- coding: utf-8 -*-
"""Pass framework: each pass mutates a Graph in place and reports a change count."""


class Pass(object):
    name = "pass"

    def run(self, graph):
        raise NotImplementedError


class PassManager(object):
    def __init__(self, passes=None):
        self.passes = list(passes or [])

    def add(self, p):
        self.passes.append(p)

    def run(self, graph):
        report = {}
        for p in self.passes:
            n = p.run(graph)
            report[p.name] = n
        graph.topo_sort()
        return report
