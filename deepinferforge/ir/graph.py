# -*- coding: utf-8 -*-
"""An IR graph: a set of nodes plus a topological ordering.

The graph stores nodes by name and maintains topological order so every
consumer runs after its producers. It also exposes the standard queries the
passes need (producers / consumers / entry / exits).
"""


class Graph(object):
    def __init__(self, name="graph"):
        self.name = name
        self.nodes = {}          # name -> Node
        self.order = []          # topologically sorted names
        self.inputs = []         # graph input names (external)
        self.outputs = []        # graph output names

    def add_node(self, node):
        if node.name in self.nodes:
            raise ValueError("duplicate node name: %s" % node.name)
        self.nodes[node.name] = node

    def mark_input(self, name):
        if name not in self.inputs:
            self.inputs.append(name)

    def mark_output(self, name):
        if name not in self.outputs:
            self.outputs.append(name)

    def topo_sort(self):
        """Kahn's algorithm; raises on a cycle."""
        indeg = {n: 0 for n in self.nodes}
        consumers = {n: [] for n in self.nodes}
        for n, node in self.nodes.items():
            for inp in node.inputs:
                if inp in self.nodes:
                    indeg[n] += 1
                    consumers[inp].append(n)
        from collections import deque
        q = deque(sorted(n for n, d in indeg.items() if d == 0))
        order = []
        while q:
            n = q.popleft()
            order.append(n)
            for c in consumers[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    q.append(c)
        if len(order) != len(self.nodes):
            raise ValueError("cycle detected in graph %s" % self.name)
        self.order = order
        return order

    def producers(self, name):
        node = self.nodes[name]
        return [i for i in node.inputs if i in self.nodes]

    def consumers(self, name):
        return [n for n, node in self.nodes.items() if name in node.inputs]

    def op_count(self):
        return len(self.nodes)

    def n_edges(self):
        return sum(len(n.inputs) for n in self.nodes.values())

    def to_dict(self):
        return {
            "name": self.name,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "nodes": [self.nodes[n].to_dict() for n in self.order or self.nodes],
        }
