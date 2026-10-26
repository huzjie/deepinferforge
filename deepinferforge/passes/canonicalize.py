# -*- coding: utf-8 -*-
"""Canonicalization.

Normalizes names so the same model, lowered twice, produces byte-identical IR
metadata: sorts graph inputs/outputs and re-derives a deterministic topo order.
This is what makes IR hashing / cache keys stable across a distributed build.
"""
import hashlib
from .base import Pass


class CanonicalizePass(Pass):
    name = "canonicalize"

    def run(self, graph):
        graph.inputs = sorted(set(graph.inputs))
        graph.outputs = sorted(set(graph.outputs))
        graph.topo_sort()
        return 0


def ir_digest(graph):
    """A stable content hash of the graph (used for the compile cache)."""
    h = hashlib.md5()
    for n in graph.order or sorted(graph.nodes):
        nd = graph.nodes[n]
        h.update(("%s|%s|%s\n" % (n, nd.op, ",".join(nd.inputs))).encode("utf-8"))
    return h.hexdigest()
