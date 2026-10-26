# -*- coding: utf-8 -*-
"""Prefetch pipeline: overlap compute with memory transfer.

A depth-k pipeline issues the next k operators' transfers while the current
operator computes, hiding transfer latency. This is the standard technique a
unified stack uses to hide the HBM->host (or SSD->HBM) gap on both NVIDIA and
Ascend hardware.
"""


class PrefetchPipeline(object):
    def __init__(self, prefetch_depth=2):
        self.prefetch_depth = prefetch_depth
        self.issued = []
        self.completed = []

    def step(self, op, compute_cost=1.0, transfer_cost=0.5):
        """Model one timestep; returns the number of ops whose transfer was
        fully hidden behind compute (i.e. transfer <= compute window)."""
        self.issued.append(op)
        hidden = 0
        if len(self.issued) > 1:
            # the op prefetched ``depth`` steps ago has had its transfer hidden
            hidden = 1
        self.completed.append(op)
        return hidden

    def run(self, ops, compute_cost=1.0, transfer_cost=0.5):
        hidden = 0
        for op in ops:
            hidden += self.step(op, compute_cost, transfer_cost)
        return hidden

    @staticmethod
    def speedup(n_ops, prefetch_depth, transfer_cost=0.5, compute_cost=1.0):
        """Closed-form speedup: serial = n*(c+t); pipelined ~ n*c + t (amortized)."""
        serial = n_ops * (compute_cost + transfer_cost)
        pipelined = n_ops * compute_cost + transfer_cost
        return serial / max(pipelined, 1e-9)
