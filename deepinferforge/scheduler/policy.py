# -*- coding: utf-8 -*-
"""Scheduling policy: latency-first vs memory-first.

Picks an execution order given a profile: latency-first launches the critical
path first; memory-first launches the path with the smallest peak memory. The
reference returns a deterministic order keyed by the chosen objective.
"""


class SchedulingPolicy(object):
    def __init__(self, objective="latency"):
        self.objective = objective

    def order(self, ops, profiles=None):
        if self.objective == "memory":
            return sorted(ops, key=lambda o: profiles.get(o, {}).get("bytes", 0))
        # latency-first: preserve critical-path order (already topo-sorted)
        return list(ops)
