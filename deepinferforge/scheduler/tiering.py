# -*- coding: utf-8 -*-
"""Operator tiering: keep hot operators in HBM, spill cold ones to host memory.

Access frequency is derived deterministically from the op name (via the oracle's
stable hash), so the "hot" set is stable across runs. The planner then assigns
the top ``hot_fraction`` operators to HBM and the rest to host, respecting the
declared byte budgets.
"""
from ..utils.random import stable_float


class OperatorProfiler(object):
    def __init__(self):
        self._freq = {}

    def frequency(self, op):
        if op not in self._freq:
            self._freq[op] = stable_float("freq:" + op, 0.0, 1.0)
        return self._freq[op]

    def bytes(self, op):
        return stable_float("bytes:" + op, 1024, 65536)


class TierPlanner(object):
    def __init__(self, hbm_bytes, host_bytes, hot_fraction=0.8):
        self.hbm_bytes = hbm_bytes
        self.host_bytes = host_bytes
        self.hot_fraction = hot_fraction

    def plan(self, ops):
        prof = OperatorProfiler()
        ranked = sorted(set(ops), key=lambda o: prof.frequency(o), reverse=True)
        n_hot = max(1, int(len(ranked) * self.hot_fraction))
        hot = set(ranked[:n_hot])
        layout = {}
        hbm_used = 0.0
        host_used = 0.0
        for o in ranked:
            b = prof.bytes(o)
            if o in hot and hbm_used + b <= self.hbm_bytes:
                layout[o] = "hbm"
                hbm_used += b
            else:
                layout[o] = "host"
                host_used += b
        return {"layout": layout, "hbm_used": hbm_used,
                "host_used": host_used, "n_hot": len([1 for o in ranked if layout[o] == "hbm"])}
