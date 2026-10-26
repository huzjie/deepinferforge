# -*- coding: utf-8 -*-
"""A cost model for kernel configurations.

Uses the oracle's deterministic ``op_cost`` so the objective is well-posed: the
gold config is always the minimum, and any deviation pays a penalty. A real
engine would plug in a measured/learned cost model here -- the interface stays
the same.
"""
from ..oracle import op_cost


class CostModel(object):
    def __init__(self, cfg=None):
        self.cfg = cfg

    def evaluate(self, op, config):
        return op_cost(
            op,
            config.get("tile", 32),
            config.get("layout", "nchw"),
            config.get("unroll", 4),
        )

    def candidate_space(self, op, cfg=None):
        tiles = list(getattr(cfg, "tile_candidates", (8, 16, 32, 64))
                     if cfg else (8, 16, 32, 64))
        layouts = ["nchw", "nhwc", "tiled"]
        unrolls = list(range(1, 8))
        return [
            {"tile": t, "layout": l, "unroll": u}
            for t in tiles for l in layouts for u in unrolls
        ]
