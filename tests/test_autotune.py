# -*- coding: utf-8 -*-
import unittest

from deepinferforge.autotune import CostModel, grid_search
from deepinferforge.oracle import gold_config, op_cost


class TestAutotune(unittest.TestCase):
    def test_grid_finds_gold(self):
        m = CostModel()
        best, cost = grid_search("matmul", m)
        self.assertEqual(best, gold_config("matmul"))
        self.assertLessEqual(cost, op_cost("matmul", **gold_config("matmul")))

    def test_gold_is_minimum(self):
        gold = gold_config("softmax")
        gcost = op_cost("softmax", **gold)
        for tile in (8, 16, 32, 64):
            for layout in ("nchw", "nhwc", "tiled"):
                c = op_cost("softmax", tile, layout, gold["unroll"])
                self.assertGreaterEqual(c, gcost)


if __name__ == "__main__":
    unittest.main()
