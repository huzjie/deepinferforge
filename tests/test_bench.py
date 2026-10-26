# -*- coding: utf-8 -*-
import unittest

from deepinferforge.bench.consistency import run_consistency
from deepinferforge.bench.fusion import run_fusion
from deepinferforge.bench.quant import run_quant
from deepinferforge.bench.speculative import run_speculative
from deepinferforge.bench.autotune import run_autotune
from deepinferforge.bench.runner import run_all


class TestBench(unittest.TestCase):
    def test_consistency_is_one(self):
        self.assertEqual(run_consistency()["consistency"], 1.0)

    def test_fusion_speedup(self):
        r = run_fusion()
        self.assertGreater(r["speedup"], 1.0)

    def test_quant_monotonic(self):
        self.assertGreater(run_quant(skill=1.0)["quality"],
                           run_quant(skill=0.5)["quality"])

    def test_speculative_monotonic(self):
        self.assertGreater(run_speculative(skill=1.0)["speedup"],
                           run_speculative(skill=0.5)["speedup"])

    def test_autotune(self):
        self.assertEqual(run_autotune(skill=1.0)["accuracy"], 1.0)
        acc = run_autotune(skill=0.5)["accuracy"]
        self.assertGreater(acc, 0.3)
        self.assertLess(acc, 0.7)

    def test_run_all(self):
        report = run_all()
        self.assertIn("consistency", report)
        self.assertIn("fusion", report)
        self.assertIn("quantization", report)
        self.assertIn("speculative", report)
        self.assertIn("autotune", report)


if __name__ == "__main__":
    unittest.main()
