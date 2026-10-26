# -*- coding: utf-8 -*-
import unittest

from deepinferforge.quant.ptq import quantize, dequantize, compute_scale
from deepinferforge.quant.schemes import quantize_vector
from deepinferforge.utils.metrics import cosine


class TestQuant(unittest.TestCase):
    def test_scale(self):
        self.assertAlmostEqual(compute_scale([1.0, -1.0], 8), 1.0 / 127, places=7)

    def test_roundtrip(self):
        w = [1.0, -1.0, 0.5, -0.25]
        q, s = quantize(w, 8)
        dq = dequantize(q, s)
        self.assertGreater(cosine(w, dq), 0.999)

    def test_schemes(self):
        w = [0.7, -0.3, 0.9, 0.1, -0.5, 0.2]
        for scheme in ("int8", "fp8_e4m3", "w4a4"):
            out = quantize_vector(w, scheme=scheme, group_size=3)
            self.assertTrue(out is not None)


if __name__ == "__main__":
    unittest.main()
