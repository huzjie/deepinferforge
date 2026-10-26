# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Quantize a weight vector under INT8 / FP8-e4m3 / W4A4 and measure round-trip."""
from deepinferforge.quant.schemes import quantize_vector
from deepinferforge.utils.random import stable_vector
from deepinferforge.utils.metrics import cosine

w = stable_vector("example:w", 64, -1.0, 1.0)
for scheme in ("int8", "fp8_e4m3", "w4a4"):
    q, scale = quantize_vector(w, scheme=scheme, group_size=16)
    # reconstruct a dequantized vector for the cosine check
    if scheme == "w4a4":
        from deepinferforge.quant.ptq import dequantize
        dq = []
        g = 0
        for i in range(0, len(w), 16):
            dq.extend(dequantize(q[i:i + 16], scale[g]))
            g += 1
    else:
        dq = q if scheme == "fp8_e4m3" else [x * scale for x in q]
    print("%-8s bits=%s  round-trip cosine=%.5f"
          % (scheme, {"int8": 8, "fp8_e4m3": 8, "w4a4": 4}[scheme], cosine(w, dq)))
