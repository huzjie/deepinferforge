# -*- coding: utf-8 -*-
"""Quantization quality: round-trip error vs. quantizer skill.

A well-calibrated quantizer (``skill=1.0``) uses full bit-width (8-bit), so its
round-trip error is minimal; an uncalibrated one (``skill=0.5``) effectively
drops to 5-bit, inflating rounding error. Quality is ``1 / (1 + relative_L2)``
and sweeps monotonically toward 1.0. (Relative L2 -- not cosine -- is used here
because cosine is invariant to uniform scale and cannot see a scalar
miscalibration.)
"""
import math
from ..bench.registry import register_benchmark
from ..utils.random import stable_vector
from ..quant.ptq import quantize, dequantize


def _rel_l2(a, b):
    num = sum((x - y) ** 2 for x, y in zip(a, b))
    den = sum(x * x for x in a)
    return math.sqrt(num / max(den, 1e-9))


@register_benchmark("quantization")
def run_quant(skill=1.0, n=128, cfg=None):
    w = stable_vector("quant:w", n, -1.0, 1.0)
    # effective bit-width rises with calibration skill: 5 bits -> 8 bits
    bits = 2 + int(round(6.0 * skill))
    q, scale = quantize(w, bits)
    dq = dequantize(q, scale)
    err = _rel_l2(w, dq)
    quality = 1.0 / (1.0 + err)
    return {"quality": round(quality, 4), "bits": bits, "rel_l2": round(err, 5)}
