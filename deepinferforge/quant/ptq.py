# -*- coding: utf-8 -*-
"""Core quantization primitives (symmetric, per-tensor)."""


def compute_scale(weights, bits, symmetric=True):
    """Scale for signed symmetric quantization: max_abs / (2^(bits-1) - 1)."""
    mx = max((abs(w) for w in weights), default=0.0)
    if mx == 0.0:
        return 1.0
    qmax = float((1 << (bits - 1)) - 1)
    return mx / qmax


def quantize(weights, bits=8):
    """Quantize to signed int, returning (qints, scale)."""
    scale = compute_scale(weights, bits)
    return [_round_clamp(w / scale, bits) for w in weights], scale


def dequantize(qints, scale):
    return [q * scale for q in qints]


def _round_clamp(x, bits):
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    v = int(round(x))
    return max(qmin, min(qmax, v))
