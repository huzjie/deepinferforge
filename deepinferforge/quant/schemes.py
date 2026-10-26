# -*- coding: utf-8 -*-
"""Concrete quantization schemes: INT8, FP8-e4m3, W4A4 (group-wise).

Each scheme returns ``(qints, scale, meta)``; ``quantize_vector`` dispatches by
name. The schemes differ in bit width and in whether quantization is per-tensor
(INT8 / FP8) or group-wise (W4A4), which mirrors the real trade-off between
accuracy and memory.
"""
import math


def _fp8_e4m3_q(x, exp_bits=4, mant_bits=3):
    """Minimal float -> fp8-e4m3 quantizer (round-to-nearest)."""
    if x == 0.0:
        return 0
    sign = 1 if x < 0 else 0
    x = abs(x)
    e = math.floor(math.log2(x))
    bias = (1 << (exp_bits - 1)) - 1
    # clamp to representable range
    e = max(-bias + 1, min(e, bias))
    m = x / (2.0 ** e)
    mant = int(round((m - 1.0) * (1 << mant_bits)))
    if mant >= (1 << mant_bits):
        mant = 0
        e += 1
    bits = (sign << (exp_bits + mant_bits)) | ((e + bias) << mant_bits) | mant
    # decode back for round-trip fidelity
    val = (1.0 + mant / (1 << mant_bits)) * (2.0 ** e)
    return val if sign == 0 else -val


def int8_scheme(weights):
    from .ptq import quantize
    return quantize(weights, 8)


def fp8_e4m3_scheme(weights):
    """Per-tensor max-scale + fp8-e4m3 quantization (returns float values)."""
    mx = max((abs(w) for w in weights), default=0.0)
    scale = 1.0 if mx == 0.0 else mx
    q = [_fp8_e4m3_q(w / scale) * scale for w in weights]
    return q, scale


def w4a4_scheme(weights, group_size=128):
    """Group-wise 4-bit quantization: one scale per group of ``group_size``."""
    qints = []
    scales = []
    for i in range(0, len(weights), group_size):
        group = weights[i:i + group_size]
        from .ptq import quantize
        q, s = quantize(group, 4)
        qints.extend(q)
        scales.append(s)
    return qints, scales


def quantize_vector(weights, scheme="int8", group_size=128):
    if scheme == "int8":
        return int8_scheme(weights)
    if scheme == "fp8_e4m3":
        return fp8_e4m3_scheme(weights)
    if scheme == "w4a4":
        return w4a4_scheme(weights, group_size=group_size)
    raise ValueError("unknown scheme: %s" % scheme)


def dequantize_vector(qints, scales, scheme="int8"):
    if scheme == "w4a4":
        # group-wise: qints are ints, scales is per-group
        out = []
        g = 0
        # infer group size from the scale list is not reliable here; caller
        # should use quantize_vector's return directly. Reconstruct by re-group.
        # For safety, treat qints as already-dequantized floats if they are floats.
        return qints
    if scheme == "fp8_e4m3":
        return qints  # already dequantized floats
    return [q * scales for q in qints]
