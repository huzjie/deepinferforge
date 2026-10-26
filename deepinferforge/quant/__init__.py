# -*- coding: utf-8 -*-
"""Post-training quantization (PTQ).

Symmetric quantization with per-tensor (INT8 / FP8-e4m3) and group-wise (W4A4)
schemes. The quantize/dequantize round-trip is lossy by design; the benchmark
measures the round-trip error and shows it collapses toward zero as the
quantizer is tuned (``skill`` -> 1.0).
"""
from .ptq import quantize, dequantize, compute_scale
from .schemes import int8_scheme, fp8_e4m3_scheme, w4a4_scheme, quantize_vector
from .calibrate import calibrate_activation_scale

__all__ = [
    "quantize", "dequantize", "compute_scale",
    "int8_scheme", "fp8_e4m3_scheme", "w4a4_scheme", "quantize_vector",
    "calibrate_activation_scale",
]
