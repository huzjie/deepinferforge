# -*- coding: utf-8 -*-
"""Activation-scale calibration.

Real PTQ needs activation scales from a calibration set (not just weights).
This computes a per-channel scale from activation samples using a simple
percentile-of-max strategy, which is robust to outliers.
"""
from ..utils.metrics import percentile


def calibrate_activation_scale(activations, p=99.0):
    """activations: list of floats; returns a single activation scale."""
    if not activations:
        return 1.0
    mx = max(abs(a) for a in activations)
    q = percentile([abs(a) for a in activations], p)
    return max(q, mx * 0.5, 1e-6)
