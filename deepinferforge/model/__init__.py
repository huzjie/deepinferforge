# -*- coding: utf-8 -*-
"""A small, hand-rolled reference model (tensor + layers + Transformer/MoE).

Pure Python, no numpy/torch. The tensor is a plain list-of-lists matrix with the
few ops a decoder-only model needs. It exists so the engine has a *real* model
to compile and run end-to-end, not just an abstract graph.
"""
from .tensor import Tensor
from .layers import Linear, LayerNorm, RMSNorm, Attention, FFN, MoE
from .transformer import Transformer

__all__ = [
    "Tensor", "Linear", "LayerNorm", "RMSNorm", "Attention", "FFN", "MoE",
    "Transformer",
]
