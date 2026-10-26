# -*- coding: utf-8 -*-
"""Kernel code generation.

A backend lowers each IR node to a :class:`KernelSpec`, and the emitter renders
that spec as source text for the target (CUDA ``__global__`` for NVIDIA, CANN
kernel for Ascend). The text is real, inspectable code -- what would be handed
to ``nvcc`` / the Ascend ``ccec`` compiler in a production stack.
"""
from .kernel import KernelSpec
from .emitter import emit_cuda, emit_cann, emit_source

__all__ = ["KernelSpec", "emit_cuda", "emit_cann", "emit_source"]
