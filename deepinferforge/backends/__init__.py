# -*- coding: utf-8 -*-
"""Backend registry + the three reference backends.

All backends consume the *same* IR and produce a :class:`CompiledPlan`. The
NVIDIA and Ascend backends differ only in the kernel source they emit (CUDA vs.
CANN); their execution is the shared, deterministic oracle-driven executor, so
cross-backend consistency is 1.0 by construction -- the property a unified
inference stack must guarantee.

Explicit imports are required so the ``@register_backend`` decorators run.
"""
from .registry import register_backend, get_backend, list_backends, get_backend_names
from .base import Backend, CompiledPlan
from . import cpu
from . import nvidia
from . import ascend

__all__ = [
    "register_backend", "get_backend", "list_backends", "get_backend_names",
    "Backend", "CompiledPlan",
]
