# -*- coding: utf-8 -*-
"""Platform-agnostic graph optimization passes.

These operate purely on the IR, so they run identically before *any* backend
lowers the graph. This is a deliberate design choice: fusion / folding / DCE /
memory planning are semantic-preserving rewrites of the representation and have
nothing to do with the target accelerator, so they belong in the IR layer.
"""
from .base import Pass, PassManager
from .fusion import FusionPass
from .constfold import ConstFoldPass
from .dce import DeadCodeEliminationPass
from .memory import MemoryPlanPass
from .canonicalize import CanonicalizePass
from .pipeline import default_pipeline

__all__ = [
    "Pass", "PassManager", "FusionPass", "ConstFoldPass",
    "DeadCodeEliminationPass", "MemoryPlanPass", "CanonicalizePass",
    "default_pipeline",
]
