# -*- coding: utf-8 -*-
"""The default optimization pipeline applied before every backend lowers the IR."""
from .base import PassManager
from .canonicalize import CanonicalizePass
from .constfold import ConstFoldPass
from .fusion import FusionPass
from .dce import DeadCodeEliminationPass
from .memory import MemoryPlanPass


def default_pipeline(cfg=None):
    """Build the default pass pipeline (canonicalize -> constfold -> fusion -> dce -> memory)."""
    pm = PassManager()
    pm.add(CanonicalizePass())
    pm.add(ConstFoldPass())
    pm.add(FusionPass())
    pm.add(DeadCodeEliminationPass())
    pm.add(MemoryPlanPass())
    return pm
