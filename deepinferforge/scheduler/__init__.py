# -*- coding: utf-8 -*-
"""Heterogeneous scheduling: tiering, pipelining and policy.

A unified stack must decide *where* each operator lives (HBM vs host memory)
and *when* it runs (pipelined / prefetched). These decisions are backend-agnostic
-- they operate on operator profiles, not on CUDA/CANN specifics.
"""
from .tiering import OperatorProfiler, TierPlanner
from .pipeline import PrefetchPipeline
from .policy import SchedulingPolicy

__all__ = ["OperatorProfiler", "TierPlanner", "PrefetchPipeline", "SchedulingPolicy"]
