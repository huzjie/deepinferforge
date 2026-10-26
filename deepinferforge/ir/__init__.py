# -*- coding: utf-8 -*-
"""The unified intermediate representation (IR).

The IR is an operator-level dataflow graph of typed nodes. It is the *single*
source of truth: the frontend lowers a model into it, the optimization passes
rewrite it (platform-agnostically), and every backend lowers it to kernels.
Because there is exactly one IR, cross-backend consistency is a property of the
representation, not an afterthought.
"""
from .dtypes import DType
from .node import Node
from .graph import Graph
from .builder import GraphBuilder
from . import dsl

__all__ = ["DType", "Node", "Graph", "GraphBuilder", "dsl"]
