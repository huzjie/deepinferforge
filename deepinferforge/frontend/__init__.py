# -*- coding: utf-8 -*-
"""Frontend: lower a model *description* into the unified IR.

The frontend is the only place a framework-specific representation is
understood. Everything downstream (passes, backends) sees only the IR, which is
why adding a new source model is a frontend-only change and never touches the
backends.
"""
from .descriptor import ModelDescriptor, build_transformer_graph
from .importer import import_model

__all__ = ["ModelDescriptor", "build_transformer_graph", "import_model"]
