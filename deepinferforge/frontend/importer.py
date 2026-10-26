# -*- coding: utf-8 -*-
"""Import a model spec (dict / JSON) into an IR graph.

Supports two shapes:

  * a model descriptor dict (``{"dim": 32, "n_layers": 4, ...}``)
  * a raw step list (``{"name":..., "steps":[[op, inputs, attrs], ...]}``)

This is the "portable" path: a JSON model can be checked into source control
and lowered identically on NVIDIA and Ascend, because the IR is the contract.
"""
from .descriptor import ModelDescriptor, build_transformer_graph
from ..ir import dsl
from ..ir.graph import Graph


def import_model(spec):
    if isinstance(spec, ModelDescriptor):
        return build_transformer_graph(spec)
    if isinstance(spec, str):
        import json
        spec = json.loads(spec)
    if not isinstance(spec, dict):
        raise TypeError("spec must be dict / ModelDescriptor / json string")

    if "steps" in spec:
        return dsl.lower(
            spec.get("name", "graph"),
            spec["steps"],
            inputs=spec.get("inputs", []),
            outputs=spec.get("outputs", []),
            dtype=spec.get("dtype", "float32"),
        )

    desc = ModelDescriptor.from_dict(spec)
    return build_transformer_graph(desc)
