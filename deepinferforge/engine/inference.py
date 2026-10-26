# -*- coding: utf-8 -*-
"""The high-level :class:`InferenceEngine`.

Usage::

    engine = InferenceEngine.from_config()
    plan   = engine.compile(model_graph)
    out    = engine.run(plan, inputs)

The engine owns the config, the pass pipeline (via backends), the autotuner and
the speculative decoder, and exposes a single coherent entry point. It is what
the CLI and the HTTP server wrap.
"""
from ..config import Config, load_config
from ..frontend.descriptor import ModelDescriptor, build_transformer_graph
from ..frontend.importer import import_model
from ..backends.registry import get_backend, list_backends
from ..autotune import CostModel, tune_graph


class InferenceEngine(object):
    def __init__(self, config=None):
        self.config = config or load_config()
        self.backend_name = self.config.backend.default_backend
        self.cost_model = CostModel(self.config.autotune)
        self.skill = 0.5  # default: untuned

    @classmethod
    def from_config(cls, path=None):
        return cls(load_config(path))

    def compile(self, graph):
        backend_cls = get_backend(self.backend_name)
        self._backend = backend_cls()
        return self._backend.compile(graph, self.config)

    def run(self, plan, inputs=None):
        return self._backend.run(plan, inputs)

    def build_model_graph(self, desc=None):
        desc = desc or ModelDescriptor(
            dim=self.config.model.dim, n_layers=self.config.model.n_layers,
            n_heads=self.config.model.n_heads, head_dim=self.config.model.head_dim,
            vocab=self.config.model.vocab, use_moe=True,
            n_experts=self.config.model.n_experts, top_k=self.config.model.top_k,
        )
        return build_transformer_graph(desc)

    def autotune(self, graph, strategy=None, skill=None):
        strategy = strategy or self.config.autotune.strategy
        skill = self.skill if skill is None else skill
        return tune_graph(graph, self.cost_model, strategy=strategy, skill=skill,
                          max_trials=self.config.autotune.max_trials,
                          cfg=self.config.autotune)

    def switch_backend(self, name):
        if name not in list_backends():
            raise KeyError("unknown backend %s" % name)
        self.backend_name = name

    def available_backends(self):
        return list_backends()
