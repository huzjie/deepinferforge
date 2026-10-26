# -*- coding: utf-8 -*-
"""Configuration for deepinferforge.

A plain dataclass plus a minimal YAML loader (falling back to a tiny hand-rolled
parser ``utils/yamlish.py`` because the managed Python runtime has no pyyaml).
Declared scale numbers (671B MoE / dual NVIDIA+Ascend targets) are informational
only; the reference implementation instantiates tiny tensors so a full compile +
run completes in milliseconds of pure Python.
"""
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:  # pragma: no cover - optional
    import yaml as _pyyaml  # type: ignore
except Exception:  # pragma: no cover
    _pyyaml = None


@dataclass
class IrConfig:
    default_dtype: str = "float32"
    name: str = "graph"


@dataclass
class BackendConfig:
    default_backend: str = "nvidia"
    # informational: which accelerator families DeepInfer-style stacks target
    targets: tuple = ("nvidia", "ascend")


@dataclass
class PassConfig:
    fusion: bool = True
    constfold: bool = True
    dce: bool = True
    memory_plan: bool = True
    canonicalize: bool = True


@dataclass
class AutotuneConfig:
    max_trials: int = 32
    strategy: str = "grid"          # grid | anneal
    tile_candidates: tuple = (8, 16, 32, 64)


@dataclass
class QuantConfig:
    scheme: str = "int8"            # int8 | fp8_e4m3 | w4a4
    group_size: int = 128
    symmetric: bool = True


@dataclass
class ModelConfig:
    dim: int = 32
    n_layers: int = 4
    n_heads: int = 4
    head_dim: int = 8
    vocab: int = 128
    seq_len: int = 16
    n_experts: int = 8
    top_k: int = 2
    # declared scale (informational)
    declared_params_b: float = 671.0
    declared_active_b: float = 37.0


@dataclass
class RuntimeConfig:
    pool_bytes: int = 1 << 20       # arena size (informational)
    max_streams: int = 4


@dataclass
class SchedulerConfig:
    hbm_bytes: int = 1 << 30        # informational
    host_bytes: int = 1 << 33
    hot_fraction: float = 0.8
    prefetch_depth: int = 2


@dataclass
class SpeculativeConfig:
    draft_len: int = 4
    cost_ratio: float = 0.2         # draft token cost / target token cost
    acceptance_base: float = 0.3


@dataclass
class ServingConfig:
    host: str = "127.0.0.1"
    port: int = 8421


@dataclass
class TrainConfig:
    learning_rate: float = 0.3
    max_steps: int = 24
    seed: int = 20261026


@dataclass
class BenchConfig:
    n_tasks: int = 256
    seed: int = 20261026


@dataclass
class DeclaredConfig:
    project: str = "DeepInfer"
    vendor: str = "DeepSeek"
    announcement: str = "2026-10-26"
    dual_target: str = "NVIDIA (CUDA) + Huawei Ascend (CANN)"
    model_params_b: float = 671.0
    model_active_b: float = 37.0
    license: str = "open source"

    def to_dict(self):
        return asdict(self)


@dataclass
class Config:
    ir: IrConfig = field(default_factory=IrConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    passes: PassConfig = field(default_factory=PassConfig)
    autotune: AutotuneConfig = field(default_factory=AutotuneConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)
    declared: DeclaredConfig = field(default_factory=DeclaredConfig)

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        factories = {
            "ir": IrConfig, "backend": BackendConfig, "passes": PassConfig,
            "autotune": AutotuneConfig, "quant": QuantConfig, "model": ModelConfig,
            "runtime": RuntimeConfig, "scheduler": SchedulerConfig,
            "speculative": SpeculativeConfig, "serving": ServingConfig,
            "train": TrainConfig, "bench": BenchConfig, "declared": DeclaredConfig,
        }
        kw = {}
        for key, factory in factories.items():
            sub = d.get(key) or {}
            if key == "backend" and "targets" in sub and not isinstance(sub["targets"], tuple):
                sub = dict(sub)
                sub["targets"] = tuple(x for x in (sub["targets"] or ["nvidia", "ascend"]))
            kw[key] = factory(**{k: v for k, v in sub.items()})
        return cls(**kw)

    def to_dict(self):
        d = asdict(self)
        d["backend"]["targets"] = list(d["backend"]["targets"])
        return d


def load_config(path=None):
    if path is None:
        here = Path(__file__).parent
        path = here / "configs" / "default.yaml"
        if not path.exists():
            path = here.parent / "configs" / "default.yaml"
    p = Path(path)
    if not p.exists():
        return Config()
    text = p.read_text(encoding="utf-8")
    data = None
    if _pyyaml is not None:
        try:
            data = _pyyaml.safe_load(text)
        except Exception:
            data = None
    if data is None:
        from .utils.yamlish import parse
        data = parse(text)
    return Config.from_dict(data)
