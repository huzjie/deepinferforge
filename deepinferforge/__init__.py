# -*- coding: utf-8 -*-
"""deepinferforge - a unified cross-platform inference compiler & runtime.

A pure-Python, zero-dependency reference implementation of the ideas behind
DeepSeek's open-sourced **DeepInfer** inference framework (announced 2026-10-26),
which compiles one model graph down to *both* NVIDIA (CUDA) and Huawei Ascend
(CANN) backends from a single intermediate representation (IR).

The composable pieces:

  * ir          - a single operator-level IR (graph of typed nodes) that is the
                  one source of truth for every backend.
  * frontend    - turns a model descriptor / ONNX-like spec into the IR.
  * passes      - platform-agnostic graph optimizations: operator fusion,
                  constant folding, DCE, memory planning, canonicalization.
  * backends    - a unified operator interface with NVIDIA / Ascend / CPU
                  reference backends, all driven by the *same* IR.
  * codegen     - emits kernel source (CUDA ``__global__`` / CANN snippets).
  * autotune    - a cost model + search that picks the best kernel config.
  * quant       - symmetric PTQ: INT8 / FP8-e4m3 / W4A4 group-wise.
  * model       - a tiny hand-rolled tensor + Transformer/MoE for reference.
  * runtime     - executor + arena memory pool.
  * scheduler   - heterogeneous tiering & pipelining (HBM / host memory).
  * speculative - Leviathan-style draft-then-verify decoding + paged KV cache.
  * engine      - the high-level ``InferenceEngine`` that ties it together.

Everything is deterministic (md5-seeded) and shares a single oracle. ``skill =
0.5`` is an untuned / untrained configuration and ``skill = 1.0`` is the
perfect one, so every benchmark sweeps monotonically from a random baseline to
the optimum.
"""

__version__ = "1.0.0"

from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
