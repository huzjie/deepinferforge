# -*- coding: utf-8 -*-
"""A declarative model descriptor + the lowering to IR.

``ModelDescriptor`` captures just enough of a decoder-only Transformer / MoE to
build a representative graph: dimensions, layer count, heads, whether MoE is on.
``build_transformer_graph`` then emits an IR graph with the canonical fusion
patterns (matmul+bias+gelu, attention, rmsnorm) so the passes have something
meaningful to optimize.
"""
from dataclasses import dataclass, field

from ..ir.builder import GraphBuilder


@dataclass
class ModelDescriptor:
    name: str = "transformer"
    dim: int = 32
    n_layers: int = 4
    n_heads: int = 4
    head_dim: int = 8
    vocab: int = 128
    use_moe: bool = True
    n_experts: int = 8
    top_k: int = 2

    def to_dict(self):
        return {
            "name": self.name, "dim": self.dim, "n_layers": self.n_layers,
            "n_heads": self.n_heads, "head_dim": self.head_dim,
            "vocab": self.vocab, "use_moe": self.use_moe,
            "n_experts": self.n_experts, "top_k": self.top_k,
        }

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        known = {f: d.get(f) for f in
                 ("name", "dim", "n_layers", "n_heads", "head_dim",
                  "vocab", "use_moe", "n_experts", "top_k") if f in d}
        return cls(**known)


def _layer(b, prefix, x, d, desc):
    # attention block: q/k/v projections -> attention -> out proj -> add
    q = b.matmul(x, prefix + "_wq")
    k = b.matmul(x, prefix + "_wk")
    v = b.matmul(x, prefix + "_wv")
    attn = b.attention(q, k, v)
    o = b.matmul(attn, prefix + "_wo")
    x2 = b.add(x, o)
    n = b.rmsnorm(x2, prefix + "_rms")
    # feed-forward block: either dense (matmul+gelu) or MoE
    if desc.use_moe:
        f = b.moe(n, desc.n_experts, desc.top_k)
    else:
        h = b.matmul(n, prefix + "_w1")
        h = b.gelu(h)
        f = b.matmul(h, prefix + "_w2")
    out = b.add(n, f)
    return out


def build_transformer_graph(desc):
    b = GraphBuilder(desc.name)
    x = b.input("tokens", shape=[1, desc.dim])
    h = x
    for i in range(desc.n_layers):
        h = _layer(b, "l%d" % i, h, desc.dim, desc)
    logits = b.matmul(h, "lm_head")
    b.output(logits)
    return b.build()
