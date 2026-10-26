# -*- coding: utf-8 -*-
"""Reference layers: Linear, LayerNorm, RMSNorm, Attention, FFN, MoE.

Weights are deterministic (md5-seeded) and fixed at construction; only the
*skill* of the surrounding engine changes the effective behavior (used by the
mock training/benchmark, never by the reference forward pass itself).
"""
from .tensor import Tensor
from ..utils.random import stable_vector, stable_ints, stable_float


class Linear(object):
    def __init__(self, in_dim, out_dim, key="lin"):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Tensor([stable_vector("%s:w%d" % (key, i), in_dim)
                              for i in range(out_dim)])
        self.bias = [stable_float("%s:b%d" % (key, i), -0.1, 0.1) for i in range(out_dim)]

    def forward(self, x):
        y = x.matmul(self.weight.transpose())
        return Tensor([[y.data[i][j] + self.bias[j] for j in range(self.out_dim)]
                       for i in range(len(x.data))])


class LayerNorm(object):
    def __init__(self, dim, key="ln"):
        self.dim = dim
        self.gamma = stable_vector("%s:g" % key, dim, 0.8, 1.2)
        self.beta = stable_vector("%s:b" % key, dim, -0.1, 0.1)

    def forward(self, x):
        n = x.layernorm()
        return Tensor([[n.data[i][j] * self.gamma[j] + self.beta[j]
                        for j in range(self.dim)] for i in range(len(x.data))])


class RMSNorm(object):
    def __init__(self, dim, key="rms"):
        self.dim = dim
        self.gamma = stable_vector("%s:g" % key, dim, 0.8, 1.2)

    def forward(self, x):
        n = x.rmsnorm()
        return Tensor([[n.data[i][j] * self.gamma[j] for j in range(self.dim)]
                       for i in range(len(x.data))])


class Attention(object):
    def __init__(self, dim, n_heads, head_dim, key="attn"):
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.wq = Linear(dim, n_heads * head_dim, key + ":wq")
        self.wk = Linear(dim, n_heads * head_dim, key + ":wk")
        self.wv = Linear(dim, n_heads * head_dim, key + ":wv")
        self.wo = Linear(n_heads * head_dim, dim, key + ":wo")

    def forward(self, x):
        q = self.wq.forward(x)
        k = self.wk.forward(x)
        v = self.wv.forward(x)
        # scaled dot-product (single head for the reference; heads collapse here)
        scores = q.matmul(k.transpose())
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = Tensor([[s * scale for s in row] for row in scores.data])
        probs = scores.softmax()
        ctx = probs.matmul(v)
        return self.wo.forward(ctx)


class FFN(object):
    def __init__(self, dim, hidden, key="ffn"):
        self.w1 = Linear(dim, hidden, key + ":w1")
        self.w2 = Linear(hidden, dim, key + ":w2")

    def forward(self, x):
        h = self.w1.forward(x).gelu()
        return self.w2.forward(h)


class MoE(object):
    """A dense-mock MoE: routes each token to top_k experts by a fixed hash,
    then averages their outputs (reference semantics; not a real sparse gate)."""
    def __init__(self, dim, hidden, n_experts, top_k, key="moe"):
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = [FFN(dim, hidden, "%s:e%d" % (key, i))
                        for i in range(n_experts)]
        self.gate = Linear(dim, n_experts, key + ":gate")

    def route(self, x):
        logits = self.gate.forward(x)
        out = []
        for row in logits.data:
            top = sorted(range(len(row)), key=lambda j: row[j], reverse=True)[:self.top_k]
            out.append(top)
        return out

    def forward(self, x):
        routes = self.route(x)
        outs = []
        for i, row in enumerate(x.data):
            acc = None
            for e in routes[i]:
                eo = self.experts[e].forward(Tensor([row]))
                acc = eo if acc is None else acc.add(eo)
            outs.append([v / len(routes[i]) for v in acc.data[0]])
        return Tensor(outs)
