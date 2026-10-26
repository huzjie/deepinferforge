# -*- coding: utf-8 -*-
"""A tiny decoder-only Transformer (with optional MoE FFN) for reference runs."""
from .tensor import Tensor
from .layers import RMSNorm, Attention, FFN, MoE, Linear
from ..utils.random import stable_ints


class Transformer(object):
    def __init__(self, dim=32, n_layers=4, n_heads=4, head_dim=8,
                 vocab=128, use_moe=True, n_experts=8, top_k=2, key="tf"):
        self.dim = dim
        self.n_layers = n_layers
        self.embed = Linear(vocab, dim, key + ":embed")
        self.layers = []
        for i in range(n_layers):
            attn = Attention(dim, n_heads, head_dim, "%s:l%d:attn" % (key, i))
            ffn = (MoE(dim, dim * 2, n_experts, top_k, "%s:l%d:moe" % (key, i))
                   if use_moe else FFN(dim, dim * 2, "%s:l%d:ffn" % (key, i)))
            self.layers.append((RMSNorm(dim, "%s:l%d:rms1" % (key, i)), attn,
                                RMSNorm(dim, "%s:l%d:rms2" % (key, i)), ffn))
        self.norm = RMSNorm(dim, key + ":final")
        self.lm_head = Linear(dim, vocab, key + ":lmhead")

    def forward(self, token_ids):
        """token_ids: list of ints; returns logits Tensor [seq, vocab]."""
        x = Tensor([self.embed.weight.transpose().data[t] for t in token_ids])
        for rms1, attn, rms2, ffn in self.layers:
            a = attn.forward(rms1.forward(x))
            x = x.add(a)
            f = ffn.forward(rms2.forward(x))
            x = x.add(f)
        return self.lm_head.forward(self.norm.forward(x))

    def sample(self, token_ids, temperature=1.0, seed="s"):
        """Deterministically sample the next token from logits."""
        logits = self.forward(token_ids)
        row = logits.data[-1]
        mx = max(row)
        ex = [((v - mx) / temperature) for v in row]
        from ..utils.random import stable_float
        # softmax + deterministic pick keyed by seed
        import math
        e = [math.exp(v) for v in ex]
        s = sum(e)
        probs = [v / s for v in e]
        u = stable_float(seed, 0.0, 1.0)
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if u <= acc:
                return i
        return len(probs) - 1
