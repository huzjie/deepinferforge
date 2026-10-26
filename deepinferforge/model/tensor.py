# -*- coding: utf-8 -*-
"""A minimal dense matrix over plain Python lists.

Supports the handful of ops the reference Transformer needs: matmul, elementwise
add/mul, relu, gelu, softmax (over the last axis), layernorm, rmsnorm and
transpose. Kept deliberately small; the point is a *correct* numerical reference,
not performance.
"""
import math


class Tensor(object):
    def __init__(self, data, dtype="float32"):
        self.data = data
        self.dtype = dtype

    @property
    def shape(self):
        if not self.data:
            return (0,)
        if isinstance(self.data[0], (list, tuple)):
            return (len(self.data), len(self.data[0]))
        return (len(self.data),)

    def __repr__(self):
        return "Tensor(shape=%s)" % (self.shape,)

    def tolist(self):
        return self.data

    @staticmethod
    def zeros(rows, cols):
        return Tensor([[0.0] * cols for _ in range(rows)])

    @staticmethod
    def rand(rows, cols, key="t"):
        from ..utils.random import stable_vector
        return Tensor([stable_vector("%s:%d" % (key, i), cols) for i in range(rows)])

    # -- ops ----------------------------------------------------------------
    def matmul(self, other):
        a = self.data
        b = other.data
        if not a or not b:
            return Tensor.zeros(0, 0)
        m, k = len(a), len(a[0])
        k2, n = len(b), len(b[0]) if isinstance(b[0], (list, tuple)) else (len(b), 1)
        if isinstance(b[0], (list, tuple)):
            out = [[sum(a[i][kk] * b[kk][j] for kk in range(k)) for j in range(n)]
                   for i in range(m)]
        else:
            out = [[sum(a[i][kk] * b[kk] for kk in range(k))] for i in range(m)]
        return Tensor(out)

    def add(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor([[self.data[i][j] + other[i][j] for j in range(len(self.data[i]))]
                       for i in range(len(self.data))])

    def mul(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor([[self.data[i][j] * other[i][j] for j in range(len(self.data[i]))]
                       for i in range(len(self.data))])

    def _elementwise(self, fn):
        return Tensor([[fn(v) for v in row] for row in self.data])

    def relu(self):
        return self._elementwise(lambda v: v if v > 0 else 0.0)

    def gelu(self):
        return self._elementwise(
            lambda v: 0.5 * v * (1.0 + math.tanh(
                math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3))))

    def softmax(self):
        out = []
        for row in self.data:
            mx = max(row)
            ex = [math.exp(v - mx) for v in row]
            s = sum(ex) or 1.0
            out.append([e / s for e in ex])
        return Tensor(out)

    def layernorm(self, eps=1e-5):
        out = []
        for row in self.data:
            mu = sum(row) / len(row)
            var = sum((v - mu) ** 2 for v in row) / len(row)
            inv = 1.0 / math.sqrt(var + eps)
            out.append([(v - mu) * inv for v in row])
        return Tensor(out)

    def rmsnorm(self, eps=1e-5):
        out = []
        for row in self.data:
            ms = sum(v * v for v in row) / len(row)
            inv = 1.0 / math.sqrt(ms + eps)
            out.append([v * inv for v in row])
        return Tensor(out)

    def transpose(self):
        if not self.data:
            return Tensor([])
        if isinstance(self.data[0], (list, tuple)):
            cols = len(self.data[0])
            return Tensor([[self.data[i][j] for i in range(len(self.data))]
                           for j in range(cols)])
        return Tensor([list(self.data)])
