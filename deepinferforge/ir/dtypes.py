# -*- coding: utf-8 -*-
"""Tensor data types used by the IR.

The cross-platform story matters most at the dtype layer: NVIDIA and Ascend
support slightly different fp8/int4 flavors, so the IR keeps a *unified* dtype
enum and lets each backend decide how to materialize it (e.g. fp8_e4m3 may map
to CUDA fp8e4m3 or Ascend fp8 depending on target).
"""


class DType(object):
    __slots__ = ("name", "bits")

    def __init__(self, name, bits):
        self.name = name
        self.bits = bits

    def __repr__(self):
        return "DType(%s)" % self.name

    def __eq__(self, other):
        return isinstance(other, DType) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


float32 = DType("float32", 32)
float16 = DType("float16", 16)
bfloat16 = DType("bfloat16", 16)
int8 = DType("int8", 8)
int32 = DType("int32", 32)
int4 = DType("int4", 4)
fp8_e4m3 = DType("fp8_e4m3", 8)
fp8_e5m2 = DType("fp8_e5m2", 8)

_BY_NAME = {d.name: d for d in
            (float32, float16, bfloat16, int8, int32, int4, fp8_e4m3, fp8_e5m2)}


def from_name(name):
    if isinstance(name, DType):
        return name
    if name not in _BY_NAME:
        raise ValueError("unknown dtype: %r" % name)
    return _BY_NAME[name]
