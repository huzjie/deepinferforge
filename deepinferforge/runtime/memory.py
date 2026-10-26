# -*- coding: utf-8 -*-
"""A simple arena memory pool with buffer reuse.

The pool hands out fixed-size buffers and recycles them when released. It models
the reuse that memory planning enables: peak bytes tracks how many buffers are
live at once, which the memory-planning pass minimizes.
"""


class Buffer(object):
    __slots__ = ("size", "data", "free")

    def __init__(self, size):
        self.size = size
        self.data = [0.0] * size
        self.free = True


class MemoryPool(object):
    def __init__(self, capacity_bytes, block_bytes=4096):
        self.capacity_bytes = capacity_bytes
        self.block_bytes = block_bytes
        self.n_blocks = max(1, capacity_bytes // block_bytes)
        self.free_blocks = list(range(self.n_blocks))
        self.live = {}
        self.peak_live = 0

    def alloc(self, name, n_blocks=1):
        if len(self.free_blocks) < n_blocks:
            raise MemoryError("pool exhausted")
        ids = [self.free_blocks.pop() for _ in range(n_blocks)]
        self.live[name] = ids
        self.peak_live = max(self.peak_live, len(self.live))
        return ids

    def free(self, name):
        if name in self.live:
            self.free_blocks.extend(self.live.pop(name))

    @property
    def free_bytes(self):
        return len(self.free_blocks) * self.block_bytes

    @property
    def utilization(self):
        return 1.0 - (len(self.free_blocks) / self.n_blocks)
