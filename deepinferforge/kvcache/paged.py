# -*- coding: utf-8 -*-
"""Paged KV cache: fixed-size blocks + a block table per sequence."""


class BlockTable(object):
    def __init__(self, n_blocks):
        self.blocks = [None] * n_blocks  # None = free
        self.tables = {}                 # seq_id -> list of block ids

    def alloc(self, seq_id, n):
        ids = [i for i, b in enumerate(self.blocks) if b is None][:n]
        if len(ids) < n:
            return []
        for i in ids:
            self.blocks[i] = seq_id
        self.tables.setdefault(seq_id, []).extend(ids)
        return ids

    def free(self, seq_id):
        for i in self.tables.pop(seq_id, []):
            self.blocks[i] = None

    def blocks_of(self, seq_id):
        return self.tables.get(seq_id, [])

    @property
    def free_blocks(self):
        return sum(1 for b in self.blocks if b is None)


class PagedKVCache(object):
    def __init__(self, n_blocks=64, block_tokens=16, n_layers=4):
        self.n_blocks = n_blocks
        self.block_tokens = block_tokens
        self.n_layers = n_layers
        self.table = BlockTable(n_blocks)

    def append(self, seq_id, n_tokens):
        n_blocks = (n_tokens + self.block_tokens - 1) // self.block_tokens
        return self.table.alloc(seq_id, n_blocks * self.n_layers)

    def evict(self, seq_id):
        self.table.free(seq_id)

    @property
    def utilization(self):
        return 1.0 - self.table.free_blocks / self.n_blocks
