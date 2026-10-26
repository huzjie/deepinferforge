# -*- coding: utf-8 -*-
"""A cache manager with a simple LRU eviction policy across sequences."""
from .paged import PagedKVCache


class KVManager(object):
    def __init__(self, n_blocks=64, block_tokens=16, n_layers=4, max_seqs=8):
        self.cache = PagedKVCache(n_blocks, block_tokens, n_layers)
        self.max_seqs = max_seqs
        self.seq_order = []   # LRU order (oldest first)

    def _touch(self, seq_id):
        if seq_id in self.seq_order:
            self.seq_order.remove(seq_id)
        self.seq_order.append(seq_id)

    def ensure(self, seq_id, n_tokens):
        self._touch(seq_id)
        ids = self.cache.append(seq_id, n_tokens)
        if ids:
            return True
        # evict oldest until we can allocate
        while self.seq_order and self.cache.table.free_blocks == 0:
            victim = self.seq_order.pop(0)
            if victim != seq_id:
                self.cache.evict(victim)
            else:
                break
        ids = self.cache.append(seq_id, n_tokens)
        return bool(ids)

    def release(self, seq_id):
        self.cache.evict(seq_id)
        if seq_id in self.seq_order:
            self.seq_order.remove(seq_id)
