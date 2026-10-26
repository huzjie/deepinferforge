# -*- coding: utf-8 -*-
"""Paged KV cache + a cache manager.

Paged KV cache (vLLM-style) allocates K/V in fixed-size blocks so long contexts
do not fragment memory and can be shared / evicted cleanly -- essential for the
million-token contexts a DeepInfer-class engine must serve.
"""
from .paged import PagedKVCache, BlockTable
from .manager import KVManager

__all__ = ["PagedKVCache", "BlockTable", "KVManager"]
