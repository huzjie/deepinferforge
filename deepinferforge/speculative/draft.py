# -*- coding: utf-8 -*-
"""A small draft model that proposes ``draft_len`` tokens quickly.

The draft is a cheap, deterministic next-token proposer (a lower-precision /
smaller stand-in for the target). Its proposals are verified by the target in
:mod:`deepinferforge.speculative.verify`.
"""
from ..utils.random import stable_int


class DraftModel(object):
    def __init__(self, vocab=128, draft_len=4):
        self.vocab = vocab
        self.draft_len = draft_len

    def propose(self, prefix, seed="draft"):
        return [stable_int("draft:%s:%d" % (seed, i), 0, self.vocab)
                for i in range(self.draft_len)]
