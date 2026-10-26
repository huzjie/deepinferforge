# -*- coding: utf-8 -*-
"""Speculative-decoding speedup via the Leviathan closed form.

Acceptance rate rises with skill, so the expected tokens-per-round and the
wall-clock speedup both sweep monotonically. The formula and its limit behavior
(a -> 1 gives g+1 tokens/round) are asserted in the tests.
"""
from ..bench.registry import register_benchmark
from ..speculative.verify import speedup, acceptance_for_skill, expected_tokens


@register_benchmark("speculative")
def run_speculative(skill=1.0, draft_len=4, cost_ratio=0.2, base=0.3, cfg=None):
    a = acceptance_for_skill(skill, base)
    sp = speedup(a, draft_len, cost_ratio)
    exp = expected_tokens(a, draft_len)
    return {
        "acceptance_rate": round(a, 4),
        "expected_tokens": round(exp, 4),
        "speedup": round(sp, 4),
    }
