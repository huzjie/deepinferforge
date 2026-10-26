# -*- coding: utf-8 -*-
"""Leviathan-style verification and the closed-form speedup.

Given an acceptance rate ``a`` and draft length ``g``, the expected number of
tokens produced per verification round is::

    expected = (1 - a^(g+1)) / (1 - a)      # = g+1 when a == 1

and the wall-clock speedup (with ``c`` = draft-token cost / target-token cost)::

    speedup = expected / (1 + g * c)

The acceptance rate rises with engine skill, so the speedup sweeps monotonically
from a modest value at ``skill=0.5`` to the theoretical maximum at ``skill=1.0``.
"""
import math


def verify(draft_tokens, target_accepts):
    """Return the accepted prefix length + the last accepted token.

    ``target_accepts`` is a per-token list of booleans (does the target accept).
    """
    n = 0
    for tok, ok in zip(draft_tokens, target_accepts):
        if not ok:
            break
        n += 1
    return n


def expected_tokens(acceptance_rate, draft_len):
    if acceptance_rate >= 1.0 - 1e-9:
        return float(draft_len + 1)
    a = acceptance_rate
    return (1.0 - a ** (draft_len + 1)) / (1.0 - a)


def speedup(acceptance_rate, draft_len, cost_ratio=0.2):
    exp = expected_tokens(acceptance_rate, draft_len)
    return exp / (1.0 + draft_len * cost_ratio)


def acceptance_for_skill(skill, base=0.3):
    """Map engine skill -> draft acceptance rate (monotonic in [0.3, 1.0])."""
    return base + (1.0 - base) * skill
