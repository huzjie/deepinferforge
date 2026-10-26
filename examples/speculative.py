# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Speculative decoding speedup across acceptance rates."""
from deepinferforge.speculative.verify import speedup, acceptance_for_skill

for skill in (0.5, 0.75, 0.9, 1.0):
    a = acceptance_for_skill(skill, base=0.3)
    sp = speedup(a, draft_len=4, cost_ratio=0.2)
    print("skill=%.2f  acceptance=%.3f  speedup=%.3fx" % (skill, a, sp))
