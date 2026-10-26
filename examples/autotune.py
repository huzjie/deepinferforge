# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Grid search vs. simulated annealing for kernel config autotuning."""
from deepinferforge.autotune import CostModel, grid_search, anneal
from deepinferforge.oracle import gold_config

model = CostModel()
for op in ("matmul", "softmax", "layernorm", "moe"):
    gold = gold_config(op)
    best_g, cost_g = grid_search(op, model)
    best_a_lo, cost_a_lo = anneal(op, model, skill=0.5, max_trials=64)
    best_a_hi, cost_a_hi = anneal(op, model, skill=1.0, max_trials=64)
    print("%-10s gold=%s grid=%s anneal(s=0.5)=%s anneal(s=1.0)=%s"
          % (op, gold, best_g, best_a_lo, best_a_hi))
    print("             cost: grid=%.0f  anneal(0.5)=%.0f  anneal(1.0)=%.0f"
          % (cost_g, cost_a_lo, cost_a_hi))
