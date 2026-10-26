# -*- coding: utf-8 -*-
"""Autotuning: pick the best kernel config per operator.

A unified compiler cannot hardcode one tile size -- it must tune per op per
backend. This package models kernel cost, then searches the config space (grid
search or simulated annealing) to minimize it. The oracle's ``gold_config`` is
the ground-truth optimum the benchmark validates against.
"""
from .cost import CostModel
from .search import grid_search, anneal, tune_graph

__all__ = ["CostModel", "grid_search", "anneal", "tune_graph"]
