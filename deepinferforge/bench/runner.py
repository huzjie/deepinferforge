# -*- coding: utf-8 -*-
"""Run all benchmarks and summarize, sweeping ``skill`` where it matters."""
from .registry import list_benchmarks, get_benchmark


def run_all(cfg=None, skills=(0.5, 1.0)):
    report = {}
    for name in list_benchmarks():
        fn = get_benchmark(name)
        if name in ("quantization", "speculative", "autotune"):
            report[name] = {("skill=%.1f" % s): fn(skill=s, cfg=cfg)
                            for s in skills}
        else:
            report[name] = fn(cfg=cfg)
    return report


def summarize(report):
    lines = []
    for name, value in report.items():
        if isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()):
            # skill-swept benchmark: show the two endpoints
            lo = value.get("skill=0.5", {})
            hi = value.get("skill=1.0", {})
            metric = (next(iter(lo), "?") if lo else "?")
            lines.append("%-14s %s: %.4f -> %.4f"
                         % (name, metric, _first(lo), _first(hi)))
        elif isinstance(value, dict):
            pairs = ", ".join("%s=%s" % (k, v) for k, v in value.items())
            lines.append("%-14s %s" % (name, pairs))
    return "\n".join(lines)


def _first(d):
    for v in d.values():
        return v
    return 0.0
