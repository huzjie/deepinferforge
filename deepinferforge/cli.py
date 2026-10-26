# -*- coding: utf-8 -*-
"""Command-line interface: doctor / compile / bench / train / serve."""
import argparse
import json
import sys

from .config import load_config
from .engine.inference import InferenceEngine
from .frontend.importer import import_model
from .backends.registry import list_backends


def _make_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="path to a YAML config")
    p = argparse.ArgumentParser(prog="deepinferforge", parents=[common])
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("doctor", parents=[common], help="smoke-test every subsystem")
    sub.add_parser("compile", parents=[common], help="compile a model descriptor to a plan")
    sub.add_parser("bench", parents=[common], help="run the benchmark suite")
    sub.add_parser("train", parents=[common], help="run mock training (skill -> 1.0)")
    sp = sub.add_parser("serve", parents=[common], help="start the HTTP server")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)
    return p


def cmd_doctor(cfg):
    engine = InferenceEngine(cfg)
    print("config:", engine.config.declared.project, "/",
          engine.config.declared.dual_target)
    print("backends:", ", ".join(engine.available_backends()),
          "(default=%s)" % engine.backend_name)
    graph = engine.build_model_graph()
    print("model graph: %d nodes, %d outputs" % (graph.op_count(), len(graph.outputs)))
    for name in list_backends():
        plan = engine.compile(graph)
        engine.switch_backend(name)
        plan = engine.compile(graph)
        out = engine.run(plan)
        print("  backend=%-8s kernels=%d outputs=%d pass=%s"
              % (name, plan.n_kernels, len(out), plan.pass_report))
    from .autotune import CostModel, grid_search
    from .oracle import gold_config
    m = CostModel(cfg.autotune)
    best, cost = grid_search("matmul", m, cfg.autotune)
    print("  autotune: grid_search(matmul) ->", best, "(gold=", gold_config("matmul"), ")")
    from .bench.runner import run_all, summarize
    print("bench summary:\n" + summarize(run_all(cfg)))
    return 0


def cmd_compile(cfg):
    engine = InferenceEngine(cfg)
    graph = engine.build_model_graph()
    plan = engine.compile(graph)
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_bench(cfg):
    from .bench.runner import run_all, summarize
    print(summarize(run_all(cfg)))
    return 0


def cmd_train(cfg):
    from .train import train_engine
    engine = InferenceEngine(cfg)
    hist = train_engine(engine)
    print("skill: %.4f -> %.4f (%d steps)" % (hist[0], hist[-1], len(hist) - 1))
    return 0


def cmd_serve(cfg, args):
    from .serving import serve
    host = args.host or cfg.serving.host
    port = args.port or cfg.serving.port
    serve(host, port)
    return 0


def main(argv=None):
    args = _make_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "doctor":
        return cmd_doctor(cfg)
    if args.cmd == "compile":
        return cmd_compile(cfg)
    if args.cmd == "bench":
        return cmd_bench(cfg)
    if args.cmd == "train":
        return cmd_train(cfg)
    if args.cmd == "serve":
        return cmd_serve(cfg, args)
    _make_parser().print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
