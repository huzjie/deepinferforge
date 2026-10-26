# -*- coding: utf-8 -*-
"""A zero-dependency HTTP server (stdlib ``http.server``) exposing the engine.

Endpoints:

  * ``GET  /health``          - liveness + version
  * ``GET  /v1/models``       - declared model card (params / targets)
  * ``GET  /v1/backends``     - available backends
  * ``POST /v1/compile``      - compile a model descriptor, return plan summary
  * ``POST /v1/run``          - run a compiled model, return outputs
  * ``GET  /v1/bench``        - run the benchmark suite, return the summary

The server wraps the same :class:`InferenceEngine` the CLI uses, so anything you
can do on the command line is scriptable over HTTP.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine.inference import InferenceEngine
from .frontend.importer import import_model
from .bench.runner import run_all, summarize


class _Handler(BaseHTTPRequestHandler):
    engine = InferenceEngine()
    _plans = {}

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok", "version": "1.0.0"})
        if self.path == "/v1/models":
            d = self.engine.config.declared
            return self._send(200, {"project": d.project, "vendor": d.vendor,
                                    "params_b": d.model_params_b,
                                    "active_b": d.model_active_b,
                                    "dual_target": d.dual_target})
        if self.path == "/v1/backends":
            return self._send(200, {"backends": self.engine.available_backends(),
                                    "default": self.engine.backend_name})
        if self.path == "/v1/bench":
            return self._send(200, {"summary": summarize(run_all(self.engine.config))})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            payload = {}
        if self.path == "/v1/compile":
            graph = import_model(payload.get("model", {"dim": 16, "n_layers": 2}))
            plan = self.engine.compile(graph)
            self._plans[payload.get("id", "default")] = plan
            return self._send(200, {"id": payload.get("id", "default"),
                                    "backend": plan.backend,
                                    "n_kernels": plan.n_kernels,
                                    "pass_report": plan.pass_report})
        if self.path == "/v1/run":
            plan = self._plans.get(payload.get("id", "default"))
            if plan is None:
                return self._send(400, {"error": "no compiled plan; POST /v1/compile first"})
            out = self.engine.run(plan, payload.get("inputs"))
            return self._send(200, {"outputs": {k: list(v) for k, v in out.items()}})
        self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        return


def serve(host="127.0.0.1", port=8421):
    srv = ThreadingHTTPServer((host, port), _Handler)
    print("deepinferforge serving on http://%s:%d" % (host, port), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
