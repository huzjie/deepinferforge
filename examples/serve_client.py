# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""A minimal stdlib client for the deepinferforge HTTP server.

Start the server first::

    python -m deepinferforge serve --port 8421

Then run this script.
"""
import json
import urllib.request

BASE = "http://127.0.0.1:8421"


def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


if __name__ == "__main__":
    print("health:", get("/health"))
    print("models:", get("/v1/models"))
    print("backends:", get("/v1/backends"))
    compiled = post("/v1/compile", {"id": "demo",
                                    "model": {"dim": 16, "n_layers": 2}})
    print("compiled:", compiled)
    ran = post("/v1/run", {"id": "demo"})
    print("run outputs (keys):", list(ran["outputs"]))
