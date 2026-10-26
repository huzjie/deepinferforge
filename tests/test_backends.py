# -*- coding: utf-8 -*-
import unittest

from deepinferforge.backends.registry import list_backends, get_backend
from deepinferforge.frontend.importer import import_model


class TestBackends(unittest.TestCase):
    def test_all_registered(self):
        self.assertIn("cpu", list_backends())
        self.assertIn("nvidia", list_backends())
        self.assertIn("ascend", list_backends())

    def test_consistency(self):
        graph = import_model({"dim": 16, "n_layers": 2, "n_heads": 2,
                              "head_dim": 8, "vocab": 32, "use_moe": True,
                              "n_experts": 4, "top_k": 2})
        outs = {}
        for name in ("cpu", "nvidia", "ascend"):
            backend = get_backend(name)()
            plan = backend.compile(graph)
            outs[name] = backend.run(plan)
        keys = list(outs["cpu"])
        for k in keys:
            ref = outs["cpu"][k]
            self.assertEqual(len(ref), len(outs["nvidia"][k]))
            self.assertEqual(len(ref), len(outs["ascend"][k]))
            for a, b in zip(ref, outs["nvidia"][k]):
                self.assertAlmostEqual(a, b, places=6)
            for a, b in zip(ref, outs["ascend"][k]):
                self.assertAlmostEqual(a, b, places=6)

    def test_kernel_source_differs(self):
        graph = import_model({"dim": 8, "n_layers": 1, "n_heads": 1,
                              "head_dim": 8, "vocab": 16, "use_moe": False})
        nv = get_backend("nvidia")().compile(graph)
        asc = get_backend("ascend")().compile(graph)
        self.assertGreater(nv.n_kernels, 0)
        self.assertEqual(nv.n_kernels, asc.n_kernels)
        # emitted source differs across backends
        nv_src = [k.source for k in nv.kernels]
        asc_src = [k.source for k in asc.kernels]
        self.assertNotEqual(nv_src, asc_src)


if __name__ == "__main__":
    unittest.main()
