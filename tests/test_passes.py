# -*- coding: utf-8 -*-
import unittest

from deepinferforge.ir.builder import GraphBuilder
from deepinferforge.passes.fusion import FusionPass
from deepinferforge.passes.constfold import ConstFoldPass
from deepinferforge.passes.dce import DeadCodeEliminationPass
from deepinferforge.passes.memory import MemoryPlanPass
from deepinferforge.passes.canonicalize import ir_digest


def _fusion_graph():
    b = GraphBuilder("f")
    x = b.input("x")
    w = b.const(2.0)
    m = b.matmul(x, w)
    s = b.add(m, w)
    r = b.relu(s)
    b.output(r)
    return b.build()


class TestFusion(unittest.TestCase):
    def test_fuse_gemm_bias_act(self):
        g = _fusion_graph()
        before = g.op_count()
        removed = FusionPass().run(g)
        self.assertEqual(removed, 1)
        self.assertEqual(g.op_count(), before - 2)  # 3 nodes -> 1
        self.assertTrue(any(n.op == "fused_gemm_bias_act" for n in g.nodes.values()))


class TestConstFold(unittest.TestCase):
    def test_fold_add(self):
        b = GraphBuilder("c")
        a = b.const(2.0)
        c = b.const(3.0)
        s = b.add(a, c)
        b.output(s)
        g = b.build()
        folded = ConstFoldPass().run(g)
        self.assertEqual(folded, 1)
        self.assertEqual(g.nodes[s].op, "const")
        self.assertEqual(g.nodes[s].attrs["value"], 5.0)


class TestDCE(unittest.TestCase):
    def test_remove_dead(self):
        b = GraphBuilder("d")
        x = b.input("x")
        live = b.relu(x)
        w = b.const(1.0)
        dead = b.matmul(x, w)
        b.output(live)
        g = b.build()
        removed = DeadCodeEliminationPass().run(g)
        self.assertEqual(removed, 2)  # matmul + its const are both dead
        self.assertNotIn(dead, g.nodes)
        self.assertNotIn(w, g.nodes)
        self.assertIn(live, g.nodes)


class TestMemoryPlan(unittest.TestCase):
    def test_buffers_saved(self):
        b = GraphBuilder("m")
        x = b.input("x")
        h = x
        for i in range(4):
            h = b.relu(h)
        b.output(h)
        g = b.build()
        saved = MemoryPlanPass().run(g)
        self.assertGreaterEqual(saved, 0)


class TestCanonicalize(unittest.TestCase):
    def test_digest_stable(self):
        g = _fusion_graph()
        d1 = ir_digest(g)
        d2 = ir_digest(g)
        self.assertEqual(d1, d2)


if __name__ == "__main__":
    unittest.main()
