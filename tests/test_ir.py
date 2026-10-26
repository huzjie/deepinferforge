# -*- coding: utf-8 -*-
import unittest

from deepinferforge.ir.node import Node
from deepinferforge.ir.graph import Graph
from deepinferforge.ir.builder import GraphBuilder
from deepinferforge.ir import dsl


class TestGraph(unittest.TestCase):
    def test_topo_sort(self):
        g = Graph("g")
        g.add_node(Node("a", "const", []))
        g.add_node(Node("b", "relu", ["a"]))
        g.add_node(Node("c", "add", ["a", "b"]))
        g.topo_sort()
        self.assertEqual(g.order.index("a"), 0)
        self.assertLess(g.order.index("a"), g.order.index("b"))
        self.assertLess(g.order.index("b"), g.order.index("c"))

    def test_cycle(self):
        g = Graph("g")
        g.add_node(Node("a", "relu", ["b"]))
        g.add_node(Node("b", "relu", ["a"]))
        with self.assertRaises(ValueError):
            g.topo_sort()


class TestBuilder(unittest.TestCase):
    def test_build(self):
        b = GraphBuilder("m")
        x = b.input("x")
        m = b.matmul(x, b.const(1.0))
        r = b.relu(b.add(m, b.const(0.0)))
        b.output(r)
        g = b.build()
        self.assertGreater(g.op_count(), 0)
        self.assertIn(r, g.outputs)


class TestDSL(unittest.TestCase):
    def test_lower(self):
        g = dsl.lower("m", [
            ("const", [], {"value": 1.0}),
            ("matmul", ["x", "n0_const"]),
        ], inputs=["x"], outputs=["n1_matmul"])
        self.assertEqual(g.op_count(), 2)
        self.assertEqual(g.outputs, ["n1_matmul"])


if __name__ == "__main__":
    unittest.main()
