# -*- coding: utf-8 -*-
"""A tiny declarative DSL for spelling out models as data.

Instead of writing imperative builder code, a model can be declared as a list
of ``(op, inputs, attrs)`` tuples; :func:`lower` compiles them into a
:class:`~deepinferforge.ir.graph.Graph`. This makes the IR easy to serialize,
diff and hand-edit -- which is exactly what you want for a cross-platform
representation.
"""
from .graph import Graph
from .node import Node
from .dtypes import from_name


def lower(name, steps, inputs=None, outputs=None, dtype="float32"):
    g = Graph(name)
    for inp in (inputs or []):
        g.mark_input(inp)
    for i, step in enumerate(steps):
        op = step[0]
        ins = step[1] if len(step) > 1 else []
        attrs = step[2] if len(step) > 2 else {}
        node = Node("n%d_%s" % (i, op), op, ins, attrs, dtype=from_name(dtype))
        g.add_node(node)
    for out in (outputs or []):
        g.mark_output(out)
    g.topo_sort()
    return g


def linear_relu(name, in_name, out_name="out", width=4):
    """The canonical fusion candidate: matmul + add(bias) + relu."""
    return lower(
        name,
        [
            ("const", [], {"value": 1.0}),
            ("matmul", [in_name, "n0_const"]),
            ("add", ["n1_matmul", "n0_const"]),
            ("relu", ["n2_add"]),
        ],
        inputs=[in_name], outputs=[out_name],
    )
