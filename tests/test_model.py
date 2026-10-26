# -*- coding: utf-8 -*-
import unittest

from deepinferforge.model.tensor import Tensor
from deepinferforge.model.transformer import Transformer


class TestTensor(unittest.TestCase):
    def test_matmul_shape(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[1.0, 0.0], [0.0, 1.0]])
        c = a.matmul(b)
        self.assertEqual(c.shape, (2, 2))
        self.assertAlmostEqual(c.data[0][0], 1.0)
        self.assertAlmostEqual(c.data[1][1], 4.0)

    def test_softmax_sums_to_one(self):
        t = Tensor([[1.0, 2.0, 3.0]]).softmax()
        self.assertAlmostEqual(sum(t.data[0]), 1.0, places=6)

    def test_layernorm_zero_mean(self):
        t = Tensor([[1.0, 3.0, 5.0]]).layernorm()
        self.assertAlmostEqual(sum(t.data[0]), 0.0, places=6)


class TestTransformer(unittest.TestCase):
    def test_forward_shape(self):
        tf = Transformer(dim=16, n_layers=2, n_heads=2, head_dim=8,
                         vocab=32, use_moe=True, n_experts=4, top_k=2)
        logits = tf.forward([1, 2, 3, 4])
        self.assertEqual(logits.shape, (4, 32))

    def test_sample_in_vocab(self):
        tf = Transformer(dim=16, n_layers=1, n_heads=2, head_dim=8, vocab=32)
        tok = tf.sample([1, 2, 3], temperature=1.0, seed="t")
        self.assertGreaterEqual(tok, 0)
        self.assertLess(tok, 32)


if __name__ == "__main__":
    unittest.main()
