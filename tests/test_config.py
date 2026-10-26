# -*- coding: utf-8 -*-
import unittest

from deepinferforge.config import Config, load_config
from deepinferforge.utils.yamlish import parse


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = load_config(None)
        self.assertEqual(cfg.model.dim, 32)
        self.assertEqual(cfg.backend.default_backend, "nvidia")

    def test_from_dict(self):
        cfg = Config.from_dict({"model": {"dim": 64}})
        self.assertEqual(cfg.model.dim, 64)

    def test_yamlish(self):
        data = parse("model:\n  dim: 64\n  n_layers: 2\nbackend:\n  targets: [nvidia, ascend]\n")
        self.assertEqual(data["model"]["dim"], 64)
        self.assertEqual(data["backend"]["targets"], ["nvidia", "ascend"])

    def test_yamlish_block_list(self):
        data = parse("passes:\n  - fusion\n  - dce\n")
        self.assertEqual(data["passes"], ["fusion", "dce"])


if __name__ == "__main__":
    unittest.main()
