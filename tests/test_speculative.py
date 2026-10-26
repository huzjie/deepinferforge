# -*- coding: utf-8 -*-
import unittest

from deepinferforge.speculative.verify import (verify, expected_tokens, speedup,
                                               acceptance_for_skill)


class TestSpeculative(unittest.TestCase):
    def test_verify(self):
        self.assertEqual(verify([1, 2, 3], [True, True, False]), 2)

    def test_expected_limit(self):
        self.assertAlmostEqual(expected_tokens(1.0, 4), 5.0, places=6)

    def test_expected_partial(self):
        self.assertAlmostEqual(expected_tokens(0.5, 2), 1.75, places=6)

    def test_speedup_monotonic(self):
        self.assertGreater(speedup(1.0, 4, 0.2), speedup(0.5, 4, 0.2))

    def test_acceptance_monotonic(self):
        self.assertAlmostEqual(acceptance_for_skill(1.0, 0.3), 1.0, places=6)
        self.assertLess(acceptance_for_skill(0.5, 0.3), 1.0)


if __name__ == "__main__":
    unittest.main()
