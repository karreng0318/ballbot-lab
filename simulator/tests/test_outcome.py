"""Locks the headline of the planar encoder-feedback comparison."""

from __future__ import annotations

import unittest

from experiments.run_study import GOAL, arrived, encoder_feedback, make_controller
from ballbot.control import BalanceLQR
from ballbot.parameters import CMU
from ballbot.planar import encoder_position, simulate


class OutcomeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.law = BalanceLQR(CMU)

    def _run(self, kind, mu):
        ctrl = encoder_feedback(make_controller(kind, mu, self.law))
        return simulate(mu, ctrl, 14.0, 0.002)

    def test_slippery_floor_splits_the_three_laws(self):
        step = self._run("step", 0.15)
        tilt = self._run("tilt", 0.15)
        aware = self._run("mu", 0.15)
        self.assertTrue(step.fallen)
        self.assertTrue(tilt.fallen)
        self.assertTrue(arrived(aware))
        self.assertAlmostEqual(encoder_position(aware, CMU)[-1], GOAL, delta=0.05)

    def test_sticky_floor_all_three_arrive(self):
        for kind in ("step", "tilt", "mu"):
            trace = self._run(kind, 1.0)
            self.assertTrue(arrived(trace), kind)


if __name__ == "__main__":
    unittest.main()
