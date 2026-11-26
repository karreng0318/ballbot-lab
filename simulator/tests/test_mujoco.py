"""MuJoCo sign and contact checks. These need a display-free MuJoCo build."""

from __future__ import annotations

import math
import unittest

import numpy as np

from ballbot.control import BalanceLQR
from ballbot.mujoco_env import MujocoBallbot


class MujocoTest(unittest.TestCase):
    def test_sliding_mu_is_writable(self):
        robot = MujocoBallbot(mu=0.37)
        self.assertAlmostEqual(float(robot.model.pair_friction[robot.pair, 0]), 0.37)
        self.assertAlmostEqual(float(robot.model.pair_friction[robot.pair, 1]), 0.37)

    def test_positive_planar_torque_reduces_pitch(self):
        robot = MujocoBallbot(mu=1.2)
        robot.reset(pitch=0.0)
        pitch0 = robot.state()["pitch"]
        for _ in range(80):
            robot.step(tau_pitch=15.0)
        self.assertLess(robot.state()["pitch"], pitch0 - 1e-4)

    def test_lqr_holds_a_two_degree_pitch_on_high_friction(self):
        robot = MujocoBallbot(mu=1.0)
        law = BalanceLQR()
        robot.reset(pitch=math.radians(2.0))
        pitches = []
        for _ in range(2500):
            robot.apply_balance(law)
            pitches.append(robot.state()["pitch"])
            self.assertLess(abs(robot.state()["pitch"]), math.radians(25))
            self.assertLess(abs(robot.state()["roll"]), math.radians(25))
        self.assertLess(abs(pitches[-1]), math.radians(1.0))
        self.assertLess(max(abs(p) for p in pitches[500:]), math.radians(8))


if __name__ == "__main__":
    unittest.main()
