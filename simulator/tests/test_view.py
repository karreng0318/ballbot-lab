"""The MuJoCo viewer rollout, without opening a window."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.view_mujoco import LiveRun, render_frame


class ViewTest(unittest.TestCase):
    def test_goal_marker_sits_at_the_goal(self):
        run = LiveRun(mu=1.0, kind="mu", goal=1.5, psi_sign=1.0)
        marker = run.robot.data.mocap_pos[run.robot._goal_mocap]
        self.assertAlmostEqual(float(marker[0]), 1.5, places=3)

    def test_high_friction_starts_toward_the_goal(self):
        run = LiveRun(mu=1.0, kind="mu", goal=1.5, psi_sign=1.0)
        for _ in range(2000):
            run.step()
        self.assertFalse(run.status["fallen"])
        self.assertGreater(run.status["x"], 0.05)

    def test_offscreen_frame_shows_the_body(self):
        run = LiveRun(mu=1.0, kind="mu", goal=1.5, psi_sign=1.0)
        for _ in range(300):
            run.step()
        frame = render_frame(run, width=480, height=270)
        self.assertEqual(frame.shape, (270, 480, 3))
        # Lit torso: red channel clearly above green. Thresholds stay below the blown-out white of an overexposed frame.
        orange = np.logical_and(frame[:, :, 0] > 120, frame[:, :, 0] > frame[:, :, 1] + 25)
        self.assertGreater(int(np.count_nonzero(orange)), 30)


if __name__ == "__main__":
    unittest.main()
