"""The MuJoCo viewer rollout, without opening a window."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.view_mujoco import (
    KEY_3,
    KEY_KP_3,
    LiveRun,
    command_from_key,
    kinds_from_group_change,
    render_frame,
)


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

    def test_three_selects_the_friction_aware_law(self):
        self.assertEqual(command_from_key(KEY_3), "mu")
        self.assertEqual(command_from_key(KEY_KP_3), "mu")
        # MuJoCo starts with geom group 3 disabled and toggles it on key 3.
        seen = [1, 1, 1, 0, 0, 0]
        groups = [1, 1, 1, 1, 0, 0]
        self.assertEqual(kinds_from_group_change(seen, groups), ["mu"])
        self.assertEqual(seen[3], 1)
        groups[3] = 0
        self.assertEqual(kinds_from_group_change(seen, groups), ["mu"])


if __name__ == "__main__":
    unittest.main()
