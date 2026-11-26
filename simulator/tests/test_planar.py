"""Tests for the planar stick-slip model and the no-slip LQR."""

from __future__ import annotations

import math
import unittest

import numpy as np

from ballbot.control import BalanceLQR
from ballbot.parameters import CMU
from ballbot.planar import (
    accelerations,
    energy,
    linear_model,
    noslip_acc,
    resolve_friction,
    steady_state,
    step,
)


class ReferenceTest(unittest.TestCase):
    def test_reference_stops_on_the_goal(self):
        from ballbot.control import GoalReference

        ref = GoalReference(goal=2.0, a_max=0.8, v_max=0.7, j_max=1.2)
        dt = 0.002
        xs = []
        for _ in range(12000):
            x, v, a = ref.step(dt)
            xs.append(x)
            self.assertLessEqual(abs(a), 0.8 + 1e-6)
            self.assertLessEqual(abs(v), 0.7 + 1e-6)
        self.assertAlmostEqual(x, 2.0, places=2)
        self.assertAlmostEqual(v, 0.0, places=2)
        self.assertLess(max(xs), 2.05)


class DynamicsTest(unittest.TestCase):
    def test_upright_equilibrium_is_unstably_leaning(self):
        a, _ = linear_model(CMU)
        self.assertGreater(a[3, 1], 0.0)
        unstable = np.linalg.eigvals(a)
        self.assertTrue(np.any(np.real(unstable) > 0.1))

    def test_lqr_stabilizes_the_linear_model(self):
        law = BalanceLQR(CMU)
        self.assertTrue(np.all(np.real(law.closed_loop_eigenvalues()) < -0.05))

    def test_stick_matches_rolling_constraint_dynamics(self):
        theta, dtheta, dx, tau = 0.18, -0.4, 0.25, 6.0
        state = np.array([0.0, 0.0, theta, dx, dx / CMU.r, dtheta])
        f, normal, mode = resolve_friction(state, tau, mu=5.0, p=CMU)
        self.assertEqual(mode, "stick")
        self.assertGreater(normal, 100.0)
        acc = accelerations(theta, dtheta, tau, f, CMU)
        xdd, thdd = noslip_acc(theta, dtheta, tau, CMU)
        self.assertAlmostEqual(acc[0], xdd, places=6)
        self.assertAlmostEqual(acc[2], thdd, places=6)
        self.assertAlmostEqual(acc[0] - CMU.r * acc[1], 0.0, places=6)

    def test_low_friction_saturates_at_the_cone(self):
        # A static lean needs little friction until the motor torques the ball.
        # 30 N m at the upright pose asks for more shear than mu = 0.05 allows.
        state = np.zeros(6)
        mu = 0.05
        f, normal, mode = resolve_friction(state, tau=30.0, mu=mu, p=CMU)
        self.assertEqual(mode, "slide")
        self.assertAlmostEqual(abs(f), mu * normal, places=5)
        f_stick, _, stick = resolve_friction(state, tau=30.0, mu=2.0, p=CMU)
        self.assertEqual(stick, "stick")
        self.assertGreater(abs(f_stick), mu * normal)

    def test_steady_friction_coefficient_is_acceleration_over_g(self):
        for degrees in (1.0, 3.0, 8.0):
            steady = steady_state(math.radians(degrees), CMU)
            self.assertAlmostEqual(steady["mu_min"], abs(steady["a"]) / CMU.g, places=6)

    def test_high_friction_does_not_accumulate_slip(self):
        state = np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
        for _ in range(400):
            state, info = step(state, tau=0.0, mu=2.0, dt=0.001, p=CMU)
            self.assertEqual(info["mode"], "stick")
        slip = state[0] - CMU.r * state[1]
        self.assertAlmostEqual(slip, 0.0, places=6)

    def test_no_slip_energy_is_nearly_conserved(self):
        # Reduced rolling model, RK4, no torque. Static friction does no work.
        p = CMU
        y = np.array([0.0, 0.12, 0.0, 0.0])  # x, theta, dx, dtheta

        def deriv(z):
            xdd, thdd = noslip_acc(z[1], z[3], 0.0, p)
            return np.array([z[2], z[3], xdd, thdd])

        def eng(z):
            state = np.array([z[0], z[0] / p.r, z[1], z[2], z[2] / p.r, z[3]])
            return energy(state, p)

        dt = 0.001
        e0 = eng(y)
        for _ in range(1000):
            k1 = deriv(y)
            k2 = deriv(y + 0.5 * dt * k1)
            k3 = deriv(y + 0.5 * dt * k2)
            k4 = deriv(y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.assertLess(abs(eng(y) - e0) / abs(e0), 2e-4)

    def test_lqr_recovers_from_a_small_lean_on_a_sticky_floor(self):
        law = BalanceLQR(CMU)
        state = np.array([0.0, 0.0, math.radians(2.0), 0.0, 0.0, 0.0])
        for _ in range(3000):
            tau = law.torque(state)
            state, info = step(state, tau, mu=1.0, dt=0.002, p=CMU)
            self.assertFalse(info["fallen"])
            self.assertNotEqual(info["mode"], "liftoff")
        self.assertLess(abs(state[2]), math.radians(0.4))
        self.assertLess(abs(state[0]), 0.05)


if __name__ == "__main__":
    unittest.main()
