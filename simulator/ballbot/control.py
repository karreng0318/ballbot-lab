"""No-slip LQR, a tilt-limited reference, and a friction-cone projection.

The inner law is an LQR on the rolling model. It does not know mu. The three
outer laws differ in what reference they track and whether torque is clipped
to the floor:

- step: regulate to the goal in one shot. This is the usual first controller.
- tilt: jerk-aware reference whose acceleration is the steady acceleration of
  a 6 degree lean. No floor model.
- mu: that same cap, further limited by kappa * mu * g, with the motor torque
  projected so the no-slip friction stays inside the cone.

In steady rolling, |f| / N = |a| / g. A lean limit and a friction limit are
different numbers, and the torque spike that acquires the lean is larger
than either steady value.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import solve_continuous_are

from ballbot.parameters import CMU, Params
from ballbot.planar import _affine_terms, lean_for_acceleration, linear_model, normal_force, steady_state


class BalanceLQR:
    def __init__(self, p: Params = CMU):
        self.p = p
        self.A, self.B = linear_model(p)
        # Bryson-style weights: 1 m, 0.1 rad, 1 m/s, 1 rad/s, and 25 N m.
        q = np.diag([1.0, 100.0, 1.0, 4.0])
        r = np.array([[1.0 / 25.0**2]])
        gramian = solve_continuous_are(self.A, self.B, q, r)
        self.K = np.linalg.solve(r, self.B.T @ gramian).ravel()

    def torque(
        self,
        state: np.ndarray,
        x_ref: float = 0.0,
        v_ref: float = 0.0,
        theta_ref: float = 0.0,
        dtheta_ref: float = 0.0,
    ) -> float:
        error = np.array(
            [
                state[0] - x_ref,
                state[2] - theta_ref,
                state[3] - v_ref,
                state[5] - dtheta_ref,
            ]
        )
        feedforward = steady_state(theta_ref, self.p)["tau"] if abs(theta_ref) > 1e-9 else 0.0
        tau = float(-self.K @ error + feedforward)
        return float(np.clip(tau, -self.p.tau_max, self.p.tau_max))

    def closed_loop_eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvals(self.A - self.B @ self.K.reshape(1, 4))


class GoalReference:
    """Point-to-point reference that stops at the goal.

    Cruise speed is limited by ``v_max``. Braking speed is sqrt(2 a_max s),
    so a full deceleration at ``a_max`` ends on the goal. Jerk limiting is
    applied only while that braking curve still has margin; once the speed
    is above the curve, deceleration is allowed at the full ``a_max``.
    """

    def __init__(self, goal: float, a_max: float, v_max: float, j_max: float = 1.5):
        self.goal = float(goal)
        self.a_max = max(float(a_max), 1e-4)
        self.v_max = max(float(v_max), 1e-4)
        self.j_max = max(float(j_max), 1e-3)
        self.x = 0.0
        self.v = 0.0
        self.a = 0.0

    def step(self, dt: float) -> tuple[float, float, float]:
        remaining = self.goal - self.x
        if abs(remaining) < 2e-3 and abs(self.v) < 2e-2:
            self.x = self.goal
            self.v = 0.0
            self.a = 0.0
            return self.x, self.v, self.a
        v_brake = math.sqrt(max(0.0, 2.0 * self.a_max * abs(remaining)))
        v_des = math.copysign(min(self.v_max, v_brake), remaining if abs(remaining) > 0.0 else self.v)
        a_track = (v_des - self.v) / max(dt, 1e-4)
        a_cmd = float(np.clip(a_track, -self.a_max, self.a_max))
        overspeed = abs(self.v) > v_brake + 0.02 and self.v * remaining > 0.0
        if overspeed:
            self.a = a_cmd
        else:
            self.a += float(np.clip(a_cmd - self.a, -self.j_max * dt, self.j_max * dt))
            self.a = float(np.clip(self.a, -self.a_max, self.a_max))
        if abs(self.v) >= self.v_max - 1e-6 and self.a * self.v > 0.0:
            self.a = 0.0
        self.v = float(np.clip(self.v + self.a * dt, -self.v_max, self.v_max))
        # Do not let the reference integrate through the goal with leftover speed.
        next_x = self.x + self.v * dt
        if (self.goal - self.x) * (self.goal - next_x) < 0.0:
            self.x = self.goal
            self.v = 0.0
            self.a = 0.0
        else:
            self.x = next_x
        return self.x, self.v, self.a


def predict_stick(state: np.ndarray, tau: float, p: Params) -> tuple[float, float]:
    """Friction and normal force required to keep rolling, whether or not mu allows it."""
    theta = float(state[2])
    dtheta = float(state[5])
    a0, da = _affine_terms(theta, dtheta, tau, p)
    slip0 = a0[0] - p.r * a0[1]
    slip_f = da[0] - p.r * da[1]
    f_stick = 0.0 if abs(slip_f) < 1e-12 else float(-slip0 / slip_f)
    thdd = a0[2] + da[2] * f_stick
    return f_stick, float(normal_force(theta, dtheta, thdd, p))


def project_torque(state: np.ndarray, tau_des: float, mu: float, p: Params, margin: float = 1.0) -> float:
    """Closest torque to tau_des whose no-slip friction stays inside margin * mu * N.

    f and N are affine in tau, so the cone is two linear inequalities. If no
    torque can honor the cone, return the one with the smallest |f| / N.
    """
    mu_use = max(margin * mu, 1e-4)
    f0, n0 = predict_stick(state, 0.0, p)
    f1, n1 = predict_stick(state, 1.0, p)
    df = f1 - f0
    dn = n1 - n0

    def satisfies(tau: float) -> bool:
        normal = n0 + dn * tau
        if normal < 20.0:
            return False
        return abs(f0 + df * tau) <= mu_use * normal + 1e-6

    # |f0 + df tau| <= mu_use (n0 + dn tau)
    bounds = [(-p.tau_max, p.tau_max)]

    def add(lhs: float, coef: float) -> None:
        # lhs <= coef * tau
        if abs(coef) < 1e-10:
            if lhs > 1e-8:
                bounds.append((1.0, -1.0))  # empty
            return
        edge = lhs / coef
        if coef > 0.0:
            bounds.append((edge, np.inf))
        else:
            bounds.append((-np.inf, edge))

    add(f0 - mu_use * n0, mu_use * dn - df)
    add(-f0 - mu_use * n0, df + mu_use * dn)
    # N >= 20
    add(20.0 - n0, dn)

    lo = max(b[0] for b in bounds)
    hi = min(b[1] for b in bounds)
    if lo <= hi:
        return float(np.clip(tau_des, lo, hi))

    # Nothing stays inside the cone. Minimize the friction ratio on a grid.
    candidates = np.linspace(-p.tau_max, p.tau_max, 81)
    best_tau = 0.0
    best_ratio = float("inf")
    for tau in candidates:
        normal = n0 + dn * tau
        if normal < 20.0:
            continue
        ratio = abs(f0 + df * tau) / normal
        if ratio < best_ratio:
            best_ratio = ratio
            best_tau = float(tau)
    return best_tau


class StepController:
    """Classical LQR to a fixed goal. The reference step ignores mu."""

    def __init__(self, goal: float, p: Params = CMU, inner: BalanceLQR | None = None):
        self.goal = goal
        self.inner = inner or BalanceLQR(p)

    def __call__(self, t: float, state: np.ndarray) -> float:
        del t
        return self.inner.torque(state, x_ref=self.goal, v_ref=0.0)


class ShapedController:
    """LQR plus a jerk-limited reference.

    ``mu is None`` is the tilt-only planner: acceleration is capped by the
    steady acceleration at ``tilt_max``, and the torque is not projected.
    That is a tilt limit with no floor model.

    A numeric ``mu`` also caps acceleration by ``kappa * mu * g`` and
    projects the torque into the friction cone, so the lean-acquisition
    spike cannot demand more shear than the floor can give.
    """

    def __init__(
        self,
        goal: float,
        mu: float | None,
        p: Params = CMU,
        tilt_max: float = math.radians(6.0),
        kappa: float = 0.65,
        v_max: float = 0.7,
        j_max: float = 1.2,
        margin: float = 1.0,
        inner: BalanceLQR | None = None,
    ):
        self.p = p
        self.mu = mu
        self.margin = margin
        self.inner = inner or BalanceLQR(p)
        a_tilt = abs(steady_state(tilt_max, p)["a"])
        if mu is None:
            a_max = a_tilt
        else:
            a_max = min(a_tilt, kappa * mu * p.g)
        self.a_max = float(a_max)
        self.reference = GoalReference(goal, a_max=self.a_max, v_max=v_max, j_max=j_max)
        self._initialized = False

    def __call__(self, t: float, state: np.ndarray) -> float:
        del t
        if not self._initialized:
            self.reference.x = float(state[0])
            self.reference.v = float(np.clip(state[3], -self.reference.v_max, self.reference.v_max))
            self._initialized = True
        x_ref, v_ref, a_ref = self.reference.step(self.dt)
        theta_ref = lean_for_acceleration(a_ref, self.p)
        tau = self.inner.torque(state, x_ref=x_ref, v_ref=v_ref, theta_ref=theta_ref)
        if self.mu is None:
            return tau
        return project_torque(state, tau, self.mu, self.p, margin=self.margin)

    dt: float = 0.002


def hold_controller(p: Params = CMU, inner: BalanceLQR | None = None):
    law = inner or BalanceLQR(p)

    def torque(t: float, state: np.ndarray) -> float:
        del t
        return law.torque(state)

    return torque
