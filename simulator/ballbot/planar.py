"""Planar ballbot with a stick-slip floor contact.

Coordinates, all in the sagittal plane:

- x: horizontal position of the ball center
- psi: ball rotation about +y. Positive psi rolls the ball toward +x.
- theta: body lean from vertical. Positive theta moves the COM toward +x.

The floor pushes up with normal force N and sideways with friction f, where
f is the force on the ball and is positive toward +x. Slip velocity at the
contact is dx - r dpsi. It is zero when the ball rolls without slip.

The torque tau is applied by the motor from the body onto the ball about +y.
The body feels -tau. With the sign above, positive tau spins the ball toward
+x and pitches the body toward -x.

The no-slip model used by the classical papers is the special case |f| <= mu N
and dx = r dpsi. When the force required to keep that constraint exceeds
mu N, the contact slides and f = -mu N sign(slip).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ballbot.parameters import CMU, Params

V_STICK = 2e-3  # m/s, Karnopp stick band
FALL_ANGLE = 0.55  # rad, about 31.5 deg


def accelerations(theta: float, dtheta: float, tau: float, f: float, p: Params) -> np.ndarray:
    """Return [xdd, psidd, thdd] for a known contact force."""
    c = math.cos(theta)
    s = math.sin(theta)
    mass = np.array(
        [
            [p.m + p.mb, 0.0, p.l * p.m * c],
            [0.0, p.Ib, 0.0],
            [p.l * p.m * c, 0.0, p.I + (p.l**2) * p.m],
        ]
    )
    rhs = np.array(
        [
            p.l * p.m * s * dtheta**2 + f,
            tau - p.r * f,
            p.m * p.g * p.l * s - tau,
        ]
    )
    return np.linalg.solve(mass, rhs)


def normal_force(theta: float, dtheta: float, thdd: float, p: Params) -> float:
    zdd = -p.l * math.sin(theta) * thdd - p.l * (dtheta**2) * math.cos(theta)
    return (p.m + p.mb) * p.g + p.m * zdd


def _affine_terms(theta: float, dtheta: float, tau: float, p: Params):
    """Accelerations are affine in f: a(f) = a0 + da * f."""
    a0 = accelerations(theta, dtheta, tau, 0.0, p)
    a1 = accelerations(theta, dtheta, tau, 1.0, p)
    return a0, a1 - a0


def resolve_friction(state: np.ndarray, tau: float, mu: float, p: Params):
    """Pick the contact force and the mode: stick, slide, or liftoff."""
    dx, dpsi, dth = state[3], state[4], state[5]
    theta = float(state[2])
    v_slip = dx - p.r * dpsi
    a0, da = _affine_terms(theta, dth, tau, p)

    def normal_of(f: float) -> float:
        thdd = a0[2] + da[2] * f
        return normal_force(theta, dth, thdd, p)

    slip0 = a0[0] - p.r * a0[1]
    slip_f = da[0] - p.r * da[1]
    f_stick = 0.0 if abs(slip_f) < 1e-12 else -slip0 / slip_f
    n_stick = normal_of(f_stick)

    if abs(v_slip) < V_STICK and n_stick > 1.0 and abs(f_stick) <= mu * n_stick + 1e-6:
        return float(f_stick), float(n_stick), "stick"

    # Sliding friction opposes slip. At the edge of the stick band, the
    # impending direction is the one the unsaturated stick force was fighting.
    if abs(v_slip) >= V_STICK:
        slip_sign = 1.0 if v_slip > 0.0 else -1.0
    else:
        slip_sign = -1.0 if f_stick >= 0.0 else 1.0

    # f = -mu * slip_sign * N, and N = n0 + n1 * f.
    n1 = -p.m * p.l * math.sin(theta) * da[2]
    n0 = normal_of(0.0)
    denom = 1.0 + mu * slip_sign * n1
    if abs(denom) < 1e-9:
        return 0.0, float(n0), "liftoff"
    f = (-mu * slip_sign * n0) / denom
    normal = n0 + n1 * f
    if normal <= 1.0:
        return 0.0, float(normal), "liftoff"
    return float(f), float(normal), "slide"


def step(state: np.ndarray, tau: float, mu: float, dt: float, p: Params):
    tau = float(np.clip(tau, -p.tau_max, p.tau_max))
    f, normal, mode = resolve_friction(state, tau, mu, p)
    info = {"tau": tau, "f": f, "N": normal, "mode": mode}
    if mode == "liftoff" or abs(float(state[2])) > FALL_ANGLE:
        info["fallen"] = True
        return state.copy(), info

    acc = accelerations(float(state[2]), float(state[5]), tau, f, p)
    vel = state[3:6] + acc * dt
    if mode == "stick":
        vel[1] = vel[0] / p.r
    pos = state[0:3] + vel * dt
    new_state = np.concatenate([pos, vel])
    info["fallen"] = abs(float(new_state[2])) > FALL_ANGLE
    return new_state, info


def noslip_acc(theta: float, dtheta: float, tau: float, p: Params) -> tuple[float, float]:
    """Horizontal and lean acceleration with the rolling constraint enforced."""
    m1 = p.m + p.mb + p.Ib / p.r**2
    couple = p.m * p.l * math.cos(theta)
    ip = p.I + p.m * p.l**2
    rhs1 = p.m * p.l * math.sin(theta) * dtheta**2 + tau / p.r
    rhs2 = p.m * p.g * p.l * math.sin(theta) - tau
    acc = np.linalg.solve(np.array([[m1, couple], [couple, ip]]), np.array([rhs1, rhs2]))
    return float(acc[0]), float(acc[1])


def steady_state(theta: float, p: Params) -> dict[str, float]:
    """Constant lean, constant acceleration, rolling without slip.

    In this regime the normal force is the weight, so the friction the floor
    must supply is exactly |a| / g. The lean that produces a given a depends
    on the robot; the friction coefficient does not.
    """
    s = math.sin(theta)
    c = math.cos(theta)
    denom = p.m * p.l * c + p.r * (p.m + p.mb) + p.Ib / p.r
    a = (p.m * p.g * p.l * s) / denom
    f = (p.m + p.mb) * a
    tau = p.r * f + p.Ib * a / p.r
    normal = (p.m + p.mb) * p.g
    return {
        "a": a,
        "tau": tau,
        "f": f,
        "N": normal,
        "mu_min": abs(f) / normal,
        "theta": theta,
    }


def lean_for_acceleration(a_des: float, p: Params) -> float:
    """Lean angle whose steady acceleration is a_des."""
    m1 = p.m + p.mb + p.Ib / p.r**2
    theta = a_des * (m1 * p.r + p.m * p.l) / (p.m * p.g * p.l)
    for _ in range(6):
        a = steady_state(theta, p)["a"]
        a_eps = steady_state(theta + 1e-5, p)["a"]
        slope = (a_eps - a) / 1e-5
        if abs(slope) < 1e-9:
            break
        theta += (a_des - a) / slope
    return float(theta)


def energy(state: np.ndarray, p: Params) -> float:
    dx, dpsi, dth = state[3], state[4], state[5]
    theta = float(state[2])
    vxc = dx + p.l * dth * math.cos(theta)
    vzc = -p.l * dth * math.sin(theta)
    kinetic = (
        0.5 * p.mb * dx**2
        + 0.5 * p.Ib * dpsi**2
        + 0.5 * p.m * (vxc**2 + vzc**2)
        + 0.5 * p.I * dth**2
    )
    potential = p.m * p.g * p.l * math.cos(theta)
    return float(kinetic + potential)


def linear_model(p: Params) -> tuple[np.ndarray, np.ndarray]:
    """State matrix of the no-slip system at the upright equilibrium.

    State order is [x, theta, dx, dtheta]. One positive eigenvalue is the
    inverted-pendulum instability.
    """
    m1 = p.m + p.mb + p.Ib / p.r**2
    couple = p.m * p.l
    ip = p.I + p.m * p.l**2
    inertia = np.array([[m1, couple], [couple, ip]])
    from_theta = np.linalg.solve(inertia, np.array([0.0, p.m * p.g * p.l]))
    from_tau = np.linalg.solve(inertia, np.array([1.0 / p.r, -1.0]))
    a = np.zeros((4, 4))
    a[0, 2] = 1.0
    a[1, 3] = 1.0
    a[2, 1] = from_theta[0]
    a[3, 1] = from_theta[1]
    b = np.zeros((4, 1))
    b[2, 0] = from_tau[0]
    b[3, 0] = from_tau[1]
    return a, b


@dataclass
class Trace:
    t: np.ndarray
    x: np.ndarray
    psi: np.ndarray
    theta: np.ndarray
    dx: np.ndarray
    dpsi: np.ndarray
    dtheta: np.ndarray
    tau: np.ndarray
    friction: np.ndarray
    normal: np.ndarray
    mode: np.ndarray
    fallen: bool
    t_fall: float

    def slip_speed(self, radius: float) -> np.ndarray:
        return self.dx - radius * self.dpsi

    def friction_ratio(self) -> np.ndarray:
        return np.abs(self.friction) / np.maximum(self.normal, 1.0)


def simulate(mu: float, controller, t_final: float, dt: float, p: Params = CMU, x0: np.ndarray | None = None) -> Trace:
    state = np.zeros(6) if x0 is None else np.array(x0, dtype=float).copy()
    n = int(round(t_final / dt))
    out = {k: np.zeros(n) for k in ("t", "x", "psi", "theta", "dx", "dpsi", "dtheta", "tau", "f", "N")}
    modes = np.empty(n, dtype=object)
    fallen = False
    t_fall = math.nan
    for i in range(n):
        t = i * dt
        tau_cmd = 0.0 if fallen else float(controller(t, state))
        if not fallen:
            state, info = step(state, tau_cmd, mu, dt, p)
            if info["fallen"] or info["mode"] == "liftoff":
                fallen = True
                t_fall = t
        else:
            info = {"tau": 0.0, "f": 0.0, "N": 0.0, "mode": "fallen"}
        out["t"][i] = t
        out["x"][i] = state[0]
        out["psi"][i] = state[1]
        out["theta"][i] = state[2]
        out["dx"][i] = state[3]
        out["dpsi"][i] = state[4]
        out["dtheta"][i] = state[5]
        out["tau"][i] = info["tau"]
        out["f"][i] = info["f"]
        out["N"][i] = info["N"]
        modes[i] = info["mode"]
    return Trace(
        t=out["t"],
        x=out["x"],
        psi=out["psi"],
        theta=out["theta"],
        dx=out["dx"],
        dpsi=out["dpsi"],
        dtheta=out["dtheta"],
        tau=out["tau"],
        friction=out["f"],
        normal=out["N"],
        mode=modes,
        fallen=fallen,
        t_fall=t_fall,
    )


def encoder_position(trace: Trace, p: Params) -> np.ndarray:
    return p.r * trace.psi
