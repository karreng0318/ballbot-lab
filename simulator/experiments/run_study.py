"""The comparison this project is built around.

Three controllers, same inner LQR, same 2 m goal, CMU parameters. Feedback
position and velocity come from the ball encoder under the no-slip assumption
(x = r psi). That is the sensor a roller-driven ballbot actually has. The
floor friction coefficient is the only thing that changes.

Outputs land in results/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ballbot.control import BalanceLQR, ShapedController, StepController, hold_controller
from ballbot.mujoco_env import MujocoBallbot
from ballbot.parameters import CMU
from ballbot.planar import encoder_position, simulate

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
GOAL = 2.0
DT = 0.002
T_FINAL = 14.0
MUS = (0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00)
COLORS = {"step": "#9a3412", "tilt": "#b45309", "mu": "#1d4e89"}
LABELS = {
    "step": "Step LQR",
    "tilt": "Tilt-limited",
    "mu": "Friction-aware",
}


def encoder_feedback(controller):
    """Hide the true base position. The controller sees roller odometry."""

    def torque(t, state):
        sensed = np.array(state, dtype=float, copy=True)
        sensed[0] = CMU.r * state[1]
        sensed[3] = CMU.r * state[4]
        return controller(t, sensed)

    return torque


def make_controller(kind: str, mu: float, law: BalanceLQR):
    if kind == "step":
        return StepController(GOAL, inner=law)
    if kind == "tilt":
        ctrl = ShapedController(GOAL, mu=None, inner=law)
    elif kind == "mu":
        ctrl = ShapedController(GOAL, mu=mu, inner=law)
    else:
        raise KeyError(kind)
    ctrl.dt = DT
    return ctrl


def arrived(trace) -> bool:
    return (not trace.fallen) and abs(trace.x[-1] - GOAL) < 0.08 and abs(trace.dx[-1]) < 0.08


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def plot_sweep(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharex=True)
    for kind in ("step", "tilt", "mu"):
        subset = [r for r in rows if r["kind"] == kind]
        mus = [r["mu"] for r in subset]
        err = [min(abs(r["x"] - GOAL), 2.5) for r in subset]
        odo = [min(abs(r["odo_err"]), 50) for r in subset]
        axes[0].plot(mus, err, color=COLORS[kind], marker="o", label=LABELS[kind])
        axes[1].plot(mus, odo, color=COLORS[kind], marker="o", label=LABELS[kind])
        for r, e, o in zip(subset, err, odo):
            if r["fallen"]:
                axes[0].scatter([r["mu"]], [e], s=80, facecolors="none", edgecolors=COLORS[kind], linewidths=1.6, zorder=3)
                axes[1].scatter([r["mu"]], [o], s=80, facecolors="none", edgecolors=COLORS[kind], linewidths=1.6, zorder=3)
    axes[0].set_ylabel("True position error at 14 s (m)")
    axes[1].set_ylabel("Encoder error, capped at 50 m")
    axes[0].set_xlabel("Floor friction coefficient μ")
    axes[1].set_xlabel("Floor friction coefficient μ")
    axes[0].set_title("Did the base reach 2 m?")
    axes[1].set_title("Did the rollers agree?")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Open circles are falls. Feedback is roller odometry, not motion capture.", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "encoder_sweep.png")
    plt.close(fig)


def plot_timeseries(mu: float, law: BalanceLQR) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.2), sharex=True)
    for kind in ("step", "tilt", "mu"):
        ctrl = encoder_feedback(make_controller(kind, mu, law))
        tr = simulate(mu, ctrl, T_FINAL, DT)
        n = len(tr.t) if not tr.fallen else max(1, int(tr.t_fall / DT))
        t = tr.t[:n]
        axes[0].plot(t, tr.x[:n], color=COLORS[kind], label=LABELS[kind])
        axes[1].plot(t, encoder_position(tr, CMU)[:n], color=COLORS[kind])
        axes[2].plot(t, np.degrees(tr.theta[:n]), color=COLORS[kind])
        if tr.fallen:
            axes[2].scatter([tr.t_fall], [np.degrees(tr.theta[min(n, len(tr.theta)) - 1])], color=COLORS[kind], marker="x", s=40, zorder=3)
    axes[0].axhline(GOAL, color="0.5", lw=0.8, ls="--")
    axes[0].set_ylim(-0.6, 2.6)
    axes[1].set_ylim(-1.0, 3.2)
    axes[0].set_ylabel("True x (m)")
    axes[1].set_ylabel("Encoder x (m)")
    axes[2].set_ylabel("Lean (deg)")
    axes[2].set_xlabel("Time (s)")
    axes[0].set_title(f"μ = {mu:.2f}. Encoder axis is clipped; traces that leave it are tens of meters off.")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / f"timeseries_mu{mu:.2f}.png")
    plt.close(fig)


def identify(law: BalanceLQR) -> list[dict]:
    """Noise-free check: during sliding, the ball equation recovers μ.

    f = (tau - Ib * psidd) / r, and kinetic friction is mu * N. The match
    should be essentially exact. Hardware will have to filter psidd and
    decide that slip has started; the algebra is not the hard part.
    """
    found = []
    for mu in (0.10, 0.15, 0.20, 0.30, 0.45, 0.70):
        probe = ShapedController(goal=3.0, mu=None, inner=law, v_max=1.2, tilt_max=math.radians(8.0))
        probe.dt = DT
        tr = simulate(mu, probe, 6.0, DT)
        slip = np.abs(tr.slip_speed(CMU.r))
        hits = np.where(slip > 0.05)[0]
        if len(hits) == 0:
            found.append({"mu": mu, "mu_hat": None, "slipped": False})
            continue
        hats = []
        for j in hits[:60]:
            if j < 1:
                continue
            psidd = (tr.dpsi[j] - tr.dpsi[j - 1]) / DT
            f_est = (tr.tau[j] - CMU.Ib * psidd) / CMU.r
            hats.append(abs(f_est) / max(tr.normal[j], 1.0))
        found.append(
            {
                "mu": mu,
                "mu_hat": float(np.median(hats)),
                "slipped": True,
                "t_slip": float(tr.t[hits[0]]),
            }
        )
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    xs = [r["mu"] for r in found if r["mu_hat"] is not None]
    ys = [r["mu_hat"] for r in found if r["mu_hat"] is not None]
    ax.plot([0, 0.8], [0, 0.8], color="0.6", lw=0.8, label="exact")
    ax.scatter(xs, ys, color=COLORS["mu"], zorder=3)
    ax.set_xlabel("True μ")
    ax.set_ylabel("Median |f| / N after slip onset")
    ax.set_title("Contact force from the ball equation")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "mu_from_slip.png")
    plt.close(fig)
    return found


def recovery_map(law: BalanceLQR) -> list[dict]:
    leans = (2, 4, 6, 8, 10, 12)
    mus = (0.05, 0.08, 0.12, 0.2, 0.4, 0.8)
    rows = []
    grid = np.zeros((len(mus), len(leans)))
    for i, mu in enumerate(mus):
        for j, deg in enumerate(leans):
            x0 = np.array([0.0, 0.0, math.radians(deg), 0.0, 0.0, 0.0])
            tr = simulate(mu, hold_controller(inner=law), 6.0, DT, x0=x0)
            ok = (not tr.fallen) and abs(tr.theta[-1]) < math.radians(1.2) and abs(tr.x[-1]) < 0.25
            grid[i, j] = 1.0 if ok else 0.0
            rows.append({"mu": mu, "lean_deg": deg, "recovered": bool(ok)})
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, origin="lower", aspect="auto")
    ax.set_xticks(range(len(leans)), [str(d) for d in leans])
    ax.set_yticks(range(len(mus)), [f"{m:.2f}" for m in mus])
    ax.set_xlabel("Initial lean (deg)")
    ax.set_ylabel("μ")
    ax.set_title("In-place recovery, true state, no goal")
    fig.tight_layout()
    fig.savefig(RESULTS / "recovery_map.png")
    plt.close(fig)
    return rows


def mujoco_move(mu: float, kind: str, law: BalanceLQR, goal: float = 1.5, t_final: float = 8.0) -> dict:
    robot = MujocoBallbot(mu=mu)
    robot.reset(pitch=0.0)
    if kind == "step":
        ctrl = StepController(goal, inner=law)
        dt = robot.model.opt.timestep
    else:
        shaped_mu = None if kind == "tilt" else mu
        ctrl = ShapedController(goal, mu=shaped_mu, inner=law)
        ctrl.dt = float(robot.model.opt.timestep)
        dt = ctrl.dt
    n = int(t_final / dt)
    xs, encs, leans, ts = [], [], [], []
    fallen = False
    t_fall = None
    psi_sign = getattr(mujoco_move, "psi_sign", 1.0)
    for i in range(n):
        s = robot.state()
        if abs(s["pitch"]) > 0.55 or abs(s["roll"]) > 0.4 or s["z"] < 0.05:
            fallen = True
            t_fall = i * dt
            break
        psi = psi_sign * robot._psi if hasattr(robot, "_psi") else 0.0
        wy = psi_sign * s["wy"]
        sensed = np.array([CMU.r * psi, psi, s["pitch"], CMU.r * wy, wy, s["pitch_rate"]])
        tau_pitch = float(ctrl(i * dt, sensed))
        roll_state = np.array([s["y"], 0.0, s["roll"], s["vy"], 0.0, s["roll_rate"]])
        tau_roll = law.torque(roll_state)
        robot.step(tau_pitch, tau_roll, -6.0 * s["yaw_rate"] - 3.0 * s["yaw"])
        xs.append(s["x"])
        encs.append(CMU.r * psi)
        leans.append(math.degrees(s["pitch"]))
        ts.append(i * dt)
    x_end = xs[-1] if xs else 0.0
    return {
        "mu": mu,
        "kind": kind,
        "x": x_end,
        "encoder": encs[-1] if encs else 0.0,
        "fallen": fallen,
        "t_fall": t_fall,
        "t": ts,
        "xs": xs,
        "encs": encs,
        "leans": leans,
        "arrived": (not fallen) and abs(x_end - goal) < 0.15,
    }


def calibrate_psi_sign(law: BalanceLQR) -> float:
    """Rolling toward +x must increase the integrated spin we treat as psi."""
    trial = mujoco_move(1.2, "mu", law, goal=0.4, t_final=1.5)
    # Re-run a short push and compare vx with wy by inspecting a fresh robot.
    robot = MujocoBallbot(mu=1.2)
    robot.reset(pitch=math.radians(1.0))
    for _ in range(400):
        robot.apply_balance(law, x_ref=0.3)
    s = robot.state()
    # After a move toward +x, wy and vx should share a sign if psi_sign is +1.
    sign = 1.0 if s["vx"] * s["wy"] >= 0.0 else -1.0
    mujoco_move.psi_sign = sign
    del trial
    return sign


def plot_mujoco(runs: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    for ax, mu in zip(axes, (0.12, 1.0)):
        for run in runs:
            if abs(run["mu"] - mu) > 1e-9:
                continue
            n = len(run["t"])
            ax.plot(run["t"], run["xs"], color=COLORS[run["kind"]], label=LABELS[run["kind"]])
            if run["fallen"] and n:
                ax.scatter([run["t"][-1]], [run["xs"][-1]], marker="x", color=COLORS[run["kind"]], zorder=3)
        ax.axhline(1.5, color="0.5", lw=0.8, ls="--")
        ax.set_title(f"MuJoCo contact, μ = {mu:.2f}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("True x (m)")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Same three laws on the contact model. Crosses mark a fall.", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "mujoco_moves.png")
    plt.close(fig)


def write_report(rows, ident, recovery, mj_rows, psi_sign) -> None:
    lines = [
        "# Study results",
        "",
        "Generated by `python3 -m experiments.run_study` from `simulator/`.",
        "Feedback for the move is roller odometry. Parameters are the CMU ballbot.",
        "Goal is 2 m in the planar model and 1.5 m in MuJoCo. A run arrives if it has not fallen, is within 8 cm of the goal, and is slower than 8 cm/s.",
        "",
        "## Planar move, encoder feedback",
        "",
        "| μ | controller | arrived | fell | true x (m) | encoder x (m) | peak lean (deg) | time in slip |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['mu']:.2f} | {r['kind']} | {r['arrived']} | {r['fallen']} | {r['x']:.3f} | {r['encoder']:.3f} | {r['peak_lean_deg']:.1f} | {r['slip_fraction']:.0%} |"
        )
    lines += ["", "## Slip-onset μ check (noise-free)", ""]
    for r in ident:
        hat = "no slip" if r["mu_hat"] is None else f"{r['mu_hat']:.3f}"
        lines.append(f"- true μ = {r['mu']:.2f}, estimate {hat}")
    n_ok = sum(1 for r in recovery if r["recovered"])
    lines += [
        "",
        f"In-place recovery recovered {n_ok} of {len(recovery)} (μ, lean) pairs. See `recovery_map.png`.",
        "",
        f"## MuJoCo, encoder feedback, psi sign {psi_sign:+.0f}",
        "",
        "| μ | controller | arrived | fell | true x (m) | encoder x (m) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in mj_rows:
        lines.append(
            f"| {r['mu']:.2f} | {r['kind']} | {r['arrived']} | {r['fallen']} | {r['x']:.3f} | {r['encoder']:.3f} |"
        )
    lines.append("")
    (RESULTS / "STUDY.md").write_text("\n".join(lines))


def main() -> None:
    style()
    RESULTS.mkdir(parents=True, exist_ok=True)
    law = BalanceLQR(CMU)
    rows = []
    for mu in MUS:
        for kind in ("step", "tilt", "mu"):
            ctrl = encoder_feedback(make_controller(kind, mu, law))
            tr = simulate(mu, ctrl, T_FINAL, DT)
            enc = encoder_position(tr, CMU)
            rows.append(
                {
                    "mu": mu,
                    "kind": kind,
                    "arrived": arrived(tr),
                    "fallen": bool(tr.fallen),
                    "x": float(tr.x[-1]),
                    "dx": float(tr.dx[-1]),
                    "encoder": float(enc[-1]),
                    "odo_err": float(enc[-1] - tr.x[-1]),
                    "peak_lean_deg": float(np.degrees(np.max(np.abs(tr.theta)))),
                    "slip_fraction": float(np.mean(tr.mode == "slide")),
                }
            )
            print(
                f"mu={mu:.2f} {kind:5} arrived={rows[-1]['arrived']} fell={rows[-1]['fallen']} "
                f"x={rows[-1]['x']:.2f} enc={rows[-1]['encoder']:.2f}"
            )
    plot_sweep(rows)
    plot_timeseries(0.15, law)
    plot_timeseries(1.00, law)
    ident = identify(law)
    recovery = recovery_map(law)
    psi_sign = calibrate_psi_sign(law)
    print("mujoco psi sign", psi_sign)
    mj_full = []
    for mu in (0.12, 1.0):
        for kind in ("step", "tilt", "mu"):
            run = mujoco_move(mu, kind, law)
            mj_full.append(run)
            print(f"mujoco mu={mu:.2f} {kind:5} arrived={run['arrived']} fell={run['fallen']} x={run['x']:.2f}")
    plot_mujoco(mj_full)
    mj_rows = [{k: run[k] for k in ("mu", "kind", "x", "encoder", "fallen", "arrived")} for run in mj_full]
    write_report(rows, ident, recovery, mj_rows, psi_sign)
    serializable = {"planar": rows, "identify": ident, "recovery": recovery, "mujoco": mj_rows, "psi_sign": psi_sign}
    (RESULTS / "summary.json").write_text(json.dumps(serializable, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
