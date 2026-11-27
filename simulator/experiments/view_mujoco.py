"""Open the ballbot in the MuJoCo viewer.

The window runs the same three laws as the study, on the sphere-on-plane
contact model. Floor μ and the outer law can be changed while it plays.
Roller housings and the marks on the ball are drawn only: the motor is still
a pure torque at the ball center, and the only contact is the sphere.

From ``simulator/``:

    python3 -m experiments.view_mujoco
    python3 -m experiments.view_mujoco --mu 0.15 --controller step
    python3 -m experiments.view_mujoco --record ../results/mujoco_view.mp4

On macOS the window has to be opened with ``mjpython`` instead of ``python3``.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time

import mujoco
import numpy as np

from ballbot.control import BalanceLQR, ShapedController, StepController
from ballbot.mujoco_env import MujocoBallbot
from ballbot.parameters import CMU

LABELS = {"step": "Step LQR", "tilt": "Tilt-limited", "mu": "Friction-aware"}
KINDS = ("step", "tilt", "mu")
MU_LADDER = (0.08, 0.12, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00)
TRAIL = {
    "step": np.array([0.60, 0.20, 0.07, 0.9]),
    "tilt": np.array([0.71, 0.33, 0.04, 0.9]),
    "mu": np.array([0.11, 0.30, 0.54, 0.9]),
}

# GLFW key codes. The viewer callback reports a press, not a character.
# The numeric keypad is a different code from the top-row digit.
KEY_SPACE = 32
KEY_1, KEY_2, KEY_3 = 49, 50, 51
KEY_KP_1, KEY_KP_2, KEY_KP_3 = 321, 322, 323
KEY_R = 82
KEY_LEFT_BRACKET, KEY_RIGHT_BRACKET = 91, 93
# MuJoCo itself binds 1, 2, and 3 to geom groups, and group 3 starts off.
# The viewer handles that shortcut on its own, so the passive callback often
# never receives 3. The group bit still flips, and that flip is the command.
GROUP_KIND = {1: "step", 2: "tilt", 3: "mu"}


def measure_psi_sign(law: BalanceLQR) -> float:
    """Match the study: rolling toward +x has to increase the spin we call psi."""
    robot = MujocoBallbot(mu=1.2)
    robot.reset(pitch=math.radians(1.0))
    for _ in range(400):
        robot.apply_balance(law, x_ref=0.3)
    state = robot.state()
    return 1.0 if state["vx"] * state["wy"] >= 0.0 else -1.0


def make_controller(kind: str, mu: float, goal: float, law: BalanceLQR, dt: float):
    if kind == "step":
        ctrl = StepController(goal, inner=law)
    elif kind == "tilt":
        ctrl = ShapedController(goal, mu=None, inner=law)
        ctrl.dt = dt
    elif kind == "mu":
        ctrl = ShapedController(goal, mu=mu, inner=law)
        ctrl.dt = dt
    else:
        raise KeyError(kind)
    return ctrl


class LiveRun:
    """One MuJoCo rollout with encoder feedback, matching experiments.run_study."""

    def __init__(self, mu: float, kind: str, goal: float, law: BalanceLQR | None = None, psi_sign: float | None = None):
        self.goal = float(goal)
        self.law = law or BalanceLQR(CMU)
        self.psi_sign = measure_psi_sign(self.law) if psi_sign is None else float(psi_sign)
        self.robot = MujocoBallbot(mu=mu)
        self.dt = float(self.robot.model.opt.timestep)
        self.kind = kind
        self.mu = float(mu)
        self.trail: list[np.ndarray] = []
        self._sample = 0
        self.reset()

    def reset(self) -> None:
        self.robot.set_mu(self.mu)
        self.robot.reset(pitch=0.0)
        self.robot.set_goal(self.goal, 0.0)
        self.ctrl = make_controller(self.kind, self.mu, self.goal, self.law, self.dt)
        self.t = 0.0
        self.fallen = False
        self.trail = []
        self._sample = 0
        self.status = self._status()

    def retarget(self, mu: float | None = None, kind: str | None = None) -> None:
        if mu is not None:
            self.mu = float(mu)
        if kind is not None:
            self.kind = kind
        self.reset()

    def step(self) -> dict:
        if self.fallen:
            return self.status
        state = self.robot.state()
        if abs(state["pitch"]) > 0.55 or abs(state["roll"]) > 0.4 or state["z"] < 0.05:
            self.fallen = True
            self.status = self._status()
            return self.status
        psi = self.psi_sign * getattr(self.robot, "_psi", 0.0)
        wy = self.psi_sign * state["wy"]
        sensed = np.array([CMU.r * psi, psi, state["pitch"], CMU.r * wy, wy, state["pitch_rate"]])
        tau_pitch = float(self.ctrl(self.t, sensed))
        roll_state = np.array([state["y"], 0.0, state["roll"], state["vy"], 0.0, state["roll_rate"]])
        tau_roll = self.law.torque(roll_state)
        tau_yaw = -6.0 * state["yaw_rate"] - 3.0 * state["yaw"]
        self.robot.step(tau_pitch, tau_roll, tau_yaw)
        self.robot.set_goal(self.goal, 0.0)
        self.t += self.dt
        self._sample += 1
        if self._sample % 20 == 0:
            pos = self.robot.state()
            self.trail.append(np.array([pos["x"], pos["y"], 0.02]))
        self.status = self._status()
        return self.status

    def _status(self) -> dict:
        state = self.robot.state()
        psi = self.psi_sign * getattr(self.robot, "_psi", 0.0)
        if self.robot.data.ncon == 0:
            contact = "air"
        else:
            ratio = self.robot.contact_friction_ratio()
            if math.isnan(ratio):
                contact = "rest"
            elif ratio >= 0.92 * max(self.mu, 1e-4):
                contact = "slide"
            else:
                contact = "stick"
        return {
            "x": state["x"],
            "y": state["y"],
            "encoder": CMU.r * psi,
            "pitch_deg": math.degrees(state["pitch"]),
            "roll_deg": math.degrees(state["roll"]),
            "contact": contact,
            "fallen": self.fallen,
            "t": self.t,
        }

    def overlay(self) -> list[tuple]:
        status = self.status
        outcome = "fallen" if status["fallen"] else "up"
        left = "\n".join(
            ["controller", "floor μ", "true x", "encoder x", "lean", "contact", "time"]
        )
        right = "\n".join(
            [
                LABELS[self.kind],
                f"{self.mu:.2f}",
                f"{status['x']:.2f} m",
                f"{status['encoder']:.2f} m",
                f"{status['pitch_deg']:+.1f} deg",
                f"{status['contact']}  {outcome}",
                f"{status['t']:.2f} s",
            ]
        )
        keys = "1 step    2 tilt    3 friction-aware\n[ ] μ    space pause    r reset"
        font = mujoco.mjtFontScale.mjFONTSCALE_150
        return [
            (font, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right),
            (font, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, keys, ""),
        ]


def append_sphere(scene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(pos, dtype=float),
        np.eye(3).ravel(),
        np.asarray(rgba, dtype=float),
    )
    scene.ngeom += 1


def paint_trail(scene, points: list[np.ndarray], rgba: np.ndarray, clear: bool) -> None:
    if clear:
        scene.ngeom = 0
    if not points:
        return
    stride = max(1, len(points) // 70)
    for pos in points[::stride][-70:]:
        append_sphere(scene, pos, 0.018, rgba)


def tracking_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    cam.distance = 1.85
    cam.azimuth = 125.0
    cam.elevation = -16.0
    return cam


def render_frame(run: LiveRun, width: int = 960, height: int = 540) -> np.ndarray:
    """One offscreen RGB frame. Used by tests and by --record."""
    if not hasattr(render_frame, "_renderer") or render_frame._size != (width, height):
        render_frame._renderer = mujoco.Renderer(run.robot.model, height, width)
        render_frame._size = (width, height)
        render_frame._cam = tracking_camera(run.robot.model)
    renderer = render_frame._renderer
    renderer.update_scene(run.robot.data, camera=render_frame._cam)
    paint_trail(renderer.scene, run.trail, TRAIL[run.kind], clear=False)
    return renderer.render().copy()


def write_video(path: str, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise RuntimeError("no frames to write")
    height, width = frames[0].shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode() if proc.stderr is not None else ""
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg exited {code}: {err[-500:]}")


def record(run: LiveRun, path: str, seconds: float, fps: int = 30) -> None:
    stride = max(1, int(round(1.0 / (fps * run.dt))))
    frames: list[np.ndarray] = []
    n = int(seconds / run.dt)
    for i in range(n):
        run.step()
        if i % stride == 0:
            frames.append(render_frame(run))
        if run.fallen and i % stride == 0 and len(frames) > fps:
            # Hold the fall on screen for a short moment, then stop.
            for _ in range(fps):
                frames.append(frames[-1])
            break
    write_video(path, frames, fps)


def command_from_key(key: int) -> str | None:
    if key in (KEY_1, KEY_KP_1):
        return "step"
    if key in (KEY_2, KEY_KP_2):
        return "tilt"
    if key in (KEY_3, KEY_KP_3):
        return "mu"
    if key == KEY_SPACE:
        return "pause"
    if key == KEY_R:
        return "reset"
    if key == KEY_LEFT_BRACKET:
        return "mu-down"
    if key == KEY_RIGHT_BRACKET:
        return "mu-up"
    return None


def kinds_from_group_change(seen: list[int], geomgroup) -> list[str]:
    """Controller laws implied by MuJoCo toggling geom groups 1, 2, or 3."""
    fired = []
    for index, kind in GROUP_KIND.items():
        now = int(geomgroup[index])
        if now != int(seen[index]):
            seen[index] = now
            fired.append(kind)
    return fired


def _neighbor_mu(mu: float, step: int) -> float:
    ladder = list(MU_LADDER)
    if all(abs(mu - item) > 1e-9 for item in ladder):
        ladder.append(mu)
        ladder.sort()
    index = min(range(len(ladder)), key=lambda i: abs(ladder[i] - mu))
    return ladder[int(np.clip(index + step, 0, len(ladder) - 1))]


def serve(run: LiveRun) -> None:
    import mujoco.viewer

    paused = {"on": False}
    pending: list[int] = []

    def on_key(keycode: int) -> None:
        pending.append(int(keycode))

    viewer = mujoco.viewer.launch_passive(run.robot.model, run.robot.data, key_callback=on_key)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(run.robot.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    viewer.cam.distance = 1.85
    viewer.cam.azimuth = 125.0
    viewer.cam.elevation = -16.0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    print("MuJoCo window is open.")
    print("1 step LQR    2 tilt-limited    3 friction-aware")
    print("[ ] changes μ    space pauses    r resets")
    group_seen = [int(viewer.opt.geomgroup[i]) for i in range(6)]

    def announce() -> None:
        print(f"{LABELS[run.kind]}    μ = {run.mu:.2f}")

    next_tick = time.perf_counter()
    while viewer.is_running():
        acted = None
        while pending:
            command = command_from_key(pending.pop(0))
            if command in ("step", "tilt", "mu"):
                run.retarget(kind=command)
                acted = command
                announce()
            elif command == "pause":
                paused["on"] = not paused["on"]
            elif command == "reset":
                run.reset()
                announce()
            elif command == "mu-down":
                run.retarget(mu=_neighbor_mu(run.mu, -1))
                announce()
            elif command == "mu-up":
                run.retarget(mu=_neighbor_mu(run.mu, +1))
                announce()
        for kind in kinds_from_group_change(group_seen, viewer.opt.geomgroup):
            if kind == acted:
                continue
            run.retarget(kind=kind)
            acted = kind
            announce()
        with viewer.lock():
            if not paused["on"]:
                run.step()
            if viewer.user_scn is not None:
                paint_trail(viewer.user_scn, run.trail, TRAIL[run.kind], clear=True)
        viewer.set_texts(run.overlay())
        viewer.sync()
        next_tick += run.dt
        delay = next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            next_tick = time.perf_counter()
    viewer.close()


def has_display() -> bool:
    if sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Watch the ballbot contact model in MuJoCo.")
    parser.add_argument("--mu", type=float, default=0.15, help="Floor sliding friction. Default 0.15.")
    parser.add_argument("--controller", choices=KINDS, default="mu", help="Outer law. Default is friction-aware.")
    parser.add_argument("--goal", type=float, default=1.5, help="Goal along x, in meters. Default 1.5.")
    parser.add_argument("--seconds", type=float, default=8.0, help="Length of a --record clip.")
    parser.add_argument("--record", default="", help="Write an mp4 here instead of opening a window.")
    args = parser.parse_args(argv)

    run = LiveRun(mu=args.mu, kind=args.controller, goal=args.goal)
    if args.record:
        record(run, args.record, seconds=args.seconds)
        print(f"wrote {args.record}")
        return
    if not has_display():
        print("This machine has no display, so the MuJoCo window cannot open.")
        print("Record a clip instead:")
        print("  python3 -m experiments.view_mujoco --record ../results/mujoco_view.mp4")
        raise SystemExit(1)
    try:
        serve(run)
    except RuntimeError as exc:
        message = str(exc)
        if "mjpython" in message:
            print(message)
            print("On macOS, run:  mjpython -m experiments.view_mujoco")
            raise SystemExit(1) from exc
        raise


if __name__ == "__main__":
    main()
