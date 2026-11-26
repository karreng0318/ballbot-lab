"""MuJoCo ballbot with a named ball-floor friction pair.

The mechanism matches the planar model: a torque between the body and the
ball, and a Coulomb contact with the floor. World lean comes from the body
orientation. The joint angle is not the lean, because the ball rolls under
the body.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ballbot.control import BalanceLQR
from ballbot.parameters import CMU, Params

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ballbot_3d.xml"


def rpy_from_quat(quat: np.ndarray) -> np.ndarray:
    """Roll, pitch, yaw from a MuJoCo wxyz quaternion."""
    w, x, y, z = [float(v) for v in quat]
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(sin_yaw, cos_yaw)
    return np.array([roll, pitch, yaw])


class MujocoBallbot:
    def __init__(self, mu: float = 0.8, p: Params = CMU):
        self.p = p
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.pair = self._pair_id()
        self.pitch_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch")
        self.roll_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "roll")
        self.yaw_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw")
        self._sensor = {
            name: (
                int(self.model.sensor_adr[i]),
                int(self.model.sensor_dim[i]),
            )
            for i, name in enumerate(
                ["ball_pos", "ball_vel", "ball_angvel", "body_quat", "body_angvel"]
            )
        }
        # Positive planar torque spins the ball toward +x and decreases pitch.
        # MuJoCo's ball-joint motor does the opposite on the child body, so the
        # pitch command is negated. Checked by tests/test_mujoco.py.
        self.pitch_sign = -1.0
        self.roll_sign = -1.0
        self.set_mu(mu)
        self.reset()

    def _pair_id(self) -> int:
        ball = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for i in range(self.model.npair):
            geoms = {int(self.model.pair_geom1[i]), int(self.model.pair_geom2[i])}
            if geoms == {ball, floor}:
                return i
        raise RuntimeError("ball_floor pair not found")

    def set_mu(self, mu: float) -> None:
        self.mu = float(mu)
        self.model.pair_friction[self.pair, 0] = self.mu
        self.model.pair_friction[self.pair, 1] = self.mu

    def _read(self, name: str) -> np.ndarray:
        adr, dim = self._sensor[name]
        return self.data.sensordata[adr : adr + dim].copy()

    def reset(self, pitch: float = 0.0, roll: float = 0.0) -> None:
        mujoco.mj_resetData(self.model, self.data)
        # freejoint (7) then ball joint quaternion (4)
        self.data.qpos[2] = self.p.r
        self.data.qpos[3] = 1.0
        half_roll = 0.5 * roll
        half_pitch = 0.5 * pitch
        # Composition of a small roll then pitch, to first order.
        self.data.qpos[7:11] = np.array(
            [
                1.0,
                half_roll,
                half_pitch,
                0.0,
            ]
        )
        self.data.qpos[7:11] /= np.linalg.norm(self.data.qpos[7:11])
        mujoco.mj_forward(self.model, self.data)

    def state(self) -> dict[str, float]:
        pos = self._read("ball_pos")
        vel = self._read("ball_vel")
        ball_w = self._read("ball_angvel")
        quat = self._read("body_quat")
        body_w = self._read("body_angvel")
        roll, pitch, yaw = rpy_from_quat(quat)
        return {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "vx": float(vel[0]),
            "vy": float(vel[1]),
            "wx": float(ball_w[0]),
            "wy": float(ball_w[1]),
            "wz": float(ball_w[2]),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll_rate": float(body_w[0]),
            "pitch_rate": float(body_w[1]),
            "yaw_rate": float(body_w[2]),
        }

    def planar_state(self) -> np.ndarray:
        """Pack world pitch motion into the planar [x, psi, theta, dx, dpsi, dtheta] order.

        psi is integrated from the ball's world spin about +y, starting at 0
        when reset() is called. It is not wrapped.
        """
        s = self.state()
        psi = getattr(self, "_psi", 0.0)
        return np.array([s["x"], psi, s["pitch"], s["vx"], s["wy"], s["pitch_rate"]])

    def contact_friction_ratio(self) -> float:
        if self.data.ncon == 0:
            return float("nan")
        tangent = 0.0
        normal = 0.0
        for i in range(self.data.ncon):
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            normal += force[0]
            tangent += float(np.hypot(force[1], force[2]))
        if normal <= 1.0:
            return float("nan")
        return tangent / normal

    def step(self, tau_pitch: float, tau_roll: float = 0.0, tau_yaw: float = 0.0) -> None:
        dt = float(self.model.opt.timestep)
        self.data.ctrl[self.pitch_act] = self.pitch_sign * tau_pitch
        self.data.ctrl[self.roll_act] = self.roll_sign * tau_roll
        self.data.ctrl[self.yaw_act] = tau_yaw
        # Integrate ball spin about world +y from the pre-step rate. One step
        # of lag is negligible next to contact noise, and it avoids touching
        # the quaternion of a rolling sphere.
        wy = float(self.state()["wy"])
        mujoco.mj_step(self.model, self.data)
        self._psi = getattr(self, "_psi", 0.0) + wy * dt

    def apply_balance(
        self,
        law: BalanceLQR,
        x_ref: float = 0.0,
        vx_ref: float = 0.0,
        pitch_ref: float = 0.0,
        y_ref: float = 0.0,
    ) -> None:
        s = self.state()
        pitch_state = np.array([s["x"], self._psi if hasattr(self, "_psi") else 0.0, s["pitch"], s["vx"], s["wy"], s["pitch_rate"]])
        roll_state = np.array([s["y"], 0.0, s["roll"], s["vy"], 0.0, s["roll_rate"]])
        tau_pitch = law.torque(pitch_state, x_ref=x_ref, v_ref=vx_ref, theta_ref=pitch_ref)
        tau_roll = law.torque(roll_state, x_ref=y_ref, v_ref=0.0, theta_ref=0.0)
        yaw_damp = -8.0 * s["yaw_rate"] - 4.0 * s["yaw"]
        self.step(tau_pitch, tau_roll, yaw_damp)
