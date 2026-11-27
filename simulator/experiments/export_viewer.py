"""Write viewer/traj.js from the planar study, subsampled for playback."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.run_study import encoder_feedback, make_controller
from ballbot.control import BalanceLQR
from ballbot.parameters import CMU
from ballbot.planar import encoder_position, simulate

OUT = Path(__file__).resolve().parents[2] / "viewer" / "traj.js"
STRIDE = 25  # 0.002 s * 25 = 50 ms
MUS = (0.15, 1.0)
KINDS = ("step", "tilt", "mu")


def pack(mu: float, kind: str, law: BalanceLQR) -> dict:
    ctrl = encoder_feedback(make_controller(kind, mu, law))
    tr = simulate(mu, ctrl, 14.0, 0.002)
    sl = slice(None, None, STRIDE)
    enc = encoder_position(tr, CMU)
    fall = None
    if tr.fallen:
        fall = int(round(tr.t_fall / 0.002)) // STRIDE
    return {
        "t": np.round(tr.t[sl], 3).tolist(),
        "x": np.round(tr.x[sl], 4).tolist(),
        "theta": np.round(tr.theta[sl], 5).tolist(),
        "enc": np.round(enc[sl], 4).tolist(),
        "fall": fall,
    }


def main() -> None:
    law = BalanceLQR(CMU)
    runs = {f"{mu:.2f}": {kind: pack(mu, kind, law) for kind in KINDS} for mu in MUS}
    payload = {"goal": 2.0, "dt": 0.05, "runs": runs}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.TRAJ = " + json.dumps(payload) + ";\n")
    print("wrote", OUT, "bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
