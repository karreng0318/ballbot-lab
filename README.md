# Ballbots: friction, balance, and where a paper could sit

Yes. This is a good collaboration, and the first project should be narrow.

A ballbot balances on one sphere. To accelerate, it leans. Every classical controller, from the CMU LQR work through the 2026 tilt-limited planners, assumes the ball rolls without slipping, so a roller encoder is a position sensor. Floor friction decides whether that assumption is true. The comparison in `results/` is the project: same balance law, three ways of commanding a 2 m move, and a floor whose friction coefficient μ is the only thing that changes.

On the CMU-sized machine in the planar model, with the controller told only what the rollers measure:

| Floor | What happens |
| --- | --- |
| μ = 1.0 | Step LQR, a tilt-limited plan, and a friction-aware plan all arrive. |
| μ = 0.8 | The tilt-limited plan arrives. The raw step still falls. |
| μ from 0.08 to 0.5 | Only the friction-aware plan arrives, and it does so without slipping, so the encoder still reads the truth. The other two slip, the encoder runs away by tens of meters, and they fall. |
| μ = 0.06 | All three fall. The friction-aware law refuses to slip and still cannot produce the moment it needs. That floor is a place to put the legs down, not a place to drive. |

That is the "different μ" idea, as one law rather than a pile of controllers. The acceleration it will ask for is the smaller of two numbers: the steady acceleration of a 6° lean, and `0.65 μ g`. The motor torque is then clipped so the shear force required to keep rolling stays inside μ times the normal force. Below the μ where even that clip cannot balance, the maneuver is declined.

The memo for your colleague is `ideas/PROJECT_IDEAS.md`. The reading order is `ideas/LITERATURE.md`. The numbers and figures are `results/STUDY.md`.

## What is in this folder

```
papers/       arXiv preprints and a bibliography. IEEE papers are linked, not copied.
simulator/    planar stick-slip model, MuJoCo contact model, and the comparison.
results/      figures from the last run of the study.
ideas/        the proposed paper, and the papers that bound it.
```

This project lives here because `/home/swekar/Documents` is not on this machine. Copy or clone it to `~/Documents/Collab_Research/ballbots` when you want it next to your other notes.

## Run the comparison

```bash
cd simulator
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests
python3 -m experiments.run_study
```

The study rewrites `results/`. The planar tests cover the dynamics, the LQR, and the headline above (μ = 0.15 versus μ = 1).

## Watch it in MuJoCo

```bash
cd simulator
python3 -m experiments.view_mujoco
```

That opens MuJoCo's own window on the sphere-on-plane model. It starts as the friction-aware law on μ = 0.15, driving toward the blue mark 1.5 m ahead. Keys: `1` step LQR, `2` tilt-limited, `3` friction-aware, `[` and `]` change μ, space pauses, `r` resets. The marks on the ball show spin. Contact points and forces are drawn. The housings around the ball are visual only; the motor is still a torque at the ball center.

On macOS, use `mjpython -m experiments.view_mujoco` instead of `python3`. With no display, write a clip:

```bash
python3 -m experiments.view_mujoco --record ../results/mujoco_view.mp4
```

## What the simulator is, and is not

The planar model is a body hinged to a ball, with a Coulomb floor. Parameters are the CMU ballbot from Nagarajan, Kantor, and Hollis, IJRR 2014, Table I: ball radius 0.1058 m, ball mass 2.44 kg, body mass 51.66 kg, COM 0.69 m above the ball center. Their Coulomb and viscous numbers are drivetrain friction, so they are left out. Floor μ is the parameter under test.

In steady rolling with no vertical acceleration, the friction coefficient the floor must supply is exactly `|a| / g`. The lean that produces `a` depends on the robot. A lean limit and a friction limit are different constraints, and the torque spike that acquires the lean is larger than either steady value.

MuJoCo uses the same masses and a sphere on a plane, with sliding, torsional, and rolling friction on the contact pair. It is the right check on the contact. It does not yet reproduce the planar success at low μ: at μ = 0.12 the friction-aware law travels to about 1.85 m of a 1.5 m goal and then the lean diverges, because the torque clip is computed from the rigid model while the compliant contact has already started to slide. At μ = 1 all three laws arrive. That gap is the next modeling task, not a result to paper over.

Wheel-on-ball friction, the other interface, is not in this model. asRoBallet's MuJoCo repository models every roller and is the code to use when that interface is the question. It is MIT licensed and linked from the literature note. OpenBallBot-RL needs a patched MuJoCo build; skip it until the patch is upstream.

## A boundary on the claim

The friction-aware law is handed the true μ. A noise-free check shows that once slip has started, `|f| / N` recovered from torque and ball acceleration matches μ. Detecting slip from an IMU and an encoder, with quantization and delay, is still open. There is no hardware in this folder.
