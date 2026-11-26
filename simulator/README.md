# Simulator

Planar stick-slip dynamics and a MuJoCo sphere-on-plane model of a CMU-scale ballbot. Both are driven by the same LQR. The study in `experiments/run_study.py` compares three outer laws while the controller's position feedback is roller odometry.

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests
python3 -m experiments.run_study
```

Run those from this directory. Figures go to `../results/`.

## Model

`ballbot/planar.py` is the model to argue with. State is ball position `x`, ball spin `psi`, and lean `theta`. Positive `psi` rolls the ball toward `+x`. Positive `theta` moves the center of mass toward `+x`. Motor torque `tau` is applied to the ball about `+y`; the body feels the opposite.

Slip velocity is `dx - r dpsi`. Inside the cone the contact sticks and that velocity is held at zero. Outside it, friction is `-μ N` times the slip direction. In steady rolling, `μ = |a| / g`.

`ballbot/control.py` has the LQR and the three outer laws used in the study: a step to the goal, a tilt-limited reference, and the same reference further capped by `0.65 μ g` with the torque projected into the cone.

`models/ballbot_3d.xml` is the MuJoCo check. The named contact pair `ball_floor` is where μ is written at runtime. Torsional and rolling friction are present and small, so the sweep is about sliding μ. The body is hinged to the ball at its center. Rollers are not modeled.

## What is left out on purpose

Drivetrain Coulomb and viscous friction from the CMU table. Wheel-on-ball friction. A patched anisotropic-friction MuJoCo. Those are real, and they are separate questions. See `../ideas/PROJECT_IDEAS.md`.
