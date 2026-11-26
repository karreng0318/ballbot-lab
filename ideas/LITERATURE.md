# What to read, and why

The field is small enough to actually read. This is the order that makes the proposed paper obvious. Full bib entries are in `papers/bibliography.bib`. Open PDFs are in `papers/pdfs/`. IEEE and Sage papers are linked rather than copied.

## The machine and the no-slip model

1. T. B. Lauwers, G. A. Kantor, and R. L. Hollis, "A Dynamically Stable Single-Wheeled Mobile Robot with Inverse Mouse-Ball Drive," ICRA 2006.
   The original ballbot. Inverse mouse-ball drive, human scale, and the reason a tall robot on a sphere is interesting in a crowded room.
   Author PDF: http://www.msl.ri.cmu.edu/publications/pdfs/ballbot_ICRA06_web.pdf

2. U. Nagarajan, A. Mampetta, G. A. Kantor, and R. L. Hollis, "State Transition, Balancing, Station Keeping, and Yaw Control for a Dynamically Stable Single Spherical Wheel Mobile Robot," ICRA 2009.
   The parameter table we use starts here and is repeated, with the same numbers, in the IJRR paper.
   Author PDF: http://www.msl.ri.cmu.edu/publications/pdfs/BallbotTransition_ICRA._2009.pdf

3. U. Nagarajan, G. Kantor, and R. Hollis, "The Ballbot: An Omnidirectional Balancing Mobile Robot," *International Journal of Robotics Research*, 33(6):917–930, 2014. doi:10.1177/0278364913509126.
   The paper to know. Hardware, LQR architecture, and rest-to-rest motions planned as body leans. Table I is the simulator's parameter source. The friction numbers in that table are drivetrain Coulomb and viscous torque, not floor μ.
   Author PDF: http://www.msl.ri.cmu.edu/publications/pdfs/ballbotIJRR_final.pdf

4. U. Nagarajan and R. Hollis, "Shape Space Planner for Shape-Accelerated Balancing Mobile Robots," *IJRR*, 32(11):1323–1341, 2013. doi:10.1177/0278364913495424.
   The planning paper. A lean trajectory is chosen so that tracking it produces the desired base acceleration, through the no-slip dynamic constraint. Our proposal is what happens when that constraint saturates.
   Author PDF: http://www.msl.ri.cmu.edu/publications/pdfs/ballbotShapePlannerIJRR_final.pdf

5. M. Kumagai and T. Ochiai, "Development of a Robot Balanced on a Ball," ICCAS 2008, and the JRM 2010 first report.
   BallIP. Three omniwheels, and they command ball acceleration rather than torque, which makes the balance loop less sensitive to inertia. doi:10.1109/ICCAS.2008.4694680.

6. P. Fankhauser and C. Gwerder, "Modeling and Control of a Ballbot," ETH Zurich, 2010. doi:10.3929/ethz-a-010056685.
   The thesis everyone copies for the three-omniwheel planar model. Rezero.

## Planning and whole-body control, still no-slip

7. M. V. Minniti, F. Farshidian, R. Grandia, and M. Hutter, "Whole-Body MPC for a Dynamically Stable Mobile Manipulator," 2019. arXiv:1902.10415. PDF in `papers/pdfs/`.
   Rezero plus an arm, one optimal-control problem for balance and manipulation.

8. C. Xiao, M. Mansouri, D. Lam, J. Ramos, and E. T. Hsiao-Wecksler, "Design and Control of a Ballbot Drivetrain with High Agility, Minimal Footprint, and High Payload," 2023. arXiv:2304.02887. PDF in `papers/pdfs/`.
   A current hardware design (MiaPURE). Useful if the conversation turns to building a machine. Their stiction discussion is about the transmission, not the floor.

9. D. C. Vu and T. L. Nguyen, "Safe-Ballbot: Tilt-Consistent MPC with Control Barrier Functions for Ballbot Navigation," arXiv:2611.03307, November 2026. PDF in `papers/pdfs/`. Project page: https://safe-ballbot.github.io/
   Read the acceleration bound. It is derived from the largest steady lean the tracker will accept. Their MuJoCo model is adapted from asRoBallet, with roller contacts. The same Hanoi group, with Pham, Vu, Nguyen, and coauthors, published the tilt-limited CBF-MPC predecessor in IEEE Access 2025, doi:10.1109/ACCESS.2025.3567474. A new obstacle-avoidance paper would be competing with this line.

## Friction, stated as a problem

10. A. Salehi, "Reinforcement Learning for Ballbot Navigation in Uneven Terrain," arXiv:2505.18417, 2025. PDF in `papers/pdfs/`. Code: https://github.com/salehiac/OpenBallBot-RL (Apache-2.0).
    Says that control-theoretic navigation usually assumes no slip, and trains RL on uneven ground. The omniwheels use anisotropic friction that needs a MuJoCo patch. Good citation, awkward code to build on.

11. F. Wan, G. Huang, T. Wu, Z. Zhang, B. Huang, H. Sun, M. Chen, and C. Song, "asRoBallet: Closing the Sim2Real Gap via Friction-Aware Reinforcement Learning for Underactuated Spherical Dynamics," arXiv:2604.24916, RSS 2026. PDF in `papers/pdfs/`. Code: https://github.com/asRoBallet/asRoBallet_mujoco (MIT).
    The paper nearest to this project. High-fidelity roller contact, ball-floor μ around 1 with randomization about 0.6–1.2, and a policy on hardware. Read their contact section before changing our MuJoCo pair. Use their model when the question moves from ball-floor μ to wheel-ball μ.

12. S. A. Tafrishi, M. Svinin, and K. Tahara, "A Survey on Path Planning Problem of Rolling Contacts," arXiv:2501.04442, 2025. PDF in `papers/pdfs/`.
    Where ballbots sit in the wider rolling-contact map: high friction, external omniwheel actuation, no-slip planning. Useful for the related-work section, and for seeing that low-friction rolling is studied in manipulation, not in ballbots.

## What we deliberately did not copy in

Author-hosted CMU PDFs are on the lab site for reading. They are IEEE and Sage papers, so they are not in `papers/pdfs/`. The arXiv files there are the authors' preprints; the authors keep the copyright, and this folder does not relicense them.
