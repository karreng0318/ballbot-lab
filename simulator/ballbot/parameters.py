"""Physical parameters.

The default set is the CMU ballbot, from Table I of Nagarajan, Kantor, and
Hollis, "The Ballbot: An Omnidirectional Balancing Mobile Robot" (IJRR).
Pitch inertia is their measured Iyy about the body center of mass.

Those authors also report a drivetrain Coulomb torque and a viscous
coefficient. Those act at the motors, not at the floor, so they are not
folded into mu. Leaving them out keeps floor friction as the only contact
parameter this study varies.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    m: float = 51.66  # body mass, kg
    mb: float = 2.44  # ball mass, kg
    Ib: float = 0.0174  # ball inertia about its center, kg m^2
    I: float = 12.48  # body pitch inertia about the COM, kg m^2
    l: float = 0.69  # COM height above the ball center, m
    r: float = 0.1058  # ball radius, m
    g: float = 9.81
    tau_max: float = 80.0  # research torque limit, N m; not a measured amplifier limit

    @property
    def mass(self) -> float:
        return self.m + self.mb


CMU = Params()
