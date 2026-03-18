"""
Triaxial NFW dark matter halo with figure rotation.

The halo has four free parameters:
    v_h     : velocity scale (km/s) — sets the halo mass
    r_h     : scale radius (kpc)
    q_z     : vertical axis ratio (z/x) — <1 oblate, >1 prolate
    Omega_p : figure rotation rate (km/s/kpc) — tumbling around z-axis

And one fixed shape parameter:
    Q_Y     : in-plane axis ratio (y/x) — fixed at 0.9

The halo is TRIAXIAL: three unequal axes (x != y != z).
This is essential because rotating an axisymmetric shape (b=1)
around its symmetry axis produces no observable effect.
With b=0.9, the in-plane shape is elliptical, so tumbling
around z changes the gravitational field over time.

Q_Y=0.9 is consistent with cosmological simulations
(Vera-Ciro & Helmi 2013, Chua+2019) and with constraints
from the Sagittarius stream (Law & Majewski 2010).

At Omega_p = 0, this reduces to a static triaxial NFW halo.

All galpy potentials use pure natural units (no ro/vo in
constructors) to avoid unit-mode confusion.
"""

import numpy as np
from galpy.potential import (
    NFWPotential,
    TriaxialNFWPotential,
    SolidBodyRotationWrapperPotential,
    vcirc,
)

# Solar parameters (must match baryons.py and the rest of the pipeline)
RO = 8.122   # kpc
VO = 229.0   # km/s

# Fixed in-plane axis ratio
Q_Y = 0.9    # y/x ratio — makes the halo triaxial

# Halo tilt angle from Nibauer & Bonaca 2025
# The halo minor axis is tilted 18 deg from the disk z-axis toward x
TILT_DEG = 18.0
_tilt_rad = np.radians(TILT_DEG)
ZVEC = [np.sin(_tilt_rad), 0.0, np.cos(_tilt_rad)]  # [0.309, 0, 0.951]


def _nfw_amp_from_vh(v_h, r_h):
    """
    Compute galpy NFW amp parameter from physical v_h and r_h.

    v_h is defined as v_circ(r_h) of the spherical-equivalent NFW.
    Computed numerically using galpy's own vcirc to guarantee
    exact consistency with galpy's internal conventions.
    """
    a_nat = r_h / RO
    nfw_unit = NFWPotential(amp=1.0, a=a_nat)
    vc_sq_unit = vcirc(nfw_unit, a_nat) ** 2
    return (v_h / VO) ** 2 / vc_sq_unit


def build_halo_potential(v_h, r_h, q_z, Omega_p=0.0, pa=0.0):
    """
    Build a triaxial, possibly rotating, NFW halo.

    Parameters
    ----------
    v_h : float
        Halo circular velocity scale in km/s. Defined as v_circ(r_h)
        for the spherical-equivalent NFW.
    r_h : float
        Halo scale radius in kpc.
    q_z : float
        Vertical (z) axis ratio. q_z < 1 = oblate, q_z > 1 = prolate.
    Omega_p : float, optional
        Figure rotation rate in km/s/kpc. Default 0 (static).
    pa : float, optional
        Initial position angle of halo major axis in radians. Default 0.

    Returns
    -------
    pot : galpy Potential instance
        Triaxial halo, wrapped in SolidBodyRotation if Omega_p != 0.
    """
    a_nat = r_h / RO
    amp = _nfw_amp_from_vh(v_h, r_h)

    halo = TriaxialNFWPotential(
        amp=amp,
        a=a_nat,
        b=Q_Y,       # in-plane axis ratio — TRIAXIAL
        c=q_z,       # vertical axis ratio
        zvec=ZVEC,   # halo minor axis tilted 18 deg (Nibauer & Bonaca 2025)
    )

    if Omega_p == 0.0:
        return halo

    # Convert Omega_p from km/s/kpc to galpy natural frequency (vo/ro)
    omega_nat = Omega_p / (VO / RO)

    wrapped = SolidBodyRotationWrapperPotential(
        pot=halo,
        omega=omega_nat,
        pa=pa,
    )

    return wrapped
