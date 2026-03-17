"""
Triaxial NFW dark matter halo with optional figure rotation.

The halo has three free parameters controlling its shape and mass:
    v_h  : velocity scale (km/s) — sets the halo mass normalization
           defined as the halo's circular velocity at its scale radius
    r_h  : scale radius (kpc) — where the density profile transitions
    q_z  : vertical axis ratio (z/x) — <1 oblate, >1 prolate

And one free parameter controlling tumbling:
    Omega_p : figure rotation rate (km/s/kpc) — pattern speed around z-axis

The in-plane axis ratio is fixed: q_y = 1.0 (axisymmetric in x-y).
This means the non-sphericity comes only from vertical flattening.
A future extension could free q_y for full triaxiality.

At Omega_p = 0, this reduces to a static flattened NFW halo.

IMPORTANT: All galpy potentials are constructed WITHOUT ro/vo kwargs,
so they operate in pure natural units (distances in units of RO,
velocities in units of VO). This avoids unit-mode confusion.
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


def _nfw_amp_from_vh(v_h, r_h):
    """
    Compute galpy NFW amp parameter from physical v_h and r_h.

    v_h is defined as v_circ(r_h) of the spherical NFW.
    We determine the amp numerically using galpy's own vcirc,
    ensuring exact consistency with galpy's internal conventions.

    Parameters
    ----------
    v_h : float
        Desired v_circ at r_h, in km/s.
    r_h : float
        Scale radius in kpc.

    Returns
    -------
    amp : float
        galpy amp parameter in natural units.
    """
    a_nat = r_h / RO
    # vcirc^2 scales linearly with amp, so compute the geometry factor
    # by evaluating vcirc for amp=1
    nfw_unit = NFWPotential(amp=1.0, a=a_nat)
    vc_sq_unit = vcirc(nfw_unit, a_nat) ** 2  # vcirc^2(r_h) for amp=1
    # amp = (v_h / VO)^2 / vc_sq_unit
    return (v_h / VO) ** 2 / vc_sq_unit


def build_halo_potential(v_h, r_h, q_z, Omega_p=0.0, pa=0.0):
    """
    Build a (possibly rotating) flattened NFW halo.

    Parameters
    ----------
    v_h : float
        Halo circular velocity scale in km/s. Defined as the circular
        velocity at the scale radius for the spherical-equivalent NFW.
    r_h : float
        Halo scale radius in kpc.
    q_z : float
        Vertical (z) axis ratio. q_z < 1 = oblate, q_z > 1 = prolate.
    Omega_p : float, optional
        Figure rotation rate in km/s/kpc. Default 0 (static).
    pa : float, optional
        Initial position angle of the halo major axis in radians.
        Default 0.

    Returns
    -------
    pot : galpy Potential instance
        The halo potential (wrapped in rotation if Omega_p != 0).
        In pure natural units (no ro/vo attached).
    """
    a_nat = r_h / RO
    amp = _nfw_amp_from_vh(v_h, r_h)

    halo = TriaxialNFWPotential(
        amp=amp,
        a=a_nat,
        b=1.0,       # q_y = 1 (axisymmetric in-plane)
        c=q_z,       # q_z = free vertical flattening
    )

    if Omega_p == 0.0:
        return halo

    # Convert Omega_p from km/s/kpc to galpy natural frequency units (vo/ro)
    omega_nat = Omega_p / (VO / RO)

    wrapped = SolidBodyRotationWrapperPotential(
        pot=halo,
        omega=omega_nat,
        pa=pa,
    )

    return wrapped
