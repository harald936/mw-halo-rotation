"""
Composite Milky Way potential: fixed baryons + free halo.

The main entry point is build_potential(v_h, r_h, q_z, Omega_p),
which returns a full MW potential as a list of galpy Potential objects.
"""

from .baryons import build_baryonic_potential
from .halo import build_halo_potential

# Solar parameters
RO = 8.122  # kpc
VO = 229.0  # km/s


def build_potential(v_h, r_h, q_z, Omega_p=0.0, pa=0.0):
    """
    Build complete MW potential with fixed baryons and free halo.

    Parameters
    ----------
    v_h : float
        Halo velocity scale (km/s).
    r_h : float
        Halo scale radius (kpc).
    q_z : float
        Halo vertical flattening.
    Omega_p : float, optional
        Halo figure rotation rate (km/s/kpc). Default 0.
    pa : float, optional
        Halo position angle at t=0 (radians). Default 0.

    Returns
    -------
    pot : list of galpy Potential objects
        [disk, bulge, halo] — can be passed to galpy orbit integration,
        vcirc computation, etc.
    """
    baryons = build_baryonic_potential()
    halo = build_halo_potential(v_h, r_h, q_z, Omega_p, pa)

    return baryons + [halo]
