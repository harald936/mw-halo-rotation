"""
Composite Milky Way potential: fixed baryons + free halo + optional LMC.

The main entry point is build_potential(v_h, r_h, q_z, Omega_p),
which returns a full MW potential as a list of galpy Potential objects.
"""

from .baryons import build_baryonic_potential
from .halo import build_halo_potential

# Solar parameters
RO = 8.122  # kpc
VO = 229.0  # km/s

# Cache for LMC potential (expensive to compute, doesn't change between MCMC steps)
_LMC_CACHE = {}


def build_potential(v_h, r_h, q_z, Omega_p=0.0, pa=0.0, include_lmc=False):
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
    include_lmc : bool, optional
        If True, include the LMC as a time-dependent perturbation.
        The LMC orbit is computed once and cached. Default False.

    Returns
    -------
    pot : list of galpy Potential objects
        [disk, bulge, halo] or [disk, bulge, halo, lmc]
    """
    baryons = build_baryonic_potential()
    halo = build_halo_potential(v_h, r_h, q_z, Omega_p, pa)

    pot = baryons + [halo]

    if include_lmc:
        if 'lmc' not in _LMC_CACHE:
            from .lmc import build_lmc_potential
            # Build LMC using the current MW potential (without LMC)
            lmc_pot, lmc_orbit = build_lmc_potential(pot)
            _LMC_CACHE['lmc'] = lmc_pot
            _LMC_CACHE['orbit'] = lmc_orbit
        pot = pot + [_LMC_CACHE['lmc']]

    return pot
