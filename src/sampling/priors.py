"""
Parameter priors for MCMC sampling.

Four free parameters with uniform priors:
    v_h     : U(100, 300) km/s — halo velocity scale
    r_h     : U(5, 40) kpc — halo scale radius
    q_z     : U(0.5, 1.5) — vertical flattening
    Omega_p : U(0, 0.5) km/s/kpc — figure rotation rate

The prior bounds are broad enough to cover all reasonable
MW halo models while excluding unphysical regions.

Simulation prediction: Omega_p ~ 0.10 km/s/kpc
(Bailin & Steinmetz 2004, scaled to h=0.7)
"""

import numpy as np

# Prior bounds: (min, max) for each parameter
PARAM_NAMES = ["v_h", "r_h", "q_z", "Omega_p"]
PARAM_BOUNDS = {
    "v_h": (100.0, 300.0),      # km/s
    "r_h": (5.0, 40.0),         # kpc
    "q_z": (0.5, 2.0),          # dimensionless — widened to avoid boundary effects
    "Omega_p": (0.0, 0.5),      # km/s/kpc
}


def ln_prior(theta):
    """
    Log-prior for the parameter vector.

    Parameters
    ----------
    theta : array-like
        [v_h, r_h, q_z, Omega_p]

    Returns
    -------
    ln_pi : float
        0.0 if within bounds, -inf if outside.
    """
    v_h, r_h, q_z, Omega_p = theta

    if not (PARAM_BOUNDS["v_h"][0] < v_h < PARAM_BOUNDS["v_h"][1]):
        return -np.inf
    if not (PARAM_BOUNDS["r_h"][0] < r_h < PARAM_BOUNDS["r_h"][1]):
        return -np.inf
    if not (PARAM_BOUNDS["q_z"][0] < q_z < PARAM_BOUNDS["q_z"][1]):
        return -np.inf
    if not (PARAM_BOUNDS["Omega_p"][0] <= Omega_p < PARAM_BOUNDS["Omega_p"][1]):
        return -np.inf

    return 0.0
