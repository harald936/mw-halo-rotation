"""
Rotation curve likelihood.

Compares the model's predicted circular velocity V_circ(R) to the
Eilers+2019 rotation curve data using a Gaussian chi-squared likelihood.

The rotation curve is measured at t=0 (today), so we evaluate
the potential at the present-day halo orientation. Omega_p does
not affect V_circ at t=0 — it only matters for orbit integration.

For the triaxial (non-axisymmetric) halo, V_circ depends on azimuth.
We phi-average V_circ^2 over 36 azimuthal samples, since the observed
rotation curve is also an azimuthal average of stellar kinematics.

Asymmetric error bars are symmetrized: sigma = (sigma+ + sigma-) / 2.
"""

import numpy as np
import pandas as pd
import os
from galpy.potential import vcirc

# Solar parameters
RO = 8.122  # kpc
VO = 229.0  # km/s

# Phi grid for azimuthal averaging
# 8 samples is sufficient — the triaxial asymmetry is only ~1 km/s
_N_PHI = 8
_PHIS = np.linspace(0, 2 * np.pi, _N_PHI, endpoint=False)


def _load_rc_data(data_dir=None):
    """Load and cache the Eilers+2019 rotation curve data."""
    if data_dir is None:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        data_dir = os.path.join(repo, "data", "rotation_curve")

    rc = pd.read_csv(os.path.join(data_dir, "eilers2019_rc.csv"), comment="#")

    R_kpc = rc["R_kpc"].values
    V_obs = rc["Vcirc_kms"].values
    eV_minus = rc["eVcirc_minus_kms"].values
    eV_plus = rc["eVcirc_plus_kms"].values
    sigma = (eV_minus + eV_plus) / 2  # symmetrize

    return R_kpc, V_obs, sigma


# Load once at import time
_RC_R, _RC_V, _RC_SIGMA = _load_rc_data()


def compute_model_vcirc(pot, R_kpc):
    """
    Compute phi-averaged circular velocity for a (possibly non-axisymmetric)
    potential at given radii.

    Parameters
    ----------
    pot : list of galpy Potential objects
    R_kpc : array-like
        Radii in kpc.

    Returns
    -------
    V_model : ndarray
        Circular velocity in km/s at each radius.
    """
    R_nat = np.asarray(R_kpc) / RO
    V_model = np.empty(len(R_nat))
    for i, r in enumerate(R_nat):
        vc_sq = np.mean([vcirc(pot, r, phi=p) ** 2 for p in _PHIS])
        V_model[i] = np.sqrt(vc_sq) * VO
    return V_model


def ln_likelihood_rc(pot):
    """
    Log-likelihood of the rotation curve data given a potential.

    ln L = -0.5 * sum[ (V_obs - V_model)^2 / sigma^2 ]

    Parameters
    ----------
    pot : list of galpy Potential objects
        The composite MW potential to evaluate.

    Returns
    -------
    ln_L : float
        Log-likelihood value.
    """
    V_model = compute_model_vcirc(pot, _RC_R)
    residuals = _RC_V - V_model
    chi2 = np.sum((residuals / _RC_SIGMA) ** 2)
    return -0.5 * chi2
