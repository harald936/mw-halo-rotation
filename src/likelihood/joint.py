"""
Joint likelihood: rotation curve + GD-1 stream + Pal 5 stream.

ln L_joint = ln L_RC + ln L_GD1 + ln L_Pal5

These are independent datasets (different tracers at different
locations in the halo), so the joint likelihood is the product
(sum of log-likelihoods).
"""

from .rotation_curve import ln_likelihood_rc
from .stream import ln_likelihood_stream
from .stream_pal5 import ln_likelihood_pal5


def ln_likelihood_joint(pot):
    """
    Joint log-likelihood of RC + GD-1 + Pal 5 data given a potential.

    Parameters
    ----------
    pot : list of galpy Potential objects

    Returns
    -------
    ln_L : float
    """
    lnL_rc = ln_likelihood_rc(pot)
    lnL_gd1 = ln_likelihood_stream(pot)
    lnL_pal5 = ln_likelihood_pal5(pot)
    return lnL_rc + lnL_gd1 + lnL_pal5
