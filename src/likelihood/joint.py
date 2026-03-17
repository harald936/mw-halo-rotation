"""
Joint likelihood: rotation curve + GD-1 stream.

ln L_joint = ln L_RC + ln L_stream

These are independent datasets (different tracers), so the
joint likelihood is the product (sum of log-likelihoods).
"""

from .rotation_curve import ln_likelihood_rc
from .stream import ln_likelihood_stream


def ln_likelihood_joint(pot):
    """
    Joint log-likelihood of RC + stream data given a potential.

    Parameters
    ----------
    pot : list of galpy Potential objects

    Returns
    -------
    ln_L : float
    """
    lnL_rc = ln_likelihood_rc(pot)
    lnL_stream = ln_likelihood_stream(pot)
    return lnL_rc + lnL_stream
