"""
Joint likelihood: RC + GD-1 + Pal 5 + Jhelum + Orphan-Chenab.

ln L = ln L_RC + ln L_GD1 + ln L_Pal5 + ln L_Jhelum + ln L_OC

Five independent datasets probing the halo from different
directions and distances. ~315 degrees of stream coverage.
"""

from .rotation_curve import ln_likelihood_rc
from .stream import ln_likelihood_stream
from .stream_pal5 import ln_likelihood_pal5
from .stream_jhelum import ln_likelihood_jhelum
from .stream_orphan import ln_likelihood_orphan


def ln_likelihood_joint(pot):
    """
    Joint log-likelihood of all datasets given a potential.
    """
    return (ln_likelihood_rc(pot)
            + ln_likelihood_stream(pot)
            + ln_likelihood_pal5(pot)
            + ln_likelihood_jhelum(pot)
            + ln_likelihood_orphan(pot))
