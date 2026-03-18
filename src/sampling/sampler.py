"""
MCMC sampler using emcee.

Samples the joint posterior:
    ln P(theta | data) = ln L_RC + ln L_stream + ln prior

Uses emcee's ensemble sampler with StretchMove (70%) + DEMove (30%)
for better exploration of correlated/multimodal posteriors.
"""

import numpy as np
import emcee
import os

from ..potential.composite import build_potential
from ..likelihood.joint import ln_likelihood_joint
from ..likelihood.rotation_curve import ln_likelihood_rc
from ..likelihood.stream import ln_likelihood_stream
from .priors import ln_prior, PARAM_NAMES


def ln_posterior(theta):
    """
    Log-posterior: prior + likelihood.

    Parameters
    ----------
    theta : array-like
        [v_h, r_h, q_z, Omega_p]

    Returns
    -------
    ln_post : float
    """
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    v_h, r_h, q_z, Omega_p = theta

    try:
        pot = build_potential(v_h, r_h, q_z, Omega_p, include_lmc=False)
        lnL = ln_likelihood_joint(pot)
    except Exception:
        return -np.inf

    if not np.isfinite(lnL):
        return -np.inf

    return lp + lnL


def ln_posterior_rc_only(theta):
    """Log-posterior using only the rotation curve (for validation runs)."""
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    v_h, r_h, q_z, Omega_p = theta
    try:
        pot = build_potential(v_h, r_h, q_z, Omega_p)
        lnL = ln_likelihood_rc(pot)
    except Exception:
        return -np.inf

    if not np.isfinite(lnL):
        return -np.inf

    return lp + lnL


def initialize_walkers(n_walkers, p0=None, spread=None):
    """
    Initialize walker positions as a tight ball around a starting point.

    Parameters
    ----------
    n_walkers : int
    p0 : array-like, optional
        Center of the ball. Default: [160, 16, 0.93, 0.05]
    spread : array-like, optional
        Half-width of uniform ball per parameter.
        Default: [5, 2, 0.05, 0.02]

    Returns
    -------
    pos : ndarray, shape (n_walkers, 4)
    """
    if p0 is None:
        p0 = np.array([160.0, 16.0, 0.93, 0.05])
    if spread is None:
        spread = np.array([5.0, 2.0, 0.05, 0.02])

    p0 = np.asarray(p0)
    spread = np.asarray(spread)

    pos = p0 + spread * (2 * np.random.rand(n_walkers, len(p0)) - 1)
    return pos


def run_sampler(n_walkers=64, n_steps=2000, p0=None, spread=None,
                backend_file=None, rc_only=False, n_cores=None):
    """
    Run the MCMC sampler.

    Parameters
    ----------
    n_walkers : int
        Number of walkers. Must be >= 2 * n_params.
    n_steps : int
        Number of MCMC steps per walker.
    p0 : array-like, optional
        Starting point center.
    spread : array-like, optional
        Starting point spread.
    backend_file : str, optional
        Path to HDF5 file for saving chains. If None, chains are
        stored in memory only.
    rc_only : bool
        If True, use RC-only likelihood (for fast validation).
    n_cores : int, optional
        Number of CPU cores for parallel evaluation. Default: all available.

    Returns
    -------
    sampler : emcee.EnsembleSampler
    """
    import multiprocessing as mp
    from multiprocessing import Pool, cpu_count
    # macOS Python 3.8+ defaults to 'spawn' which requires pickling
    # and hangs with galpy C extensions. 'fork' works.
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass  # already set

    n_dim = 4

    if n_cores is None:
        n_cores = cpu_count()

    # Choose posterior function
    log_prob_fn = ln_posterior_rc_only if rc_only else ln_posterior

    # Moves: stretch (70%) + differential evolution (30%)
    moves = [
        (emcee.moves.StretchMove(), 0.7),
        (emcee.moves.DEMove(), 0.3),
    ]

    # Backend for saving chains
    backend = None
    if backend_file is not None:
        backend = emcee.backends.HDFBackend(backend_file)
        backend.reset(n_walkers, n_dim)

    # Initialize walkers
    pos = initialize_walkers(n_walkers, p0, spread)

    print(f"Running MCMC: {n_walkers} walkers, {n_steps} steps, "
          f"{'RC-only' if rc_only else 'joint'} likelihood, {n_cores} cores")

    if n_cores > 1:
        pool = Pool(n_cores)
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, log_prob_fn,
            moves=moves, backend=backend, pool=pool,
        )
        sampler.run_mcmc(pos, n_steps, progress=True)
        pool.close()
        pool.join()
    else:
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, log_prob_fn,
            moves=moves, backend=backend,
        )
        sampler.run_mcmc(pos, n_steps, progress=True)

    return sampler
