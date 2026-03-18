"""
Run D: Dynesty nested sampling with all 4 streams.
=====================================================
Uses dynesty instead of emcee for guaranteed convergence.
No acceptance rate issues, computes Bayesian evidence.

4 streams: GD-1 + Pal 5 + Jhelum + Orphan-Chenab (~315 deg coverage)
+ Eilers+2019 rotation curve
No LMC (systematic uncertainty)
Tilted triaxial halo (b=0.9, tilt=18 deg)
"""
import multiprocessing as mp
mp.set_start_method('fork', force=True)

import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.potential.composite import build_potential
from src.likelihood.joint import ln_likelihood_joint
from src.sampling.priors import PARAM_BOUNDS, PARAM_NAMES

import dynesty
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

PLOTS = os.path.join(REPO, "results", "plots")

# -----------------------------------------------------------------------
# Prior transform for dynesty (unit cube -> physical)
# -----------------------------------------------------------------------
bounds = [PARAM_BOUNDS[n] for n in PARAM_NAMES]

def prior_transform(u):
    """Transform unit cube [0,1]^4 to physical parameters."""
    theta = np.empty(len(u))
    for i, (lo, hi) in enumerate(bounds):
        theta[i] = lo + u[i] * (hi - lo)
    return theta

def log_likelihood(theta):
    v_h, r_h, q_z, Omega_p = theta
    try:
        pot = build_potential(v_h, r_h, q_z, Omega_p, include_lmc=False)
        return ln_likelihood_joint(pot)
    except Exception:
        return -1e10

# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------
if __name__ == '__main__':
    N_CORES = 12

    print(f"Running dynesty: 4 params, {N_CORES} cores, 4 streams + RC")
    print(f"Bounds: {bounds}")

    with mp.Pool(N_CORES) as pool:
        sampler = dynesty.NestedSampler(
            log_likelihood, prior_transform, ndim=4,
            nlive=200, pool=pool, queue_size=N_CORES,
        )
        sampler.run_nested(print_progress=True)

    results = sampler.results

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    from dynesty import utils as dyfunc

    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samples, weights)

    # Resample for corner plot
    resampled = dyfunc.resample_equal(samples, weights)

    print(f"\n{'='*55}")
    print(f"DYNESTY RESULTS")
    print(f"{'='*55}")
    print(f"Log-evidence: {results.logz[-1]:.1f} +/- {results.logzerr[-1]:.1f}")
    print(f"Iterations: {results.niter}")
    print(f"Likelihood evaluations: {sum(results.ncall)}")

    for i, name in enumerate(PARAM_NAMES):
        med = np.median(resampled[:, i])
        lo, hi = np.percentile(resampled[:, i], [16, 84])
        p95 = np.percentile(resampled[:, i], 95)
        print(f"  {name:10s}: {med:.4f}  68%=[{lo:.4f}, {hi:.4f}]  95%<{p95:.4f}")

    omega = resampled[:, 3]
    print(f"\n*** OMEGA_P RESULT ***")
    print(f"  Median: {np.median(omega):.4f} km/s/kpc")
    print(f"  95% upper limit: {np.percentile(omega, 95):.4f} km/s/kpc")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    # Corner
    fig = corner.corner(resampled, labels=PARAM_NAMES,
                        quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        title_fmt=".4f")
    fig.savefig(os.path.join(PLOTS, "dynesty_corner.png"), dpi=150)

    # Omega_p posterior
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.hist(omega, bins=50, density=True, color="steelblue", alpha=0.7,
            edgecolor="white", weights=np.ones_like(omega))
    ax.axvline(np.median(omega), color="red", lw=2,
               label=f"Median: {np.median(omega):.3f}")
    ax.axvline(np.percentile(omega, 95), color="orange", ls="--", lw=2,
               label=f"95% UL: {np.percentile(omega, 95):.3f}")
    ax.axvspan(0.08, 0.12, alpha=0.2, color="green", label="Bailin+04")
    ax.set_xlabel("Omega_p (km/s/kpc)")
    ax.set_ylabel("Posterior density")
    ax.set_title("Figure Rotation Rate — 4-Stream Constraint")
    ax.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(PLOTS, "dynesty_omega_p.png"), dpi=200)

    print("\nAll plots saved. Done!")
