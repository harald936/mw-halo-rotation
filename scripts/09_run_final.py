"""
FINAL RUN: Dynesty + Mock Streams + LMC (Orphan only) + sigma_sys
==================================================================
5 free parameters: v_h, r_h, q_z, Omega_p, sigma_sys
4 streams with 200 mock particles each
LMC perturbation on Orphan-Chenab only
sigma_sys fitted (not guessed) — model uncertainty determined by data
12 cores, estimated ~3-4 hours
"""
import multiprocessing as mp
mp.set_start_method('fork', force=True)

import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.potential.composite import build_potential
from src.potential.lmc import build_lmc_potential
from src.likelihood.rotation_curve import ln_likelihood_rc
from src.likelihood.stream_mock import mock_stream_likelihood_single

import dynesty
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

PLOTS = os.path.join(REPO, "results", "plots")
os.makedirs(PLOTS, exist_ok=True)


# -----------------------------------------------------------------------
# Parameter setup: 5D
# -----------------------------------------------------------------------
PARAM_NAMES = ["v_h", "r_h", "q_z", "Omega_p", "sigma_sys"]
BOUNDS = [
    (100.0, 300.0),   # v_h
    (5.0, 40.0),      # r_h
    (0.5, 2.0),       # q_z
    (0.0, 0.5),       # Omega_p
    (0.01, 3.0),      # sigma_sys (degrees, applied to phi2 sky track only; RV uses fixed 5 km/s)
]

def prior_transform(u):
    theta = np.empty(5)
    for i, (lo, hi) in enumerate(BOUNDS):
        theta[i] = lo + u[i] * (hi - lo)
    return theta


def log_likelihood(theta):
    v_h, r_h, q_z, Omega_p, sigma_sys = theta
    try:
        pot = build_potential(v_h, r_h, q_z, Omega_p, include_lmc=False)

        lnL = ln_likelihood_rc(pot)

        for name in ['gd1', 'pal5', 'jhelum']:
            lnL += mock_stream_likelihood_single(pot, name, sigma_sys)

        # Orphan-Chenab with LMC rebuilt for current halo parameters
        lmc_pot, _ = build_lmc_potential(pot)
        pot_lmc = pot + [lmc_pot]
        lnL += mock_stream_likelihood_single(pot_lmc, 'orphan', sigma_sys)

        return lnL if np.isfinite(lnL) else -1e10
    except (RuntimeError, ValueError):
        return -1e10


# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------
if __name__ == '__main__':
    N_CORES = 12

    print(f"FINAL RUN: 5 params, {N_CORES} cores, mock streams + LMC")
    print(f"Params: {PARAM_NAMES}")
    print(f"Bounds: {BOUNDS}")

    with mp.Pool(N_CORES) as pool:
        sampler = dynesty.NestedSampler(
            log_likelihood, prior_transform, ndim=5,
            nlive=250, pool=pool, queue_size=N_CORES,
        )
        sampler.run_nested(print_progress=True)

    results = sampler.results

    # Save raw results to disk
    results_file = os.path.join(REPO, "results", "dynesty_final.npz")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    np.savez(results_file,
             samples=results.samples,
             logwt=results.logwt,
             logz=results.logz,
             logzerr=results.logzerr,
             ncall=results.ncall)
    print(f"Results saved to {results_file}")

    from dynesty import utils as dyfunc
    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    resampled = dyfunc.resample_equal(samples, weights)

    print(f"\n{'='*55}")
    print(f"FINAL RESULTS")
    print(f"{'='*55}")
    print(f"Log-evidence: {results.logz[-1]:.1f} +/- {results.logzerr[-1]:.1f}")
    print(f"Evaluations: {sum(results.ncall)}")

    for i, name in enumerate(PARAM_NAMES):
        med = np.median(resampled[:, i])
        lo, hi = np.percentile(resampled[:, i], [16, 84])
        p95 = np.percentile(resampled[:, i], 95)
        print(f"  {name:10s}: {med:.4f}  68%=[{lo:.4f}, {hi:.4f}]  95%<{p95:.4f}")

    omega = resampled[:, 3]
    sigma = resampled[:, 4]
    print(f"\n*** OMEGA_P: {np.median(omega):.4f} km/s/kpc, 95% UL < {np.percentile(omega,95):.4f} ***")
    print(f"*** SIGMA_SYS: {np.median(sigma):.3f} deg (fitted, not guessed) ***")

    # Corner plot
    fig = corner.corner(resampled, labels=PARAM_NAMES,
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4f")
    fig.savefig(os.path.join(PLOTS, "final_corner.png"), dpi=150)

    # Omega_p posterior
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.hist(omega, bins=50, density=True, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(np.median(omega), color="red", lw=2, label=f"Median: {np.median(omega):.3f}")
    ax.axvline(np.percentile(omega,95), color="orange", ls="--", lw=2,
               label=f"95% UL: {np.percentile(omega,95):.3f}")
    ax.axvspan(0.08, 0.12, alpha=0.2, color="green", label="Bailin+04")
    ax.set_xlabel("Omega_p (km/s/kpc)")
    ax.set_ylabel("Posterior density")
    ax.set_title("Figure Rotation Rate — Final (Mock Streams + LMC + sigma_sys)")
    ax.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(PLOTS, "final_omega_p.png"), dpi=200)

    print("\nDone!")
