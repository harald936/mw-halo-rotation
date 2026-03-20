"""
SIGNED OMEGA RUN: Dynesty + Mock Streams + LMC + sigma_sys
==========================================================
5 free parameters: v_h, r_h, q_z, Omega_p, sigma_sys
Omega_p prior: U(-0.5, 0.5) — allows retrograde rotation
200 mock particles per stream (spray method)
LMC perturbation on Orphan-Chenab only
317 data points across 6 observable channels
12 cores, estimated ~20-25 hours
"""
import multiprocessing as mp
mp.set_start_method('fork', force=True)

import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Override N_PARTICLES to 200 BEFORE importing the likelihood
import src.likelihood.stream_mock as stream_mock_module
stream_mock_module.N_PARTICLES = 200

from src.potential.composite import build_potential
from src.potential.lmc import build_lmc_potential
from src.likelihood.rotation_curve import ln_likelihood_rc
from src.likelihood.stream_mock import ln_likelihood_mock_streams

import dynesty
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

PLOTS = os.path.join(REPO, "results", "plots")
os.makedirs(PLOTS, exist_ok=True)


PARAM_NAMES = ["v_h", "r_h", "q_z", "Omega_p", "sigma_sys"]
BOUNDS = [
    (100.0, 300.0),   # v_h (km/s)
    (5.0, 40.0),      # r_h (kpc)
    (0.5, 2.0),       # q_z
    (-0.5, 0.5),      # Omega_p (km/s/kpc) — SIGNED
    (0.01, 3.0),      # sigma_sys (deg)
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

        # Orphan-Chenab gets LMC rebuilt for current halo parameters
        try:
            lmc_pot, _ = build_lmc_potential(pot)
            pot_lmc = pot + [lmc_pot]
        except (RuntimeError, ValueError):
            return -1e10

        lnL += ln_likelihood_mock_streams(pot, sigma_sys, pot_with_lmc=pot_lmc)

        return lnL if np.isfinite(lnL) else -1e10
    except (RuntimeError, ValueError):
        return -1e10


if __name__ == '__main__':
    N_CORES = 12

    print(f"SIGNED OMEGA RUN — 200 particles, 317 data points")
    print(f"  Params: {PARAM_NAMES}")
    print(f"  Bounds: {BOUNDS}")
    print(f"  N_PARTICLES: {stream_mock_module.N_PARTICLES}")
    print(f"  Cores: {N_CORES}")
    print(f"  nlive: 250")

    with mp.Pool(N_CORES) as pool:
        sampler = dynesty.NestedSampler(
            log_likelihood, prior_transform, ndim=5,
            nlive=250, pool=pool, queue_size=N_CORES,
        )
        sampler.run_nested(print_progress=True)

    results = sampler.results

    # Save raw results
    results_file = os.path.join(REPO, "results", "dynesty_final_signed.npz")
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
    print(f"SIGNED OMEGA RESULTS — 200 particles")
    print(f"{'='*55}")
    print(f"Log-evidence: {results.logz[-1]:.1f} +/- {results.logzerr[-1]:.1f}")
    print(f"Evaluations: {sum(results.ncall)}")

    for i, name in enumerate(PARAM_NAMES):
        med = np.median(resampled[:, i])
        lo, hi = np.percentile(resampled[:, i], [16, 84])
        lo95, hi95 = np.percentile(resampled[:, i], [2.5, 97.5])
        print(f"  {name:10s}: {med:.4f}  68%=[{lo:.4f}, {hi:.4f}]  95%=[{lo95:.4f}, {hi95:.4f}]")

    omega = resampled[:, 3]
    sigma = resampled[:, 4]
    med_o = np.median(omega)
    lo_o, hi_o = np.percentile(omega, [16, 84])
    lo95_o, hi95_o = np.percentile(omega, [2.5, 97.5])
    print(f"\n*** OMEGA_P: {med_o:.4f} km/s/kpc")
    print(f"    68% CI: [{lo_o:.4f}, {hi_o:.4f}]")
    print(f"    95% CI: [{lo95_o:.4f}, {hi95_o:.4f}]")
    print(f"*** SIGMA_SYS: {np.median(sigma):.3f} deg ***")

    # Corner plot
    fig = corner.corner(resampled, labels=PARAM_NAMES,
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4f")
    fig.savefig(os.path.join(PLOTS, "final_signed_corner.png"), dpi=150)

    # Omega_p posterior
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.hist(omega, bins=60, density=True, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(med_o, color="red", lw=2, label=f"Median: {med_o:.3f}")
    ax.axvline(lo_o, color="red", lw=1.5, ls="--")
    ax.axvline(hi_o, color="red", lw=1.5, ls="--", label=f"68%: [{lo_o:.3f}, {hi_o:.3f}]")
    ax.axvline(0, color="black", lw=1, ls=":", alpha=0.5, label="Static halo")
    ax.axvspan(0.08, 0.12, alpha=0.2, color="green", label="Bailin+04")
    ax.axvspan(-0.12, -0.08, alpha=0.2, color="green")
    ax.set_xlabel(r"$\Omega_p$ (km/s/kpc)")
    ax.set_ylabel("Posterior density")
    ax.set_title(r"Figure Rotation Rate — Signed $\Omega_p$ (200 particles)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig2.savefig(os.path.join(PLOTS, "final_signed_omega_p.png"), dpi=200)

    print("\nDone!")
