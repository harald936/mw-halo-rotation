"""
Run C: Joint MCMC (RC + GD-1 + Pal 5) — THE MAIN RUN.
32 walkers, 5000 steps, 12 cores.
No LMC (reported as systematic).
"""
import multiprocessing as mp
mp.set_start_method('fork', force=True)

import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.potential.composite import build_potential
from src.likelihood.joint import ln_likelihood_joint
from src.sampling.priors import ln_prior, PARAM_NAMES

import emcee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

RESULTS = os.path.join(REPO, "results")
CHAIN = os.path.join(RESULTS, "chains", "joint.h5")
PLOTS = os.path.join(RESULTS, "plots")

# -----------------------------------------------------------------------
# Posterior function (must be at module level for pickling)
# -----------------------------------------------------------------------
def ln_posterior(theta):
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

# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------
if __name__ == '__main__':
    N_WALKERS = 32
    N_STEPS = 5000
    N_CORES = 12
    N_DIM = 4

    # Tight initialization around grid-search best point
    p0 = np.array([190.0, 15.0, 0.6, 0.05])
    spread = np.array([2.0, 0.5, 0.02, 0.02])
    pos = p0 + spread * (2 * np.random.rand(N_WALKERS, N_DIM) - 1)

    # Backend
    if os.path.exists(CHAIN):
        os.remove(CHAIN)
    backend = emcee.backends.HDFBackend(CHAIN)
    backend.reset(N_WALKERS, N_DIM)

    # Moves
    moves = [
        (emcee.moves.StretchMove(), 0.7),
        (emcee.moves.DEMove(), 0.3),
    ]

    print(f"Running MCMC: {N_WALKERS} walkers, {N_STEPS} steps, {N_CORES} cores")

    with mp.Pool(N_CORES) as pool:
        sampler = emcee.EnsembleSampler(
            N_WALKERS, N_DIM, ln_posterior,
            moves=moves, backend=backend, pool=pool,
        )
        sampler.run_mcmc(pos, N_STEPS, progress=True)

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    chain = sampler.get_chain()
    print(f"\nChain: {chain.shape}")
    print(f"Acceptance: {np.mean(sampler.acceptance_fraction):.3f}")

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"Autocorrelation times: {tau}")
    except Exception as e:
        print(f"Autocorrelation failed: {e}")

    burn = 1000
    flat = sampler.get_chain(discard=burn, flat=True)
    print(f"\nPost burn-in: {len(flat)} samples")
    for i, name in enumerate(PARAM_NAMES):
        med = np.median(flat[:, i])
        lo, hi = np.percentile(flat[:, i], [16, 84])
        p95 = np.percentile(flat[:, i], 95)
        print(f"  {name:10s}: {med:.4f}  68%=[{lo:.4f}, {hi:.4f}]  95%<{p95:.4f}")

    omega_samples = flat[:, 3]
    omega_med = np.median(omega_samples)
    omega_95 = np.percentile(omega_samples, 95)
    print(f"\n*** OMEGA_P RESULT ***")
    print(f"  Median: {omega_med:.4f} km/s/kpc")
    print(f"  95% upper limit: {omega_95:.4f} km/s/kpc")
    print(f"  Bailin+2004 prediction: ~0.10 km/s/kpc")

    # Trace plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        ax.plot(chain[:, :, i], alpha=0.1, lw=0.3)
        ax.axvline(burn, color="red", ls="--", lw=1)
        ax.set_ylabel(name)
    axes[-1].set_xlabel("Step")
    plt.suptitle("Joint MCMC (RC + GD-1 + Pal 5)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "joint_trace.png"), dpi=150)

    # Corner plot
    fig2 = corner.corner(flat, labels=PARAM_NAMES,
                          quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4f")
    fig2.savefig(os.path.join(PLOTS, "joint_corner.png"), dpi=150)

    # Omega_p posterior
    fig3, ax = plt.subplots(figsize=(8, 5))
    ax.hist(omega_samples, bins=50, density=True, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(omega_med, color="red", ls="-", lw=2, label=f"Median: {omega_med:.3f}")
    ax.axvline(omega_95, color="orange", ls="--", lw=2, label=f"95% UL: {omega_95:.3f}")
    ax.axvspan(0.08, 0.12, alpha=0.2, color="green", label="Bailin+04 prediction")
    ax.set_xlabel("Omega_p (km/s/kpc)")
    ax.set_ylabel("Posterior density")
    ax.set_title("Figure Rotation Rate of the MW Dark Matter Halo")
    ax.legend()
    plt.tight_layout()
    fig3.savefig(os.path.join(PLOTS, "omega_p_posterior.png"), dpi=200)

    print("\nAll plots saved. Done!")
