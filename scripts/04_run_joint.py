"""
Run C: Joint MCMC (RC + GD-1 stream) — THE MAIN RUN.

64 walkers, 5000 steps, 11 cores.
Expected runtime: ~8 hours.

This is the run that constrains Omega_p.
"""

import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.sampling.sampler import run_sampler
from src.sampling.priors import PARAM_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

RESULTS = os.path.join(REPO, "results")
CHAIN = os.path.join(RESULTS, "chains", "joint.h5")
PLOTS = os.path.join(RESULTS, "plots")

if os.path.exists(CHAIN):
    os.remove(CHAIN)

# Initialize from RC-only posterior (v_h~188, r_h~21.5)
# with wider spread to explore stream-preferred region too
sampler = run_sampler(
    n_walkers=64, n_steps=5000,
    p0=[188.0, 21.5, 0.93, 0.05],
    spread=[15.0, 5.0, 0.15, 0.04],
    backend_file=CHAIN, rc_only=False, n_cores=11,
)

chain = sampler.get_chain()
print(f"\nChain: {chain.shape}")
print(f"Acceptance: {np.mean(sampler.acceptance_fraction):.3f}")

# Convergence: autocorrelation time
try:
    tau = sampler.get_autocorr_time(quiet=True)
    print(f"Autocorrelation times: {tau}")
    print(f"Chain length / tau: {chain.shape[0] / tau}")
except Exception as e:
    print(f"Autocorrelation estimate failed: {e}")

burn = 1000
flat = sampler.get_chain(discard=burn, flat=True)
print(f"\nPost burn-in: {len(flat)} samples")
for i, name in enumerate(PARAM_NAMES):
    med = np.median(flat[:, i])
    lo, hi = np.percentile(flat[:, i], [16, 84])
    p95 = np.percentile(flat[:, i], 95)
    print(f"  {name:10s}: {med:.4f}  68%=[{lo:.4f}, {hi:.4f}]  95%<{p95:.4f}")

# Omega_p headline result
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
plt.suptitle("Joint MCMC (RC + GD-1)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "joint_trace.png"), dpi=150)

# Corner plot
fig2 = corner.corner(flat, labels=PARAM_NAMES,
                      quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4f")
fig2.savefig(os.path.join(PLOTS, "joint_corner.png"), dpi=150)

# Omega_p marginal posterior
fig3, ax = plt.subplots(figsize=(8, 5))
ax.hist(omega_samples, bins=50, density=True, color="steelblue", alpha=0.7,
        edgecolor="white")
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
