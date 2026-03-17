"""
Run A: RC-only MCMC (validation run).

Fits (v_h, r_h) using only the rotation curve, with q_z and Omega_p
as free but weakly constrained nuisance parameters. This validates
the potential model and sampler before adding the stream likelihood.

Expected runtime: ~5-10 minutes.
"""

import sys
import os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.sampling.sampler import run_sampler
from src.sampling.priors import PARAM_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(REPO, "results")
CHAIN_FILE = os.path.join(RESULTS_DIR, "chains", "rc_only.h5")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

# -----------------------------------------------------------------------
# Run MCMC
# -----------------------------------------------------------------------
# Remove old chain file if it exists
if os.path.exists(CHAIN_FILE):
    os.remove(CHAIN_FILE)

sampler = run_sampler(
    n_walkers=32,
    n_steps=500,
    p0=[160.0, 16.0, 0.93, 0.05],
    spread=[10.0, 4.0, 0.1, 0.04],
    backend_file=CHAIN_FILE,
    rc_only=True,
    n_cores=1,
)

# -----------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------
chain = sampler.get_chain()
log_prob = sampler.get_log_prob()
print(f"\nChain shape: {chain.shape}")
print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

# Discard burn-in (first 30%)
burn = int(0.3 * chain.shape[0])
flat = sampler.get_chain(discard=burn, flat=True)
print(f"\nPost burn-in samples: {len(flat)}")

for i, name in enumerate(PARAM_NAMES):
    med = np.median(flat[:, i])
    lo = np.percentile(flat[:, i], 16)
    hi = np.percentile(flat[:, i], 84)
    print(f"  {name:10s}: {med:.3f}  [{lo:.3f}, {hi:.3f}]")

# -----------------------------------------------------------------------
# Trace plot
# -----------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for i, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
    ax.plot(chain[:, :, i], alpha=0.2, lw=0.5)
    ax.axvline(burn, color="red", ls="--", lw=1, label="burn-in")
    ax.set_ylabel(name)
    if i == 0:
        ax.legend(fontsize=8)
axes[-1].set_xlabel("Step")
plt.suptitle("RC-Only MCMC Trace", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "rc_only_trace.png"), dpi=150)
print(f"\nSaved trace plot")

# -----------------------------------------------------------------------
# Corner plot
# -----------------------------------------------------------------------
try:
    import corner
    fig2 = corner.corner(
        flat, labels=PARAM_NAMES,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_fmt=".3f",
    )
    fig2.savefig(os.path.join(PLOT_DIR, "rc_only_corner.png"), dpi=150)
    print("Saved corner plot")
except ImportError:
    print("corner not installed — skipping corner plot")

print("\nDone!")
