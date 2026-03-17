"""
Run A: Proper RC-only MCMC.
32 walkers, 3000 steps, 1 core.
Expected runtime: ~1 hour.
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
CHAIN = os.path.join(RESULTS, "chains", "rc_proper.h5")
PLOTS = os.path.join(RESULTS, "plots")

if os.path.exists(CHAIN):
    os.remove(CHAIN)

sampler = run_sampler(
    n_walkers=32, n_steps=3000,
    p0=[170.0, 21.0, 0.93, 0.05],
    spread=[8.0, 3.0, 0.1, 0.04],
    backend_file=CHAIN, rc_only=True, n_cores=1,
)

chain = sampler.get_chain()
print(f"\nChain: {chain.shape}, acceptance: {np.mean(sampler.acceptance_fraction):.3f}")

burn = 500
flat = sampler.get_chain(discard=burn, flat=True)
print(f"Post burn-in: {len(flat)} samples")
for i, name in enumerate(PARAM_NAMES):
    med = np.median(flat[:, i])
    lo, hi = np.percentile(flat[:, i], [16, 84])
    print(f"  {name:10s}: {med:.3f}  [{lo:.3f}, {hi:.3f}]")

# Trace
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for i, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
    ax.plot(chain[:, :, i], alpha=0.15, lw=0.3)
    ax.axvline(burn, color="red", ls="--", lw=1)
    ax.set_ylabel(name)
axes[-1].set_xlabel("Step")
plt.suptitle("RC-Only MCMC (proper run)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "rc_proper_trace.png"), dpi=150)

# Corner
fig2 = corner.corner(flat, labels=PARAM_NAMES,
                      quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".2f")
fig2.savefig(os.path.join(PLOTS, "rc_proper_corner.png"), dpi=150)
print("Done!")
