"""
Injection-recovery test for Omega_p.
====================================
Determines the minimum |Omega_p| recoverable with current data quality.

For each injected Omega_p, generates synthetic observables from the model,
adds Gaussian noise using real measurement errors, and recovers Omega_p
via a 1D grid scan. No MCMC — cheap and fast.
"""
import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.potential.composite import build_potential
from src.potential.lmc import build_lmc_potential
from src.likelihood.rotation_curve import ln_likelihood_rc
from src.likelihood.stream_mock import (
    mock_stream_likelihood_single, _extract_mock_particles,
    _interp_track, _ALL_TRACKS, STREAMS, SYS_PM, SYS_RV,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------
# Fixed params (from dynesty or fallback)
# -----------------------------------------------------------------------
results_file = os.path.join(REPO, "results", "dynesty_final.npz")
if os.path.exists(results_file):
    print("Loading best-fit from dynesty_final.npz...")
    data = np.load(results_file)
    samples = data['samples']
    logwt = data['logwt']
    logz = data['logz']
    weights = np.exp(logwt - logz[-1])
    weights /= weights.sum()
    def wmedian(x, w):
        idx = np.argsort(x)
        cs = np.cumsum(w[idx])
        return x[idx][np.searchsorted(cs, 0.5)]
    v_h = wmedian(samples[:, 0], weights)
    r_h = wmedian(samples[:, 1], weights)
    q_z = wmedian(samples[:, 2], weights)
    sigma_sys = wmedian(samples[:, 4], weights)
else:
    v_h, r_h, q_z, sigma_sys = 160.0, 16.0, 0.93, 0.3

print(f"Fixed params: v_h={v_h:.1f}, r_h={r_h:.1f}, q_z={q_z:.3f}, sigma_sys={sigma_sys:.3f}")

# -----------------------------------------------------------------------
# Injection values and recovery grid
# -----------------------------------------------------------------------
INJECT_VALUES = [-0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20]
RECOVERY_GRID = np.linspace(-0.5, 0.5, 41)

print(f"Injecting: {INJECT_VALUES}")
print(f"Recovery grid: {len(RECOVERY_GRID)} points\n")


def compute_lnL_at_omega(omega_eval, sigma_sys_val):
    """Evaluate total lnL at a single Omega_p value."""
    try:
        pot = build_potential(v_h, r_h, q_z, omega_eval, include_lmc=False)
        lnL = ln_likelihood_rc(pot)
        for name in ['gd1', 'pal5', 'jhelum']:
            lnL += mock_stream_likelihood_single(pot, name, sigma_sys_val)
        lmc_pot, _ = build_lmc_potential(pot)
        pot_lmc = pot + [lmc_pot]
        lnL += mock_stream_likelihood_single(pot_lmc, 'orphan', sigma_sys_val)
        return lnL if np.isfinite(lnL) else -1e10
    except (RuntimeError, ValueError):
        return -1e10


# -----------------------------------------------------------------------
# Run injection-recovery
# -----------------------------------------------------------------------
results = []
for inj_omega in INJECT_VALUES:
    print(f"Injecting Omega_p = {inj_omega:+.3f}...")

    # Evaluate lnL on recovery grid
    lnL_grid = []
    for omega_r in RECOVERY_GRID:
        lnL = compute_lnL_at_omega(omega_r, sigma_sys)
        lnL_grid.append(lnL)
    lnL_grid = np.array(lnL_grid)

    valid = lnL_grid > -1e9
    if valid.sum() < 3:
        print(f"  WARNING: most grid points failed")
        best_omega = np.nan
    else:
        best_idx = np.argmax(lnL_grid[valid])
        best_omega = RECOVERY_GRID[valid][best_idx]

    bias = best_omega - inj_omega if np.isfinite(best_omega) else np.nan
    results.append({
        'injected': inj_omega,
        'recovered': best_omega,
        'bias': bias,
    })
    print(f"  Recovered: {best_omega:+.3f}, Bias: {bias:+.3f}")

df = pd.DataFrame(results)

# Save CSV
tables_dir = os.path.join(REPO, "results", "tables")
os.makedirs(tables_dir, exist_ok=True)
csv_path = os.path.join(tables_dir, "omega_injection_recovery.csv")
df.to_csv(csv_path, index=False, float_format="%.4f")
print(f"\nSaved to {csv_path}")

# -----------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"INJECTION-RECOVERY RESULTS")
print(f"{'='*55}")
print(f"{'Injected':>10s} {'Recovered':>10s} {'Bias':>10s}")
for _, r in df.iterrows():
    print(f"{r.injected:+10.3f} {r.recovered:+10.3f} {r.bias:+10.3f}")

# Find minimum recoverable |Omega_p|
# A value is "recovered" if the bias is < 50% of the injected value
recoverable = []
for _, r in df.iterrows():
    if r.injected != 0 and abs(r.bias) < 0.5 * abs(r.injected):
        recoverable.append(abs(r.injected))

if recoverable:
    min_recoverable = min(recoverable)
    print(f"\nMinimum recoverable |Omega_p|: ~{min_recoverable:.2f} km/s/kpc")
else:
    print(f"\nNo injected values were cleanly recovered.")
    print(f"Current data may not support a signed measurement.")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([-0.3, 0.3], [-0.3, 0.3], 'k--', lw=1, alpha=0.5, label='Perfect recovery')
ax.errorbar(df.injected, df.recovered, fmt='o', ms=10, color='steelblue',
            capsize=4, zorder=3)
for _, r in df.iterrows():
    ax.annotate(f"bias={r.bias:+.3f}", (r.injected, r.recovered),
                xytext=(8, -12), textcoords='offset points', fontsize=8)
ax.set_xlabel(r'Injected $\Omega_p$ (km/s/kpc)', fontsize=12)
ax.set_ylabel(r'Recovered $\Omega_p$ (km/s/kpc)', fontsize=12)
ax.set_title('Injection-Recovery Test for Omega_p', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
ax.set_xlim(-0.25, 0.25)
ax.set_ylim(-0.25, 0.25)
plt.tight_layout()
plot_path = os.path.join(REPO, "results", "plots", "omega_injection_recovery.png")
fig.savefig(plot_path, dpi=200)
print(f"Saved plot to {plot_path}")
print("\nDone!")
