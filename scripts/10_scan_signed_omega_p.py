"""
Quick diagnostic: scan log-likelihood vs signed Omega_p.
=========================================================
Fixes v_h, r_h, q_z, sigma_sys at best-fit (or fallback) values
and evaluates total + per-dataset lnL on a grid of Omega_p.

No sampling — just a 1D grid scan. Takes ~20-40 minutes.
"""
import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.potential.composite import build_potential
from src.potential.lmc import build_lmc_potential
from src.likelihood.rotation_curve import ln_likelihood_rc
from src.likelihood.stream_mock import mock_stream_likelihood_single

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------
# Load best-fit params or use fallback
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
    # Weighted median
    def wmedian(x, w):
        idx = np.argsort(x)
        cs = np.cumsum(w[idx])
        return x[idx][np.searchsorted(cs, 0.5)]
    v_h = wmedian(samples[:, 0], weights)
    r_h = wmedian(samples[:, 1], weights)
    q_z = wmedian(samples[:, 2], weights)
    sigma_sys = wmedian(samples[:, 4], weights)
    print(f"  v_h={v_h:.1f}, r_h={r_h:.1f}, q_z={q_z:.3f}, sigma_sys={sigma_sys:.3f}")
else:
    print("No dynesty results yet — using fallback values.")
    v_h, r_h, q_z, sigma_sys = 160.0, 16.0, 0.93, 0.3
    print(f"  v_h={v_h}, r_h={r_h}, q_z={q_z}, sigma_sys={sigma_sys}")

# -----------------------------------------------------------------------
# Grid scan
# -----------------------------------------------------------------------
Omega_grid = np.linspace(-0.5, 0.5, 41)
print(f"\nScanning {len(Omega_grid)} Omega_p values from {Omega_grid[0]} to {Omega_grid[-1]}...")

results = []
for i, omega in enumerate(Omega_grid):
    try:
        pot = build_potential(v_h, r_h, q_z, omega, include_lmc=False)
        lnL_rc = ln_likelihood_rc(pot)

        lnL_gd1 = mock_stream_likelihood_single(pot, 'gd1', sigma_sys)
        lnL_pal5 = mock_stream_likelihood_single(pot, 'pal5', sigma_sys)
        lnL_jhelum = mock_stream_likelihood_single(pot, 'jhelum', sigma_sys)

        try:
            lmc_pot, _ = build_lmc_potential(pot)
            pot_lmc = pot + [lmc_pot]
            lnL_orphan = mock_stream_likelihood_single(pot_lmc, 'orphan', sigma_sys)
            if not np.isfinite(lnL_orphan):
                lnL_orphan = -1e10
        except Exception:
            lnL_orphan = -1e10

        total = lnL_rc + lnL_gd1 + lnL_pal5 + lnL_jhelum + lnL_orphan
        if not np.isfinite(total):
            total = -1e10
    except (RuntimeError, ValueError, Exception):
        lnL_rc = lnL_gd1 = lnL_pal5 = lnL_jhelum = lnL_orphan = total = -1e10

    results.append({
        'omega_p': omega,
        'lnL_rc': lnL_rc,
        'lnL_gd1': lnL_gd1,
        'lnL_pal5': lnL_pal5,
        'lnL_jhelum': lnL_jhelum,
        'lnL_orphan': lnL_orphan,
        'lnL_total': total,
    })
    print(f"  [{i+1}/{len(Omega_grid)}] Omega_p={omega:+.3f}  total={total:.1f}")

df = pd.DataFrame(results)

# Save CSV
tables_dir = os.path.join(REPO, "results", "tables")
os.makedirs(tables_dir, exist_ok=True)
csv_path = os.path.join(tables_dir, "omega_scan.csv")
df.to_csv(csv_path, index=False, float_format="%.4f")
print(f"\nSaved to {csv_path}")

# -----------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------
valid = df.lnL_total > -1e9
dfv = df[valid]
best_idx = dfv.lnL_total.idxmax()
best_omega = dfv.loc[best_idx, 'omega_p']
best_lnL = dfv.loc[best_idx, 'lnL_total']

zero_row = dfv.iloc[(dfv.omega_p - 0.0).abs().argsort().iloc[0]]
lnL_at_zero = zero_row['lnL_total']
delta_lnL = best_lnL - lnL_at_zero

# Which dataset contributes most
datasets = ['lnL_rc', 'lnL_gd1', 'lnL_pal5', 'lnL_jhelum', 'lnL_orphan']
delta_per_dataset = {}
for d in datasets:
    d_best = dfv.loc[best_idx, d]
    d_zero = zero_row[d]
    delta_per_dataset[d] = d_best - d_zero
strongest = max(delta_per_dataset, key=delta_per_dataset.get)

print(f"\n{'='*55}")
print(f"OMEGA SCAN RESULTS")
print(f"{'='*55}")
print(f"Best grid Omega_p: {best_omega:+.3f} km/s/kpc")
print(f"Delta lnL (best - zero): {delta_lnL:.2f}")
print(f"Strongest contributor: {strongest} (Delta lnL = {delta_per_dataset[strongest]:.2f})")
for d in datasets:
    print(f"  {d:15s}: Delta lnL = {delta_per_dataset[d]:+.2f}")

if abs(delta_lnL) < 1.0:
    print("\n=> No significant preference away from Omega_p=0.")
    print("   Current setup is upper-limit dominated.")
elif delta_lnL > 2.0:
    print(f"\n=> Significant preference for Omega_p={best_omega:+.3f} (Delta lnL={delta_lnL:.1f}).")
    print("   A signed measurement may be possible!")
else:
    print(f"\n=> Mild preference for Omega_p={best_omega:+.3f} (Delta lnL={delta_lnL:.1f}).")
    print("   Marginal — full sampling needed to confirm.")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

ax = axes[0]
ax.plot(dfv.omega_p, dfv.lnL_total - best_lnL, 'k-', lw=2, label='Total')
ax.axvline(0, color='gray', ls=':', lw=1, label='Static halo')
ax.axvline(best_omega, color='red', ls='--', lw=1.5, label=f'Best: {best_omega:+.3f}')
ax.axhline(-2, color='orange', ls=':', alpha=0.5, label=r'$\Delta \ln L = -2$ (~2$\sigma$)')
ax.set_ylabel(r'$\Delta \ln L$ (relative to best)', fontsize=12)
ax.set_title(f'Omega_p Grid Scan (v_h={v_h:.0f}, r_h={r_h:.0f}, q_z={q_z:.2f})', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

ax = axes[1]
colors = {'lnL_rc': 'gray', 'lnL_gd1': '#2196F3', 'lnL_pal5': '#FF9800',
          'lnL_jhelum': '#4CAF50', 'lnL_orphan': '#E91E63'}
labels = {'lnL_rc': 'RC', 'lnL_gd1': 'GD-1', 'lnL_pal5': 'Pal 5',
          'lnL_jhelum': 'Jhelum', 'lnL_orphan': 'Orphan-Chenab'}
for d in datasets:
    vals = dfv[d] - dfv[d].max()
    ax.plot(dfv.omega_p, vals, '-', color=colors[d], lw=1.5, label=labels[d])
ax.axvline(0, color='gray', ls=':', lw=1)
ax.set_xlabel(r'$\Omega_p$ (km/s/kpc)', fontsize=12)
ax.set_ylabel(r'$\Delta \ln L$ per dataset', fontsize=12)
ax.legend(fontsize=9, ncol=3)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plot_path = os.path.join(REPO, "results", "plots", "omega_scan.png")
fig.savefig(plot_path, dpi=200)
print(f"Saved plot to {plot_path}")
print("\nDone!")
