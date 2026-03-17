"""
GD-1 Stream Track Binning
==========================
Bins the cleaned GD-1 member catalog into a smooth stream track
suitable for orbit fitting.

Input:  data/gd1/gd1_members_cleaned.csv
Output: data/gd1/gd1_track.csv
        results/plots/gd1_track_binned.png

Method:
  - Adaptive binning in phi1 with minimum effective sample size
  - Weighted statistics using membership probability as weights
  - Bootstrap resampling (10,000 iterations) for robust error estimation
  - Iterative 2.5-sigma clipping within each bin to reject remaining outliers
  - Gap/spur region (phi1 = -45 to -28) masked per Bonaca+2019
  - Three observables binned: phi2, pm1, pm2 (and RV where available)

References:
  Bonaca et al. 2019, ApJL, 880, L38 (gap/spur identification)
  Tavangar & Price-Whelan 2025, arXiv:2502.13236 (source catalog)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data", "gd1")
PLOT_DIR = os.path.join(REPO, "results", "plots")

np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
P_THRESHOLD = 0.5          # minimum membership probability
PHI1_MIN = -80.0           # stream extent (avoid noisy edges)
PHI1_MAX = 10.0
GAP_SPUR_MIN = -45.0       # mask the gap/spur region (Bonaca+2019)
GAP_SPUR_MAX = -28.0
N_BINS = 45                # number of phi1 bins (outside masked region)
MIN_NEFF = 10              # minimum effective sample size per bin
N_BOOTSTRAP = 10000        # bootstrap iterations for error estimation
SIGMA_CLIP = 2.5           # iterative sigma-clipping threshold
MAX_CLIP_ITER = 5          # maximum clipping iterations

# ---------------------------------------------------------------------------
# 1. Load cleaned catalog
# ---------------------------------------------------------------------------
print("Loading cleaned GD-1 catalog...")
df = pd.read_csv(os.path.join(DATA_DIR, "gd1_members_cleaned.csv"))
print(f"  Total members: {len(df)}")

# Apply phi1 range cut
mask = (df.phi1 >= PHI1_MIN) & (df.phi1 <= PHI1_MAX) & (df.memb_prob >= P_THRESHOLD)
df = df[mask].copy()
print(f"  After phi1 range [{PHI1_MIN}, {PHI1_MAX}] cut: {len(df)}")

# Flag gap/spur region
df["in_gap_spur"] = (df.phi1 >= GAP_SPUR_MIN) & (df.phi1 <= GAP_SPUR_MAX)
n_gap = df.in_gap_spur.sum()
print(f"  In gap/spur region [{GAP_SPUR_MIN}, {GAP_SPUR_MAX}]: {n_gap}")

# ---------------------------------------------------------------------------
# 2. Define bin edges
# ---------------------------------------------------------------------------
# Create bins only outside the gap/spur region
phi1_range_left = PHI1_MAX - GAP_SPUR_MAX   # right of gap
phi1_range_right = GAP_SPUR_MIN - PHI1_MIN  # left of gap
total_range = phi1_range_left + phi1_range_right

n_bins_left = int(np.round(N_BINS * phi1_range_right / total_range))
n_bins_right = N_BINS - n_bins_left

edges_left = np.linspace(PHI1_MIN, GAP_SPUR_MIN, n_bins_left + 1)
edges_right = np.linspace(GAP_SPUR_MAX, PHI1_MAX, n_bins_right + 1)

print(f"  Bins left of gap: {n_bins_left}, right of gap: {n_bins_right}")

# ---------------------------------------------------------------------------
# 3. Weighted statistics with bootstrap errors and sigma-clipping
# ---------------------------------------------------------------------------
def weighted_median(values, weights):
    """Compute weighted median."""
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumw = np.cumsum(sorted_weights)
    cutoff = 0.5 * cumw[-1]
    idx = np.searchsorted(cumw, cutoff)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def weighted_std(values, weights):
    """Compute weighted standard deviation."""
    avg = np.average(values, weights=weights)
    variance = np.average((values - avg) ** 2, weights=weights)
    return np.sqrt(variance)


def sigma_clip_weighted(values, weights, sigma=2.5, max_iter=5):
    """Iterative sigma clipping with weights."""
    mask = np.ones(len(values), dtype=bool)
    for _ in range(max_iter):
        if np.sum(mask) < 3:
            break
        med = weighted_median(values[mask], weights[mask])
        std = weighted_std(values[mask], weights[mask])
        if std == 0:
            break
        new_mask = np.abs(values - med) < sigma * std
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask
    return mask


def bootstrap_weighted_median(values, weights, n_boot=10000):
    """Bootstrap estimate of weighted median and its uncertainty."""
    n = len(values)
    medians = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        medians[i] = weighted_median(values[idx], weights[idx])
    return np.median(medians), np.std(medians)


def bin_observable(phi1, obs, weights, edges, n_bootstrap=N_BOOTSTRAP):
    """Bin an observable along phi1 with sigma-clipping and bootstrap errors."""
    n_bins = len(edges) - 1
    centers = np.empty(n_bins)
    medians = np.empty(n_bins)
    errors = np.empty(n_bins)
    n_eff = np.empty(n_bins)
    n_stars = np.empty(n_bins, dtype=int)
    valid = np.ones(n_bins, dtype=bool)

    for i in range(n_bins):
        in_bin = (phi1 >= edges[i]) & (phi1 < edges[i + 1])
        if i == n_bins - 1:  # include right edge for last bin
            in_bin = (phi1 >= edges[i]) & (phi1 <= edges[i + 1])

        vals = obs[in_bin]
        w = weights[in_bin]

        if len(vals) < 3:
            valid[i] = False
            centers[i] = (edges[i] + edges[i + 1]) / 2
            medians[i] = np.nan
            errors[i] = np.nan
            n_eff[i] = 0
            n_stars[i] = len(vals)
            continue

        # Sigma-clip
        clip_mask = sigma_clip_weighted(vals, w, sigma=SIGMA_CLIP, max_iter=MAX_CLIP_ITER)
        vals_clipped = vals[clip_mask]
        w_clipped = w[clip_mask]

        if len(vals_clipped) < 3:
            valid[i] = False
            centers[i] = (edges[i] + edges[i + 1]) / 2
            medians[i] = np.nan
            errors[i] = np.nan
            n_eff[i] = 0
            n_stars[i] = len(vals)
            continue

        # Effective sample size
        neff = np.sum(w_clipped) ** 2 / np.sum(w_clipped ** 2)

        # Weighted phi1 center
        centers[i] = np.average(phi1[in_bin][clip_mask], weights=w_clipped)

        # Bootstrap weighted median
        med, err = bootstrap_weighted_median(vals_clipped, w_clipped, n_bootstrap)

        medians[i] = med
        errors[i] = err
        n_eff[i] = neff
        n_stars[i] = len(vals_clipped)

        if neff < MIN_NEFF:
            valid[i] = False

    return centers, medians, errors, n_eff, n_stars, valid


# ---------------------------------------------------------------------------
# 4. Bin all observables
# ---------------------------------------------------------------------------
print("\nBinning observables (10,000 bootstrap iterations per bin)...")

# Stars outside the gap/spur
df_clean = df[~df.in_gap_spur].copy()
phi1 = df_clean.phi1.values
weights = df_clean.memb_prob.values

results = {}
for obs_name in ["phi2", "pm1", "pm2"]:
    print(f"  Binning {obs_name}...")
    obs = df_clean[obs_name].values

    # Process left and right of gap separately
    mask_left = phi1 < GAP_SPUR_MIN
    mask_right = phi1 >= GAP_SPUR_MAX

    c_l, m_l, e_l, n_l, ns_l, v_l = bin_observable(
        phi1[mask_left], obs[mask_left], weights[mask_left], edges_left
    )
    c_r, m_r, e_r, n_r, ns_r, v_r = bin_observable(
        phi1[mask_right], obs[mask_right], weights[mask_right], edges_right
    )

    results[obs_name] = {
        "phi1": np.concatenate([c_l, c_r]),
        "median": np.concatenate([m_l, m_r]),
        "error": np.concatenate([e_l, e_r]),
        "n_eff": np.concatenate([n_l, n_r]),
        "n_stars": np.concatenate([ns_l, ns_r]),
        "valid": np.concatenate([v_l, v_r]),
    }

# Bin RV separately (only for stars with RV data)
print("  Binning rv (where available)...")
has_rv = df_clean.rv != 0
if has_rv.sum() > 20:
    phi1_rv = phi1[has_rv]
    rv_vals = df_clean.rv.values[has_rv]
    rv_weights = weights[has_rv]

    # Use fewer, wider bins for RV (sparse data)
    n_rv_bins_left = max(3, n_bins_left // 3)
    n_rv_bins_right = max(2, n_bins_right // 3)
    rv_edges_left = np.linspace(PHI1_MIN, GAP_SPUR_MIN, n_rv_bins_left + 1)
    rv_edges_right = np.linspace(GAP_SPUR_MAX, PHI1_MAX, n_rv_bins_right + 1)

    mask_left_rv = phi1_rv < GAP_SPUR_MIN
    mask_right_rv = phi1_rv >= GAP_SPUR_MAX

    c_l, m_l, e_l, n_l, ns_l, v_l = bin_observable(
        phi1_rv[mask_left_rv], rv_vals[mask_left_rv], rv_weights[mask_left_rv],
        rv_edges_left, n_bootstrap=N_BOOTSTRAP
    )
    c_r, m_r, e_r, n_r, ns_r, v_r = bin_observable(
        phi1_rv[mask_right_rv], rv_vals[mask_right_rv], rv_weights[mask_right_rv],
        rv_edges_right, n_bootstrap=N_BOOTSTRAP
    )

    results["rv"] = {
        "phi1": np.concatenate([c_l, c_r]),
        "median": np.concatenate([m_l, m_r]),
        "error": np.concatenate([e_l, e_r]),
        "n_eff": np.concatenate([n_l, n_r]),
        "n_stars": np.concatenate([ns_l, ns_r]),
        "valid": np.concatenate([v_l, v_r]),
    }

# ---------------------------------------------------------------------------
# 5. Build output table (phi2, pm1, pm2 on common phi1 grid)
# ---------------------------------------------------------------------------
# Use phi2 grid as reference (phi1 centers are the same for all 3)
phi1_centers = results["phi2"]["phi1"]
valid_all = results["phi2"]["valid"] & results["pm1"]["valid"] & results["pm2"]["valid"]

track = pd.DataFrame({
    "phi1_deg": phi1_centers[valid_all],
    "phi2_med": results["phi2"]["median"][valid_all],
    "phi2_err": results["phi2"]["error"][valid_all],
    "pm1_med": results["pm1"]["median"][valid_all],
    "pm1_err": results["pm1"]["error"][valid_all],
    "pm2_med": results["pm2"]["median"][valid_all],
    "pm2_err": results["pm2"]["error"][valid_all],
    "n_eff": results["phi2"]["n_eff"][valid_all],
    "n_stars": results["phi2"]["n_stars"][valid_all],
})

# Save RV track separately (different phi1 grid, fewer bins)
if "rv" in results:
    rv_valid = results["rv"]["valid"]
    rv_track = pd.DataFrame({
        "phi1_deg": results["rv"]["phi1"][rv_valid],
        "rv_med": results["rv"]["median"][rv_valid],
        "rv_err": results["rv"]["error"][rv_valid],
        "n_eff": results["rv"]["n_eff"][rv_valid],
        "n_stars": results["rv"]["n_stars"][rv_valid],
    })

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
track_path = os.path.join(DATA_DIR, "gd1_track.csv")
track.to_csv(track_path, index=False, float_format="%.6f")
print(f"\n  Saved track ({len(track)} bins) to {track_path}")

if "rv" in results:
    rv_path = os.path.join(DATA_DIR, "gd1_track_rv.csv")
    rv_track.to_csv(rv_path, index=False, float_format="%.6f")
    print(f"  Saved RV track ({len(rv_track)} bins) to {rv_path}")

# Print summary
print(f"\n=== Track Summary ===")
print(f"  Bins: {len(track)}")
print(f"  phi1 range: [{track.phi1_deg.min():.1f}, {track.phi1_deg.max():.1f}] deg")
print(f"  phi2 error range: [{track.phi2_err.min():.4f}, {track.phi2_err.max():.4f}] deg")
print(f"  pm1 error range: [{track.pm1_err.min():.4f}, {track.pm1_err.max():.4f}] mas/yr")
print(f"  pm2 error range: [{track.pm2_err.min():.4f}, {track.pm2_err.max():.4f}] mas/yr")
print(f"  n_eff range: [{track.n_eff.min():.0f}, {track.n_eff.max():.0f}]")
print(f"  Gap/spur masked: [{GAP_SPUR_MIN}, {GAP_SPUR_MAX}] deg")

# ---------------------------------------------------------------------------
# 7. Diagnostic plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

colors = {"clean": "#2c3e50", "bin": "#e74c3c", "gap": "#f39c12"}

# Individual stars (background)
for ax_idx, (obs, label, unit) in enumerate([
    ("phi2", "phi2", "deg"),
    ("pm1", "pm1 (along stream)", "mas/yr"),
    ("pm2", "pm2 (perp. to stream)", "mas/yr"),
]):
    ax = axes[ax_idx]

    # Plot individual stars
    ax.scatter(df_clean.phi1, df_clean[obs], s=0.5, c="gray", alpha=0.15, rasterized=True)

    # Plot gap/spur region stars
    df_gap = df[df.in_gap_spur]
    ax.scatter(df_gap.phi1, df_gap[obs], s=0.5, c=colors["gap"], alpha=0.2, rasterized=True)

    # Plot binned track
    ax.errorbar(track.phi1_deg, track[f"{obs}_med"], yerr=track[f"{obs}_err"],
                fmt="o", color=colors["bin"], markersize=4, capsize=2,
                elinewidth=1.2, label=f"Binned track ({len(track)} bins)")

    # Shade gap/spur region
    ax.axvspan(GAP_SPUR_MIN, GAP_SPUR_MAX, alpha=0.1, color=colors["gap"],
               label="Gap/spur (masked)")

    ax.set_ylabel(f"{label} ({unit})", fontsize=12)
    if ax_idx == 0:
        ax.legend(fontsize=9, loc="upper right")

# RV panel
ax = axes[3]
if "rv" in results:
    # Individual RV stars
    has_rv_clean = df_clean.rv != 0
    ax.scatter(df_clean.phi1[has_rv_clean], df_clean.rv[has_rv_clean],
               s=8, c="gray", alpha=0.4, rasterized=True)

    # Gap/spur RV stars
    has_rv_gap = df_gap.rv != 0
    if has_rv_gap.sum() > 0:
        ax.scatter(df_gap.phi1[has_rv_gap], df_gap.rv[has_rv_gap],
                   s=8, c=colors["gap"], alpha=0.4, rasterized=True)

    # Binned RV track
    ax.errorbar(rv_track.phi1_deg, rv_track.rv_med, yerr=rv_track.rv_err,
                fmt="s", color=colors["bin"], markersize=5, capsize=2,
                elinewidth=1.2, label=f"Binned RV ({len(rv_track)} bins)")

    ax.axvspan(GAP_SPUR_MIN, GAP_SPUR_MAX, alpha=0.1, color=colors["gap"])
    ax.legend(fontsize=9, loc="upper right")

ax.set_ylabel("Radial velocity (km/s)", fontsize=12)
ax.set_xlabel("phi1 (deg)", fontsize=12)

plt.suptitle(
    "GD-1 Binned Stream Track\n"
    f"2079 cleaned members | {len(track)} astrometric bins | "
    f"{len(rv_track) if 'rv' in results else 0} RV bins | "
    f"gap/spur [{GAP_SPUR_MIN}, {GAP_SPUR_MAX}] masked | "
    f"{N_BOOTSTRAP} bootstrap iterations",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()

plot_path = os.path.join(PLOT_DIR, "gd1_track_binned.png")
plt.savefig(plot_path, dpi=200)
print(f"  Saved track plot to {plot_path}")

# --- Additional plot: error budget and n_eff ---
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

ax = axes2[0]
ax.plot(track.phi1_deg, track.phi2_err * 1000, "o-", ms=3, label="phi2 (mdeg)")
ax.plot(track.phi1_deg, track.pm1_err * 100, "s-", ms=3, label="pm1 (0.01 mas/yr)")
ax.plot(track.phi1_deg, track.pm2_err * 100, "^-", ms=3, label="pm2 (0.01 mas/yr)")
ax.set_ylabel("Error (scaled units)", fontsize=11)
ax.legend(fontsize=9)
ax.set_title("Error budget per bin", fontsize=12)
ax.axvspan(GAP_SPUR_MIN, GAP_SPUR_MAX, alpha=0.1, color=colors["gap"])

ax = axes2[1]
ax.bar(track.phi1_deg, track.n_eff, width=(track.phi1_deg.diff().median() or 1) * 0.8,
       color=colors["clean"], alpha=0.7)
ax.axhline(MIN_NEFF, color="red", ls="--", label=f"Min n_eff = {MIN_NEFF}")
ax.set_xlabel("phi1 (deg)", fontsize=12)
ax.set_ylabel("Effective sample size", fontsize=11)
ax.set_title("Effective sample size per bin", fontsize=12)
ax.legend(fontsize=9)
ax.axvspan(GAP_SPUR_MIN, GAP_SPUR_MAX, alpha=0.1, color=colors["gap"])

plt.tight_layout()
plot_path2 = os.path.join(PLOT_DIR, "gd1_track_errors.png")
plt.savefig(plot_path2, dpi=200)
print(f"  Saved error plot to {plot_path2}")
