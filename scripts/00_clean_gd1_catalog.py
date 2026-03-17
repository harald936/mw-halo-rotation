"""
GD-1 Member Catalog Cleaning Pipeline
======================================
Takes the Tavangar & Price-Whelan (2025) GD-1 member catalog (Gaia DR3)
and applies a four-criterion quality filter to remove contaminants.

Input:  data/gd1/gd1_tbl_with_memb_prob.mrt  (Zenodo: 10.5281/zenodo.15428120)
Output: data/gd1/gd1_members_cleaned.csv

Cleaning criteria:
  1. Parallax foreground rejection: parallax > 0.3 mas at SNR > 3
     (GD-1 is at ~8-10 kpc, so true members have parallax ~ 0.1 mas)
  2. Proper motion track outliers: >3-sigma deviation from smooth polynomial track
  3. Radial velocity track outliers: >3-sigma deviation from smooth polynomial track
  4. Metallicity contaminants: spectroscopic [Fe/H] > -1.5
     (GD-1 has [Fe/H] ~ -2.3; field stars are typically > -1.0)

Reference:
  Base catalog: Tavangar & Price-Whelan 2025, arXiv:2502.13236
  Gaia DR3: Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1
"""

import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data", "gd1")
PLOT_DIR = os.path.join(REPO, "results", "plots")

# ---------------------------------------------------------------------------
# 1. Load catalog
# ---------------------------------------------------------------------------
print("Loading Tavangar & Price-Whelan 2025 catalog...")
t = ascii.read(os.path.join(DATA_DIR, "gd1_tbl_with_memb_prob.mrt"), format="cds")

# Select members with p > 0.5
mask_member = t["memb_prob"] > 0.5
members = t[mask_member]
n_initial = len(members)
print(f"  Members with p > 0.5: {n_initial}")

# ---------------------------------------------------------------------------
# 2. Flag contaminants
# ---------------------------------------------------------------------------
flags = np.zeros(n_initial, dtype=int)
flag_reasons = np.full(n_initial, "", dtype="U64")

# --- 2a. Parallax foreground rejection ---
plx = np.array(members["parallax"], dtype=float)
plx_err = np.array(members["e_parallax"], dtype=float)
plx_snr = plx / plx_err
is_foreground = (plx > 0.3) & (plx_snr > 3)
flags[is_foreground] |= 1
n_plx = np.sum(is_foreground)
print(f"  Parallax foreground (plx > 0.3 mas, SNR > 3): {n_plx}")

# --- 2b. Proper motion track outliers ---
phi1 = np.array(members["phi1"], dtype=float)
pm1 = np.array(members["pm1"], dtype=float)
pm2 = np.array(members["pm2"], dtype=float)

# Fit smooth polynomial tracks (3rd order) using only non-foreground stars
clean_for_fit = ~is_foreground
coef_pm1 = np.polyfit(phi1[clean_for_fit], pm1[clean_for_fit], 3)
coef_pm2 = np.polyfit(phi1[clean_for_fit], pm2[clean_for_fit], 3)

resid_pm1 = pm1 - np.polyval(coef_pm1, phi1)
resid_pm2 = pm2 - np.polyval(coef_pm2, phi1)
std_pm1 = np.std(resid_pm1[clean_for_fit])
std_pm2 = np.std(resid_pm2[clean_for_fit])

is_pm_outlier = (np.abs(resid_pm1) > 3 * std_pm1) | (np.abs(resid_pm2) > 3 * std_pm2)
flags[is_pm_outlier] |= 2
n_pm = np.sum(is_pm_outlier & ~is_foreground)  # unique to this criterion
print(f"  PM track outliers (3-sigma): {np.sum(is_pm_outlier)} ({n_pm} new)")

# --- 2c. Radial velocity track outliers ---
rv = np.array(members["rv"], dtype=float)
e_rv = np.array(members["e_rv"], dtype=float)
has_rv = e_rv < 1000  # sentinel: e_rv=10000 means no measurement
is_rv_outlier = np.zeros(n_initial, dtype=bool)
if np.sum(has_rv) > 10:
    # Fit RV track using only clean stars (exclude foreground + PM outliers)
    clean_rv_mask = has_rv & ~is_foreground & ~is_pm_outlier
    coef_rv = np.polyfit(phi1[clean_rv_mask], rv[clean_rv_mask], 3)
    resid_rv = rv[has_rv] - np.polyval(coef_rv, phi1[has_rv])
    std_rv = np.std(resid_rv)

    is_rv_outlier_sub = np.abs(resid_rv) > 3 * std_rv
    is_rv_outlier[has_rv] = is_rv_outlier_sub
    flags[is_rv_outlier] |= 4
    n_rv = np.sum(is_rv_outlier & ~is_foreground & ~is_pm_outlier)
    print(f"  RV track outliers (3-sigma): {np.sum(is_rv_outlier)} ({n_rv} new)")

# --- 2d. Metallicity contaminants ---
feh = np.array(members["feh"], dtype=float)
has_feh = feh > -5  # sentinel: feh=-10000 means no measurement
is_metal_rich = np.zeros(n_initial, dtype=bool)
is_metal_rich[has_feh] = feh[has_feh] > -1.5
flags[is_metal_rich] |= 8
n_feh = np.sum(is_metal_rich & ~is_foreground & ~is_pm_outlier & ~is_rv_outlier)
print(f"  Metallicity contaminants ([Fe/H] > -1.5): {np.sum(is_metal_rich)} ({n_feh} new)")

# ---------------------------------------------------------------------------
# 3. Apply filters
# ---------------------------------------------------------------------------
is_clean = flags == 0
n_removed = np.sum(~is_clean)
n_clean = np.sum(is_clean)
print(f"\n  Total contaminants removed: {n_removed} ({n_removed/n_initial*100:.1f}%)")
print(f"  Clean members remaining: {n_clean}")

# ---------------------------------------------------------------------------
# 4. Save cleaned catalog
# ---------------------------------------------------------------------------
clean = members[is_clean]

df = pd.DataFrame({
    "source_id": np.array(clean["source_id"]),
    "ra": np.array(clean["ra"], dtype=float),
    "dec": np.array(clean["dec"], dtype=float),
    "parallax": np.array(clean["parallax"], dtype=float),
    "e_parallax": np.array(clean["e_parallax"], dtype=float),
    "pmra": np.array(clean["pmra"], dtype=float),
    "pmdec": np.array(clean["pmdec"], dtype=float),
    "phi1": np.array(clean["phi1"], dtype=float),
    "phi2": np.array(clean["phi2"], dtype=float),
    "pm1": np.array(clean["pm1"], dtype=float),
    "pm2": np.array(clean["pm2"], dtype=float),
    "e_pm1": np.array(clean["e_pm1"], dtype=float),
    "e_pm2": np.array(clean["e_pm2"], dtype=float),
    "g0": np.array(clean["g0"], dtype=float),
    "r0": np.array(clean["r0"], dtype=float),
    "i0": np.array(clean["i0"], dtype=float),
    "rv": np.array(clean["rv"], dtype=float),
    "e_rv": np.array(clean["e_rv"], dtype=float),
    "feh": np.array(clean["feh"], dtype=float),
    "memb_prob": np.array(clean["memb_prob"], dtype=float),
})

outpath = os.path.join(DATA_DIR, "gd1_members_cleaned.csv")
df.to_csv(outpath, index=False)
print(f"\n  Saved cleaned catalog to {outpath}")

# ---------------------------------------------------------------------------
# 5. Summary statistics
# ---------------------------------------------------------------------------
has_rv_clean = df["e_rv"] < 1000
has_feh_clean = df["feh"] > -5
print(f"\n=== Cleaned Catalog Summary ===")
print(f"  Total members: {len(df)}")
print(f"  phi1 range: [{df.phi1.min():.1f}, {df.phi1.max():.1f}] deg")
print(f"  phi2 range: [{df.phi2.min():.1f}, {df.phi2.max():.1f}] deg")
print(f"  With RV: {has_rv_clean.sum()}")
print(f"  With [Fe/H]: {has_feh_clean.sum()}")
print(f"  Median memb_prob: {df.memb_prob.median():.3f}")

# ---------------------------------------------------------------------------
# 6. Diagnostic plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Panel 1: Sky track (phi1 vs phi2) — before and after
ax = axes[0, 0]
ax.scatter(members["phi1"][~is_clean], members["phi2"][~is_clean],
           s=3, c="red", alpha=0.5, label=f"Removed ({n_removed})")
ax.scatter(df.phi1, df.phi2, s=1, c="k", alpha=0.3, label=f"Clean ({n_clean})")
ax.set_xlabel("phi1 (deg)")
ax.set_ylabel("phi2 (deg)")
ax.set_title("Sky track")
ax.legend(fontsize=8, markerscale=3)
ax.set_ylim(-5, 5)

# Panel 2: PM1 track
ax = axes[0, 1]
ax.scatter(members["phi1"][~is_clean], members["pm1"][~is_clean],
           s=3, c="red", alpha=0.5)
ax.scatter(df.phi1, df.pm1, s=1, c="k", alpha=0.3)
phi1_grid = np.linspace(-90, 20, 200)
ax.plot(phi1_grid, np.polyval(coef_pm1, phi1_grid), "b-", lw=1, label="Poly fit")
ax.fill_between(phi1_grid,
                np.polyval(coef_pm1, phi1_grid) - 3 * std_pm1,
                np.polyval(coef_pm1, phi1_grid) + 3 * std_pm1,
                alpha=0.15, color="blue", label="3-sigma")
ax.set_xlabel("phi1 (deg)")
ax.set_ylabel("pm1 (mas/yr)")
ax.set_title("PM along stream")
ax.legend(fontsize=8)

# Panel 3: PM2 track
ax = axes[0, 2]
ax.scatter(members["phi1"][~is_clean], members["pm2"][~is_clean],
           s=3, c="red", alpha=0.5)
ax.scatter(df.phi1, df.pm2, s=1, c="k", alpha=0.3)
ax.plot(phi1_grid, np.polyval(coef_pm2, phi1_grid), "b-", lw=1)
ax.fill_between(phi1_grid,
                np.polyval(coef_pm2, phi1_grid) - 3 * std_pm2,
                np.polyval(coef_pm2, phi1_grid) + 3 * std_pm2,
                alpha=0.15, color="blue")
ax.set_xlabel("phi1 (deg)")
ax.set_ylabel("pm2 (mas/yr)")
ax.set_title("PM perpendicular to stream")

# Panel 4: RV track
ax = axes[1, 0]
if np.sum(has_rv) > 10:
    rv_clean_mask = df["e_rv"] < 1000
    rv_removed = members[~is_clean]
    rv_removed_mask = np.array(rv_removed["e_rv"], dtype=float) < 1000
    if np.sum(rv_removed_mask) > 0:
        ax.scatter(np.array(rv_removed["phi1"], dtype=float)[rv_removed_mask],
                   np.array(rv_removed["rv"], dtype=float)[rv_removed_mask],
                   s=15, c="red", alpha=0.7, zorder=3)
    ax.scatter(df.phi1[rv_clean_mask], df.rv[rv_clean_mask], s=8, c="k", alpha=0.5)
    ax.plot(phi1_grid, np.polyval(coef_rv, phi1_grid), "b-", lw=1)
    ax.fill_between(phi1_grid,
                    np.polyval(coef_rv, phi1_grid) - 3 * std_rv,
                    np.polyval(coef_rv, phi1_grid) + 3 * std_rv,
                    alpha=0.15, color="blue")
ax.set_xlabel("phi1 (deg)")
ax.set_ylabel("RV (km/s)")
ax.set_title("Radial velocity track")

# Panel 5: Parallax distribution
ax = axes[1, 1]
ax.hist(np.array(members["parallax"][is_clean], dtype=float), bins=50,
        range=(-0.5, 1.5), color="k", alpha=0.5, label="Clean")
ax.hist(np.array(members["parallax"][is_foreground], dtype=float), bins=50,
        range=(-0.5, 1.5), color="red", alpha=0.5, label="Foreground")
ax.axvline(0.3, color="blue", ls="--", label="Cut (0.3 mas)")
ax.axvline(0.1, color="green", ls=":", label="Expected GD-1 (~0.1 mas)")
ax.set_xlabel("Parallax (mas)")
ax.set_ylabel("Count")
ax.set_title("Parallax distribution")
ax.legend(fontsize=8)

# Panel 6: Metallicity
ax = axes[1, 2]
feh_clean = df.feh[df.feh > -5]
feh_removed_arr = np.array(members["feh"][~is_clean], dtype=float)
feh_removed_vals = feh_removed_arr[feh_removed_arr > -5]
if len(feh_clean) > 0:
    ax.hist(feh_clean, bins=30, range=(-4, 0), color="k", alpha=0.5, label="Clean")
if len(feh_removed_vals) > 0:
    ax.hist(feh_removed_vals, bins=30, range=(-4, 0), color="red", alpha=0.5, label="Removed")
ax.axvline(-1.5, color="blue", ls="--", label="Cut (-1.5)")
ax.axvline(-2.3, color="green", ls=":", label="GD-1 mean (-2.3)")
ax.set_xlabel("[Fe/H]")
ax.set_ylabel("Count")
ax.set_title("Metallicity distribution")
ax.legend(fontsize=8)

plt.suptitle(
    f"GD-1 Catalog Cleaning: {n_initial} -> {n_clean} members "
    f"({n_removed} removed, {n_removed/n_initial*100:.1f}%)",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
plotpath = os.path.join(PLOT_DIR, "gd1_cleaning_diagnostics.png")
plt.savefig(plotpath, dpi=150)
print(f"  Saved diagnostic plot to {plotpath}")
