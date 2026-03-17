"""
Pal 5 Stream Track Binning
===========================
Bins the Kuzma+2022 Pal 5 kinematic catalog into a stream track
for orbit fitting. Much simpler than GD-1 because:
  - All 109 stars are spectroscopically confirmed members
  - The progenitor cluster is known (phi1 = 0)
  - The catalog is already clean (no membership probability needed)

Input:  data/external/kuzma2022_pal5.txt
Output: data/pal5/pal5_track.csv

Reference:
  Kuzma et al. 2022, MNRAS, 512, 315
"""

import numpy as np
import pandas as pd
import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc
import matplotlib.pyplot as plt
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data", "pal5")
PLOT_DIR = os.path.join(REPO, "results", "plots")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# 1. Load catalog
# -----------------------------------------------------------------------
print("Loading Kuzma+2022 Pal 5 catalog...")
raw = np.loadtxt(os.path.join(REPO, "data", "external", "kuzma2022_pal5.txt"),
                 comments="#", usecols=range(10))

ra = raw[:, 0]
dec = raw[:, 1]
pmra = raw[:, 2]
e_pmra = raw[:, 3]
pmdec = raw[:, 4]
e_pmdec = raw[:, 5]
rv = raw[:, 6]
e_rv = raw[:, 7]

n_stars = len(ra)
print(f"  Total stars: {n_stars}")

# -----------------------------------------------------------------------
# 2. Transform to Pal 5 stream coordinates
# -----------------------------------------------------------------------
sc = coord.SkyCoord(
    ra=ra * u.deg, dec=dec * u.deg,
    pm_ra_cosdec=pmra * u.mas / u.yr,
    pm_dec=pmdec * u.mas / u.yr,
    radial_velocity=rv * u.km / u.s,
    distance=21.9 * u.kpc,  # Baumgardt; same for all (no individual distances)
)

pal5 = sc.transform_to(gc.Pal5PriceWhelan18())
phi1 = pal5.phi1.deg
phi2 = pal5.phi2.deg
pm_phi1_cosphi2 = pal5.pm_phi1_cosphi2.value  # mas/yr
pm_phi2 = pal5.pm_phi2.value  # mas/yr

# Convert pm_phi1_cosphi2 to pm_phi1
cos_phi2 = np.cos(np.radians(phi2))
pm1 = pm_phi1_cosphi2 / cos_phi2

print(f"  phi1 range: [{phi1.min():.1f}, {phi1.max():.1f}] deg")
print(f"  phi2 range: [{phi2.min():.2f}, {phi2.max():.2f}] deg")

# -----------------------------------------------------------------------
# 3. Bin the track
# -----------------------------------------------------------------------
# With only 109 stars over ~20 degrees, use ~10 bins (~10 stars/bin)
N_BINS = 10
edges = np.linspace(phi1.min() - 0.1, phi1.max() + 0.1, N_BINS + 1)

bins = []
for i in range(N_BINS):
    mask = (phi1 >= edges[i]) & (phi1 < edges[i + 1])
    if i == N_BINS - 1:
        mask = (phi1 >= edges[i]) & (phi1 <= edges[i + 1])

    if np.sum(mask) < 3:
        continue

    phi1_c = np.median(phi1[mask])
    phi2_med = np.median(phi2[mask])
    phi2_err = np.std(phi2[mask]) / np.sqrt(np.sum(mask))
    pm1_med = np.median(pm1[mask])
    pm1_err = np.std(pm1[mask]) / np.sqrt(np.sum(mask))
    pm2_med = np.median(pm_phi2[mask])
    pm2_err = np.std(pm_phi2[mask]) / np.sqrt(np.sum(mask))
    rv_med = np.median(rv[mask])
    rv_err = np.std(rv[mask]) / np.sqrt(np.sum(mask))
    n = np.sum(mask)

    bins.append({
        "phi1_deg": phi1_c,
        "phi2_med": phi2_med, "phi2_err": phi2_err,
        "pm1_med": pm1_med, "pm1_err": pm1_err,
        "pm2_med": pm2_med, "pm2_err": pm2_err,
        "rv_med": rv_med, "rv_err": rv_err,
        "n_stars": n,
    })

track = pd.DataFrame(bins)
print(f"\n  Binned track: {len(track)} bins")

# -----------------------------------------------------------------------
# 4. Save
# -----------------------------------------------------------------------
track_path = os.path.join(DATA_DIR, "pal5_track.csv")
track.to_csv(track_path, index=False, float_format="%.6f")
print(f"  Saved to {track_path}")

print(f"\n=== Pal 5 Track Summary ===")
print(f"  Bins: {len(track)}")
print(f"  phi1 range: [{track.phi1_deg.min():.1f}, {track.phi1_deg.max():.1f}] deg")
print(f"  phi2 range: [{track.phi2_med.min():.2f}, {track.phi2_med.max():.2f}] deg")
print(f"  RV range: [{track.rv_med.min():.1f}, {track.rv_med.max():.1f}] km/s")
print(f"  Stars per bin: [{track.n_stars.min()}, {track.n_stars.max()}]")

# -----------------------------------------------------------------------
# 5. Plot
# -----------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

for ax_i, (col, label) in enumerate([
    ("phi2", "phi2 (deg)"), ("pm1", "pm1 (mas/yr)"),
    ("pm2", "pm2 (mas/yr)"), ("rv", "RV (km/s)")
]):
    ax = axes[ax_i]
    ax.scatter(phi1, locals().get(col, rv if col == "rv" else pm_phi2),
               s=8, c="gray", alpha=0.5)
    ax.errorbar(track.phi1_deg, track[f"{col}_med"], yerr=track[f"{col}_err"],
                fmt="o", color="red", ms=5, capsize=3, label=f"Binned ({len(track)} bins)")
    ax.set_ylabel(label)
    if ax_i == 0:
        ax.legend()

axes[-1].set_xlabel("phi1 (deg)")
plt.suptitle(f"Pal 5 Binned Track ({n_stars} stars, Kuzma+2022)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, "pal5_track_binned.png")
plt.savefig(plot_path, dpi=200)
print(f"  Saved plot to {plot_path}")
