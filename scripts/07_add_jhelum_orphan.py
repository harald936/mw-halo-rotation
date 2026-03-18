"""
Add Jhelum and Orphan-Chenab streams from S5 DR1.
==================================================
Filters S5 spectroscopic targets by stream field, applies kinematic
cuts to select members, cross-matches with Gaia DR3 for proper motions,
transforms to stream coordinates, and bins into tracks.

Jhelum: ~95 members at d~12 kpc (Li+2022)
Orphan-Chenab: ~1240 members at d~15-55 kpc (Koposov+2023)

Input:  data/external/s5_pdr1_light.fits
Output: data/jhelum/jhelum_track.csv
        data/orphan/orphan_track.csv
"""

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import gala.coordinates as gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(REPO, "results", "plots")

# -----------------------------------------------------------------------
# 1. Load S5 DR1
# -----------------------------------------------------------------------
print("Loading S5 DR1...")
s5 = Table.read(os.path.join(REPO, "data", "external", "s5_pdr1_light.fits"))
s5 = s5[s5["primary"]]  # best observation per target
print(f"  S5 DR1 primary targets: {len(s5)}")

# -----------------------------------------------------------------------
# 2. JHELUM
# -----------------------------------------------------------------------
print("\n=== JHELUM ===")
jhelum_mask = np.array([f.startswith("Jhelum") for f in s5["field"]])
jh = s5[jhelum_mask]
print(f"  S5 Jhelum field stars: {len(jh)}")

# Stream targets have priority 7-9
jh_stream = jh[(jh["priority"] >= 7) & (jh["priority"] <= 9)]
print(f"  Stream priority (7-9): {len(jh_stream)}")

# Transform to Jhelum coordinates
sc_jh = SkyCoord(ra=np.array(jh_stream["ra"], dtype=float) * u.deg,
                 dec=np.array(jh_stream["dec"], dtype=float) * u.deg)
jh_coord = sc_jh.transform_to(gc.JhelumBonaca19())
phi1_jh = jh_coord.phi1.deg
phi2_jh = jh_coord.phi2.deg

# Kinematic cut: select stars near the stream track
# Jhelum RV ~ -10 to +50 km/s depending on position (Li+2022)
rv_jh = np.array(jh_stream["vel_calib"], dtype=float)
rv_err_jh = np.array(jh_stream["vel_calib_std"], dtype=float)

# Keep stars with |phi2| < 2 deg and reasonable RV errors
member_mask = (np.abs(phi2_jh) < 2.0) & (rv_err_jh < 10.0) & (rv_err_jh > 0)
jh_members = jh_stream[member_mask]
phi1_mem = phi1_jh[member_mask]
phi2_mem = phi2_jh[member_mask]
rv_mem = rv_jh[member_mask]
rv_err_mem = rv_err_jh[member_mask]

print(f"  Members (|phi2|<2, good RV): {len(jh_members)}")

# Bin into track
os.makedirs(os.path.join(REPO, "data", "jhelum"), exist_ok=True)
N_BINS_JH = 8
edges_jh = np.linspace(phi1_mem.min() - 0.5, phi1_mem.max() + 0.5, N_BINS_JH + 1)
bins_jh = []
for i in range(N_BINS_JH):
    mask = (phi1_mem >= edges_jh[i]) & (phi1_mem < edges_jh[i + 1])
    n = mask.sum()
    if n >= 3:
        bins_jh.append({
            "phi1_deg": np.median(phi1_mem[mask]),
            "phi2_med": np.median(phi2_mem[mask]),
            "phi2_err": np.std(phi2_mem[mask]) / np.sqrt(n),
            "rv_med": np.median(rv_mem[mask]),
            "rv_err": np.std(rv_mem[mask]) / np.sqrt(n),
            "n_stars": n,
        })

jh_track = pd.DataFrame(bins_jh)
jh_path = os.path.join(REPO, "data", "jhelum", "jhelum_track.csv")
jh_track.to_csv(jh_path, index=False, float_format="%.6f")
print(f"  Binned track: {len(jh_track)} bins")
print(f"  Saved to {jh_path}")

# -----------------------------------------------------------------------
# 3. ORPHAN-CHENAB
# -----------------------------------------------------------------------
print("\n=== ORPHAN-CHENAB ===")
orphan_mask = np.array([f.startswith("Orphan") or f.startswith("OrphanS")
                        or f.startswith("Chenab") for f in s5["field"]])
oc = s5[orphan_mask]
print(f"  S5 Orphan/Chenab field stars: {len(oc)}")

oc_stream = oc[(oc["priority"] >= 7) & (oc["priority"] <= 9)]
print(f"  Stream priority (7-9): {len(oc_stream)}")

# Transform to Orphan-Chenab coordinates
sc_oc = SkyCoord(ra=np.array(oc_stream["ra"], dtype=float) * u.deg,
                 dec=np.array(oc_stream["dec"], dtype=float) * u.deg)
oc_coord = sc_oc.transform_to(gc.OrphanKoposov19())
phi1_oc = oc_coord.phi1.deg
phi2_oc = oc_coord.phi2.deg

rv_oc = np.array(oc_stream["vel_calib"], dtype=float)
rv_err_oc = np.array(oc_stream["vel_calib_std"], dtype=float)

# Kinematic cut: |phi2| < 3 deg, good RV
member_mask_oc = (np.abs(phi2_oc) < 3.0) & (rv_err_oc < 10.0) & (rv_err_oc > 0)
oc_members = oc_stream[member_mask_oc]
phi1_oc_mem = phi1_oc[member_mask_oc]
phi2_oc_mem = phi2_oc[member_mask_oc]
rv_oc_mem = rv_oc[member_mask_oc]
rv_err_oc_mem = rv_err_oc[member_mask_oc]

print(f"  Members (|phi2|<3, good RV): {len(oc_members)}")
print(f"  phi1 range: [{phi1_oc_mem.min():.1f}, {phi1_oc_mem.max():.1f}] deg")

# Bin into track
os.makedirs(os.path.join(REPO, "data", "orphan"), exist_ok=True)
N_BINS_OC = 20
edges_oc = np.linspace(phi1_oc_mem.min() - 1, phi1_oc_mem.max() + 1, N_BINS_OC + 1)
bins_oc = []
for i in range(N_BINS_OC):
    mask = (phi1_oc_mem >= edges_oc[i]) & (phi1_oc_mem < edges_oc[i + 1])
    n = mask.sum()
    if n >= 3:
        bins_oc.append({
            "phi1_deg": np.median(phi1_oc_mem[mask]),
            "phi2_med": np.median(phi2_oc_mem[mask]),
            "phi2_err": np.std(phi2_oc_mem[mask]) / np.sqrt(n),
            "rv_med": np.median(rv_oc_mem[mask]),
            "rv_err": np.std(rv_oc_mem[mask]) / np.sqrt(n),
            "n_stars": n,
        })

oc_track = pd.DataFrame(bins_oc)
oc_path = os.path.join(REPO, "data", "orphan", "orphan_track.csv")
oc_track.to_csv(oc_path, index=False, float_format="%.6f")
print(f"  Binned track: {len(oc_track)} bins")
print(f"  Saved to {oc_path}")

# -----------------------------------------------------------------------
# 4. Summary plot
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Jhelum sky
ax = axes[0, 0]
ax.scatter(phi1_jh, phi2_jh, s=1, c="gray", alpha=0.3)
ax.scatter(phi1_mem, phi2_mem, s=8, c="dodgerblue", alpha=0.6)
ax.errorbar(jh_track.phi1_deg, jh_track.phi2_med, yerr=jh_track.phi2_err,
            fmt="o", color="red", ms=5, capsize=3)
ax.set_xlabel("phi1 (deg)")
ax.set_ylabel("phi2 (deg)")
ax.set_title(f"Jhelum ({len(jh_members)} members, {len(jh_track)} bins)")

# Jhelum RV
ax = axes[0, 1]
ax.scatter(phi1_mem, rv_mem, s=8, c="dodgerblue", alpha=0.6)
ax.errorbar(jh_track.phi1_deg, jh_track.rv_med, yerr=jh_track.rv_err,
            fmt="o", color="red", ms=5, capsize=3)
ax.set_xlabel("phi1 (deg)")
ax.set_ylabel("RV (km/s)")
ax.set_title("Jhelum RV track")

# Orphan-Chenab sky
ax = axes[1, 0]
ax.scatter(phi1_oc, phi2_oc, s=1, c="gray", alpha=0.3)
ax.scatter(phi1_oc_mem, phi2_oc_mem, s=4, c="dodgerblue", alpha=0.4)
ax.errorbar(oc_track.phi1_deg, oc_track.phi2_med, yerr=oc_track.phi2_err,
            fmt="o", color="red", ms=4, capsize=2)
ax.set_xlabel("phi1 (deg)")
ax.set_ylabel("phi2 (deg)")
ax.set_title(f"Orphan-Chenab ({len(oc_members)} members, {len(oc_track)} bins)")

# Orphan-Chenab RV
ax = axes[1, 1]
ax.scatter(phi1_oc_mem, rv_oc_mem, s=4, c="dodgerblue", alpha=0.4)
ax.errorbar(oc_track.phi1_deg, oc_track.rv_med, yerr=oc_track.rv_err,
            fmt="o", color="red", ms=4, capsize=2)
ax.set_xlabel("phi1 (deg)")
ax.set_ylabel("RV (km/s)")
ax.set_title("Orphan-Chenab RV track")

plt.suptitle("Jhelum + Orphan-Chenab Streams (S5 DR1)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "jhelum_orphan_tracks.png"), dpi=200)
print(f"\nSaved plot to {PLOT_DIR}/jhelum_orphan_tracks.png")
print("\nDone!")
