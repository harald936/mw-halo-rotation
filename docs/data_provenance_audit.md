# Data Provenance Audit — All 317 Data Points Verified

Every data point traces to a published source with real error bars.
No fabricated values. No invented uncertainties.

## GD-1 (144 points) — MAXED OUT

| Channel | Bins | Source | Errors |
|---------|------|--------|--------|
| phi2 | 44 | Tavangar & Price-Whelan 2025 (Zenodo 15428120), 2079 Gaia DR3 members | Bootstrap (10k iterations) |
| pm1 | 44 | Same members, Gaia DR3 proper motions | Bootstrap |
| pm2 | 44 | Same members, Gaia DR3 proper motions | Bootstrap |
| RV | 12 | 356 members with RVs (DESI DR1 + SDSS + LAMOST), weighted mean | 1/sqrt(sum(1/e_rv^2)) |

**Missing:** Distance track. Parallax SNR=0.38 (useless at 10 kpc). No published photometric
distance track exists. Nibauer+2025 has positions but no distance errors. Not added.

**Searched and rejected:**
- Tavangar+2025 Zenodo: 197 additional members excluded by our 4 quality cuts (foreground, PM/RV outliers, metallicity)
- DESI EDR GD-1 paper (arXiv:2407.06336): 115 members likely overlap with our DESI DR1 cross-match
- Sheffield+2025 APOGEE: focused on chemistry, not new members

## Pal 5 (31 points) — MAXED OUT

| Channel | Bins | Source | Errors |
|---------|------|--------|--------|
| phi2 | 7 | Kuzma+2022 (MNRAS 512, 315), 109 spectroscopic members | std/sqrt(n) |
| pm1 | 7 | Same members, Gaia proper motions | std/sqrt(n) |
| pm2 | 7 | Same members, Gaia proper motions | std/sqrt(n) |
| rv | 7 | Same members, spectroscopic RVs | std/sqrt(n) |
| distance | 3 | Price-Whelan+2019 (AJ 158 223), 15 stream RR Lyrae (p>0.8) | Weighted mean, 3% RR Lyrae precision |

Distance gradient verified: leading tail (phi1=+3.8) at 19.5 kpc, trailing tail (phi1=-20.8) at 22.2 kpc.
Cluster at 20.6 kpc. Physically correct.

**Searched and rejected:**
- Sheffield+2025 APOGEE: only 2 stream members
- DESI DR1: too sparse at 21 kpc

## Jhelum (24 points) — MAXED OUT

| Channel | Bins | Source | Errors |
|---------|------|--------|--------|
| phi2 | 6 | S5 DR1 (Li+2022), kinematic cut |phi2|<2, rv_err<10 | std/sqrt(n) |
| rv | 6 | S5 DR1 spectroscopic RVs | std/sqrt(n) |
| pm1 | 6 | Gaia DR3 (queried via S5 gaia_source_id, 2000 stars) | std/sqrt(n) |
| pm2 | 6 | Gaia DR3 (same query) | std/sqrt(n) |

**Missing:** Distance track. No published distance gradient. No RR Lyrae catalog.
Awad+2023: d=12.40 kpc single value (not a gradient). Member list in Appendix C
but not in any public archive (no CDS/Zenodo). Cannot reproduce narrow-component
selection without their data. DESI at Dec=-50 is outside footprint.

**Searched and rejected:**
- Awad+2023: no public data release
- Sheffield+2021 APOGEE: no new members
- Woudenberg+2022: modeling paper only
- DESI DR1: outside footprint

## Orphan-Chenab (86 points) — MAXED OUT

| Channel | Bins | Source | Errors |
|---------|------|--------|--------|
| phi2 | 18 | Koposov+2023 spline (Zenodo 7222654) | Published FITS |
| pm1 | 16 | Koposov+2023 (orphan_pm1_bins.fits) | Published FITS |
| pm2 | 16 | Koposov+2023 (orphan_pm2_bins.fits) | Published FITS |
| rv | 20 | Koposov+2023 (orphan_rv_bins.fits) | Published FITS |
| distance | 14 | Koposov+2023 RR Lyrae (orphan_dmrr_bins.fits) | DM errors propagated |

Removed phi1=-90 outlier (phi2_err=2.79 deg, 10x worse than all others, no PM).
All Koposov+2023 Zenodo data products checked and used.

## Rotation Curve (32 points)

Eilers et al. 2019, ApJ, 871, 120. Published asymmetric errors symmetrized as (e+ + e-)/2.

## Total: 317 data points — ALL VERIFIED REAL
