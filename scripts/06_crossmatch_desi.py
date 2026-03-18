"""
Cross-match GD-1 and Pal 5 members with DESI DR1 stellar RVs.
=============================================================
Queries NOIRLab's DataLab for DESI DR1 stellar spectra in the
GD-1 and Pal 5 sky regions, then cross-matches by position
(1 arcsec radius) with our cleaned member catalogs.

This adds precision RVs (~1-2 km/s) from DESI for stream members
that Gaia couldn't measure (too faint for Gaia RVS).

Data source: DESI DR1 MWS (Milky Way Survey)
Access: NOIRLab Astro Data Lab SQL interface
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DATA_DIR = os.path.join(REPO, "data")

# -----------------------------------------------------------------------
# 1. Query DESI DR1 for stellar RVs in GD-1 region
# -----------------------------------------------------------------------
print("Querying DESI DR1 via NOIRLab DataLab...")

try:
    from dl import queryClient as qc

    # GD-1 region: RA ~ 120-250, Dec ~ 20-65
    query_gd1 = """
    SELECT mean_fiber_ra AS target_ra, mean_fiber_dec AS target_dec,
           z AS redshift, zerr, spectype, subtype, zwarn, deltachi2, targetid
    FROM desi_dr1.zpix
    WHERE mean_fiber_ra BETWEEN 120 AND 250
      AND mean_fiber_dec BETWEEN 20 AND 65
      AND spectype = 'STAR'
      AND zwarn = 0
      AND zcat_primary = true
    """

    print("  Querying GD-1 region...")
    result_gd1 = qc.query(sql=query_gd1, fmt='pandas')
    print(f"  DESI stars in GD-1 region: {len(result_gd1)}")

    # Pal 5 region: RA ~ 223-241, Dec ~ -6 to 7
    query_pal5 = """
    SELECT mean_fiber_ra AS target_ra, mean_fiber_dec AS target_dec,
           z AS redshift, zerr, spectype, subtype, zwarn, deltachi2, targetid
    FROM desi_dr1.zpix
    WHERE mean_fiber_ra BETWEEN 222 AND 242
      AND mean_fiber_dec BETWEEN -7 AND 7
      AND spectype = 'STAR'
      AND zwarn = 0
      AND zcat_primary = true
    """

    print("  Querying Pal 5 region...")
    result_pal5 = qc.query(sql=query_pal5, fmt='pandas')
    print(f"  DESI stars in Pal 5 region: {len(result_pal5)}")

except Exception as e:
    print(f"  DataLab query failed: {e}")
    print("  Falling back to local cross-match only...")
    result_gd1 = None
    result_pal5 = None

# -----------------------------------------------------------------------
# 2. Convert DESI redshifts to radial velocities
# -----------------------------------------------------------------------
C_KMS = 299792.458  # speed of light in km/s

if result_gd1 is not None and len(result_gd1) > 0:
    result_gd1["rv_kms"] = result_gd1["redshift"].astype(float) * C_KMS
    result_gd1["rv_err_kms"] = result_gd1["zerr"].astype(float) * C_KMS
    print(f"\n  GD-1 region DESI RVs: min={result_gd1.rv_kms.min():.0f}, "
          f"max={result_gd1.rv_kms.max():.0f} km/s")

if result_pal5 is not None and len(result_pal5) > 0:
    result_pal5["rv_kms"] = result_pal5["redshift"].astype(float) * C_KMS
    result_pal5["rv_err_kms"] = result_pal5["zerr"].astype(float) * C_KMS
    print(f"  Pal 5 region DESI RVs: min={result_pal5.rv_kms.min():.0f}, "
          f"max={result_pal5.rv_kms.max():.0f} km/s")

# -----------------------------------------------------------------------
# 3. Cross-match with GD-1 cleaned catalog
# -----------------------------------------------------------------------
print("\nCross-matching with GD-1 members...")
gd1 = pd.read_csv(os.path.join(DATA_DIR, "gd1", "gd1_members_cleaned.csv"))

if result_gd1 is not None and len(result_gd1) > 0:
    cat_gd1 = SkyCoord(ra=gd1.ra.values * u.deg, dec=gd1.dec.values * u.deg)
    cat_desi = SkyCoord(
        ra=result_gd1.target_ra.astype(float).values * u.deg,
        dec=result_gd1.target_dec.astype(float).values * u.deg,
    )

    # Match within 1 arcsec
    idx, sep, _ = cat_gd1.match_to_catalog_sky(cat_desi)
    matched = sep < 1.0 * u.arcsec

    n_match = matched.sum()
    print(f"  GD-1 matches (< 1 arcsec): {n_match} / {len(gd1)}")

    if n_match > 0:
        gd1_matched = gd1[matched].copy()
        desi_matched = result_gd1.iloc[idx[matched]].reset_index(drop=True)
        gd1_matched["desi_rv"] = desi_matched["rv_kms"].values
        gd1_matched["desi_rv_err"] = desi_matched["rv_err_kms"].values
        gd1_matched["desi_targetid"] = desi_matched["targetid"].values

        # Save
        out_path = os.path.join(DATA_DIR, "gd1", "gd1_desi_rv.csv")
        gd1_matched.to_csv(out_path, index=False)
        print(f"  Saved to {out_path}")

        # Compare to existing RVs where available
        has_old_rv = gd1_matched.e_rv < 1000
        if has_old_rv.sum() > 0:
            print(f"\n  Stars with both old + DESI RVs: {has_old_rv.sum()}")
            old_rv = gd1_matched.rv[has_old_rv].values
            new_rv = gd1_matched.desi_rv[has_old_rv].values
            diff = old_rv - new_rv
            print(f"  Mean RV difference (old - DESI): {np.mean(diff):.2f} km/s")
            print(f"  Std RV difference: {np.std(diff):.2f} km/s")

        # Summary
        print(f"\n  === GD-1 DESI RV Summary ===")
        print(f"  New RVs from DESI: {n_match}")
        print(f"  RV range: [{gd1_matched.desi_rv.min():.1f}, {gd1_matched.desi_rv.max():.1f}] km/s")
        print(f"  Median RV error: {gd1_matched.desi_rv_err.median():.2f} km/s")
else:
    print("  No DESI data available for cross-match")

# -----------------------------------------------------------------------
# 4. Cross-match with Pal 5 catalog
# -----------------------------------------------------------------------
print("\nCross-matching with Pal 5 members...")
pal5_raw = np.loadtxt(os.path.join(DATA_DIR, "external", "kuzma2022_pal5.txt"),
                      comments="#", usecols=range(10))

if result_pal5 is not None and len(result_pal5) > 0:
    cat_pal5 = SkyCoord(ra=pal5_raw[:, 0] * u.deg, dec=pal5_raw[:, 1] * u.deg)
    cat_desi_p5 = SkyCoord(
        ra=result_pal5.target_ra.astype(float).values * u.deg,
        dec=result_pal5.target_dec.astype(float).values * u.deg,
    )

    idx_p5, sep_p5, _ = cat_pal5.match_to_catalog_sky(cat_desi_p5)
    matched_p5 = sep_p5 < 1.0 * u.arcsec

    n_match_p5 = matched_p5.sum()
    print(f"  Pal 5 matches (< 1 arcsec): {n_match_p5} / {len(pal5_raw)}")

    if n_match_p5 > 0:
        desi_p5 = result_pal5.iloc[idx_p5[matched_p5]].reset_index(drop=True)
        print(f"  DESI RV range: [{desi_p5.rv_kms.min():.1f}, {desi_p5.rv_kms.max():.1f}] km/s")
        print(f"  Median RV error: {desi_p5.rv_err_kms.median():.2f} km/s")

        # Compare to Kuzma RVs
        old_rv_p5 = pal5_raw[matched_p5, 6]
        new_rv_p5 = desi_p5.rv_kms.values
        diff_p5 = old_rv_p5 - new_rv_p5
        print(f"  Mean RV difference (Kuzma - DESI): {np.mean(diff_p5):.2f} km/s")
        print(f"  Std RV difference: {np.std(diff_p5):.2f} km/s")
else:
    print("  No DESI data available for cross-match")

print("\nDone!")
