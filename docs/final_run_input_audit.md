# Final Run Input Audit

Generated: 2026-03-21 10:17 UTC

This audit checks the files actually used by the signed final run and classifies each input as:

- `PASS`: exactly or near-exactly reproduced in this audit
- `SOURCE_BACKED`: tied to an official public source, but not fully rebuilt here
- `WARNING`: a real reproducibility caveat remains

## Final Input Counts

| Dataset | Points |
|---|---:|
| GD-1 | 144 |
| Pal 5 | 31 |
| Jhelum | 24 |
| Orphan-Chenab | 86 |
| Rotation curve | 32 |
| **Total** | **317** |

## Audit Results

- **Final run script**: `PASS`. Found signed final-run script with 200 particles and the fast LMC builder.
- **GD-1 RV track**: `PASS`. Exact local rebuild from gd1_members_cleaned.csv matches stored gd1_track_rv.csv.
- **Pal 5 kinematic track**: `PASS`. Exact local rebuild from Kuzma+2022 catalog matches stored pal5_track.csv.
- **Pal 5 distance track**: `PASS`. Official CDS RR Lyrae list matches the vetted local list and reproduces stored pal5_dist_track.csv.
- **Jhelum sky+RV track**: `PASS`. Exact local rebuild from S5 DR1 and adopted cuts matches stored jhelum_track.csv sky/RV columns.
- **Jhelum PM track**: `WARNING`. Current scripted Gaia DR3 rebuild does not match stored PM columns (max abs diff = 0.1046).
- **Orphan sky track**: `PASS`. Stored orphan_track.csv sky columns match published orphan_M_track_bins.fits.
- **Orphan PM1 track**: `PASS`. Stored orphan pm1 track matches published orphan_pm1_bins.fits.
- **Orphan PM2 track**: `PASS`. Stored orphan pm2 track matches published orphan_pm2_bins.fits.
- **Orphan RV track**: `PASS`. Stored orphan_rv_track.csv matches published orphan_rv_bins.fits.
- **Orphan distance track**: `SOURCE_BACKED`. Stored orphan_dist_track.csv is source-backed by the Koposov+2023 Zenodo release, but orphan_dmrr_bins.fits is not present locally in this clone so exact rebuild was not run here.
- **Rotation curve**: `PASS`. Eilers+2019 file loads cleanly with 32 points and expected columns.

## Verdict

- `PASS`: 10
- `SOURCE_BACKED`: 1
- `WARNING`: 1

- The strongest remaining caveat is the Jhelum PM augmentation: the current scripted Gaia DR3 rebuild does not exactly reproduce the stored PM bins.
- The recommended next production script remains `scripts/16_run_final_signed_lmcfast.py`.
