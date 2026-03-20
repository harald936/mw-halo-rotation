# MW Halo Figure Rotation

**First observational constraint on the figure rotation rate of the Milky Way's dark matter halo using stellar streams and rotation curve data.**

LCDM simulations predict that triaxial dark matter halos tumble about their minor axis at ~0.1 km/s/kpc (Bailin & Steinmetz 2004; Arora & Valluri 2023). This has never been constrained observationally. We use four stellar streams (GD-1, Pal 5, Jhelum, Orphan-Chenab) spanning ~315 degrees as dynamical probes: the stream tracks on the sky encode the past orientation of the halo potential, which differs from the present if the halo is rotating.

## Method

- **Potential model:** Triaxial NFW halo (b=0.9, tilted 18 deg per Nibauer & Bonaca 2025) + SolidBodyRotation wrapper + fixed MWPotential2014 baryons + LMC perturbation on Orphan-Chenab (Erkal+2019)
- **Data:** Eilers+2019 rotation curve (32 points) + GD-1 (141 bins: phi2+PM+DESI RV) + Pal 5 (28 bins, Kuzma+2022) + Jhelum (12 bins, S5 DR1) + Orphan-Chenab (87 bins: phi2+PM+RV+distance, Koposov+2023). Total: 300 data points across 6 observable channels.
- **Forward model:** Mock stream generation (100 test particles per stream, spray method)
- **Inference:** Dynesty nested sampling with 5 free parameters (v_h, r_h, q_z, Omega_p, sigma_sys) — sigma_sys is the model uncertainty fitted by the data

## Structure

```
src/
  potential/    Triaxial NFW + rotation wrapper + LMC (galpy)
  likelihood/   RC + 4 stream likelihoods (GD-1, Pal 5, Jhelum, Orphan-Chenab)
                + mock-stream joint likelihood
  stream/       Mock stream generator (spray method, 100 particles)
  sampling/     Priors + deprecated emcee sampler
scripts/        Numbered pipeline steps (00-09)
data/           Input data (processed CSVs tracked; large raw files gitignored)
results/        Output chains, plots, tables
paper/          Project plan and manuscript
```

## Pipeline

| Script | Description |
|--------|-------------|
| `00_clean_gd1_catalog.py` | Four-criterion cleaning of GD-1 catalog (2276 -> 2079 members) |
| `01_bin_gd1_track.py` | Bin GD-1 into 44-point track with 10k bootstrap errors |
| `02_validate_potential.py` | Validate composite potential against rotation curve |
| `03_run_rc_proper.py` | RC-only MCMC (validation run) |
| `04_run_joint.py` | Joint MCMC: RC + 4 streams (emcee, deprecated) |
| `05_bin_pal5_track.py` | Bin Pal 5 into 7-point track from Kuzma+2022 |
| `06_crossmatch_desi.py` | Cross-match streams with DESI DR1 for precision RVs |
| `07_add_jhelum_orphan.py` | Process Jhelum + Orphan-Chenab from S5 DR1 / Koposov+2023 |
| `08_run_dynesty.py` | Dynesty nested sampling (4 params, no LMC/mock streams) |
| `09_run_final.py` | **Definitive run:** dynesty, 5 params, mock streams + LMC |
| `10_scan_signed_omega_p.py` | Quick 1D lnL scan vs signed Omega_p (diagnostic) |
| `11_injection_recovery.py` | Injection-recovery test: minimum detectable Omega_p |
| `12_run_final_signed.py` | Signed Omega_p run: U(-0.5, 0.5) prior |

## Parameters

| Parameter | Symbol | Prior | Unit |
|-----------|--------|-------|------|
| Halo velocity scale | v_h | U(100, 300) | km/s |
| Halo scale radius | r_h | U(5, 40) | kpc |
| Vertical flattening | q_z | U(0.5, 2.0) | -- |
| **Figure rotation rate** | **Omega_p** | **U(0, 0.5)** | **km/s/kpc** |
| Model systematic | sigma_sys | U(0.01, 3.0) | deg |

Fixed: in-plane axis ratio b=0.9, halo tilt=18 deg, LMC mass=1.38e11 Msun.

## Setup

```bash
conda env create -f environment.yml
conda activate mw-halo
pip install -e .
```

## Data Sources

- Rotation curve: Eilers et al. 2019, ApJ, 871, 120
- GD-1 catalog: Tavangar & Price-Whelan 2025 (Zenodo 15428120)
- GD-1 RVs: DESI DR1 (NOIRLab DataLab cross-match)
- Pal 5 catalog: Kuzma et al. 2022, MNRAS, 512, 315
- Jhelum: S5 DR1 (Li et al. 2022)
- Orphan-Chenab: Koposov et al. 2023 (sky + PM + RV + RR Lyrae distances, Zenodo 7222654)
- LMC parameters: Erkal et al. 2019, MNRAS, 487, 2685
- Halo tilt: Nibauer & Bonaca 2025

## Key References

- Bailin & Steinmetz (2004) -- simulation prediction: Omega_p ~ 0.15h km/s/kpc
- Arora & Valluri (2023) -- TNG50 figure rotation statistics
- Nibauer & Bonaca (2025) -- tilted halo from GD-1
- Erkal et al. (2019) -- LMC perturbation on stellar streams
- Koposov et al. (2023) -- Orphan-Chenab stream catalog
