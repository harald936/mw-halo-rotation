# MW Halo Figure Rotation

**First observational constraint on the figure rotation rate of the Milky Way's dark matter halo using stellar streams and rotation curve data.**

LCDM simulations predict that triaxial dark matter halos tumble about their minor axis at ~0.1 km/s/kpc (Bailin & Steinmetz 2004; Arora & Valluri 2023). This has never been constrained observationally. We use four stellar streams (GD-1, Pal 5, Jhelum, Orphan-Chenab) spanning ~315 degrees as dynamical probes: the stream tracks on the sky encode the past orientation of the halo potential, which differs from the present if the halo is rotating.

## Method

- **Potential model:** Triaxial NFW halo (b=0.9, tilted 18 deg per Nibauer & Bonaca 2025) + SolidBodyRotation wrapper + fixed MWPotential2014 baryons + LMC perturbation on Orphan-Chenab (Erkal+2019)
- **Data:** Eilers+2019 rotation curve (32 points) + GD-1 (44 bins + 9 DESI RV bins) + Pal 5 (7 bins, Kuzma+2022) + Jhelum (6 bins, S5 DR1) + Orphan-Chenab (19+20 bins, Koposov+2023)
- **Forward model:** Mock stream generation (200 test particles per stream, spray method) or single-orbit integration
- **Inference:** Dynesty nested sampling with 5 free parameters (v_h, r_h, q_z, Omega_p, sigma_sys) — sigma_sys is the model uncertainty fitted by the data

## Structure

```
src/
  potential/    Triaxial NFW + rotation wrapper + LMC (galpy)
  likelihood/   RC, GD-1 stream, and Pal 5 stream likelihoods
  sampling/     emcee sampler, priors
scripts/        Numbered pipeline steps (00-06)
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
| `04_run_joint.py` | Joint MCMC: RC + GD-1 + Pal 5 + LMC (main science run) |
| `05_bin_pal5_track.py` | Bin Pal 5 into 7-point track from Kuzma+2022 |
| `06_crossmatch_desi.py` | Cross-match streams with DESI DR1 for precision RVs |

## Parameters

| Parameter | Symbol | Prior | Unit |
|-----------|--------|-------|------|
| Halo velocity scale | v_h | U(100, 300) | km/s |
| Halo scale radius | r_h | U(5, 40) | kpc |
| Vertical flattening | q_z | U(0.5, 2.0) | -- |
| **Figure rotation rate** | **Omega_p** | **U(0, 0.5)** | **km/s/kpc** |

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
- LMC parameters: Erkal et al. 2019, MNRAS, 487, 2685
- Halo tilt: Nibauer & Bonaca 2025

## Key References

- Bailin & Steinmetz (2004) -- simulation prediction: Omega_p ~ 0.15h km/s/kpc
- Arora & Valluri (2023) -- TNG50 figure rotation statistics
- Nibauer & Bonaca (2025) -- tilted halo from GD-1
- Erkal et al. (2021) -- detecting figure rotation with tidal streams
