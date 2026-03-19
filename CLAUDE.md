# MW Halo Rotation Project

## Overview
Joint Bayesian inference on the Milky Way dark matter halo's figure rotation rate (Omega_p)
using 4 stellar streams + rotation curve data, with LMC perturbation and tilted triaxial halo.

## Key Parameters (5 free)
- v_h: halo circular velocity scale (km/s)
- r_h: halo scale radius (kpc)
- q_z: halo vertical flattening
- Omega_p: halo figure rotation rate (km/s/kpc), the novel parameter
- sigma_sys: model systematic uncertainty (deg), fitted by the data

## Fixed Parameters
- b = 0.9: in-plane axis ratio (triaxial, from simulations)
- tilt = 18 deg: halo minor axis tilt (Nibauer & Bonaca 2025)
- LMC: M=1.38e11 Msun, a=14.9 kpc Hernquist (Erkal+2019), hardcoded phase-space coords
- Baryons: MWPotential2014 (Bovy 2015), disk normalize=0.6, bulge normalize=0.05

## Architecture
- `src/potential/` — triaxial NFW + rotation + LMC + baryons (galpy)
- `src/likelihood/` — RC, GD-1, Pal 5, Jhelum, Orphan-Chenab likelihoods
- `src/stream/` — mock stream generator (spray method, 100 particles)
- `src/sampling/` — priors + deprecated emcee sampler
- `scripts/09_run_final.py` — definitive dynesty run with mock streams + LMC

## Conventions
- All potentials use galpy natural units (no ro/vo in constructors)
- Coordinates: Galactocentric frame with R0=8.122 kpc, z_sun=0.0208 kpc
  (matches Eilers+2019 RC; Gravity Collaboration 2018)
- Solar peculiar motion: [11.1, 12.24, 7.25] km/s (Schoenrich+2010)
- GD-1 frame: gala GD1Koposov10, anchor phi1=-15.5, d=10kpc, RV=-183.3
- Pal 5 frame: gala Pal5PriceWhelan18, anchor at cluster d=21.9kpc, RV=-58.4
- Jhelum frame: gala JhelumBonaca19, anchor RA=343.2, Dec=-50.8, d=12kpc
- Orphan frame: gala OrphanKoposov19, anchor RA=163, Dec=1.5, d=21kpc

## Data Sources
- Rotation curve: Eilers+2019 (32 points, 5-21 kpc)
- GD-1 catalog: Tavangar & Price-Whelan 2025 (Zenodo 15428120)
- GD-1 RVs: DESI DR1 cross-match (230 stars, 9 bins)
- Pal 5 catalog: Kuzma+2022 (109 spectroscopic members, 7 bins)
- Jhelum: S5 DR1 (Li+2022), cleaned with 5 criteria (250 members, 6 bins)
- Orphan-Chenab: Koposov+2023 curated catalog (360 members, 19+20 bins)
- LMC: hardcoded from Gaia DR3 + Kallivayalil+2013
