# MW Halo Rotation Project

## Overview
Joint Bayesian inference on the Milky Way dark matter halo's figure rotation rate (Omega_p)
using GD-1 stellar stream and rotation curve data.

## Key Parameters
- v_h: halo circular velocity scale (km/s)
- r_h: halo scale radius (kpc)
- q_z: halo vertical flattening
- Omega_p: halo figure rotation rate (km/s/kpc), the novel parameter

## Architecture
- `src/potential/` — unified potential model (TriaxialNFW + SolidBodyRotation)
- `src/likelihood/` — RC and stream likelihood functions
- `src/sampling/` — MCMC sampler configuration
- `src/stream/` — GD-1 orbit integration and track comparison
- `src/utils/` — coordinate transforms, data loaders, diagnostics
- `scripts/` — executable pipeline steps (numbered)
- `notebooks/` — exploration and plotting
- `tests/` — unit tests for potential, likelihood, coordinate transforms

## Conventions
- All potentials use galpy units internally
- Coordinates: Galactocentric frame with R0=8.275 kpc, z_sun=0.0208 kpc
- Solar motion: (U,V,W) = (11.1, 245.04, 7.25) km/s (Gravity Collab + Reid+2020)
- GD-1 frame: Koposov+2010 rotation matrix
- GD-1 anchor RV: -133 km/s (not zero)

## Data Sources
- Rotation curve: Eilers+2019 or Beordo+2024 (Gaia DR3)
- GD-1 catalog: Gaia DR3 with Price-Whelan & Bonaca 2018 selection
- GD-1 RVs: Koposov+2010, Bonaca+2020
