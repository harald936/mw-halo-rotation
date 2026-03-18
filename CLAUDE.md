# MW Halo Rotation Project

## Overview
Joint Bayesian inference on the Milky Way dark matter halo's figure rotation rate (Omega_p)
using GD-1 + Pal 5 stellar streams and rotation curve data, with LMC perturbation and tilted halo.

## Key Parameters
- v_h: halo circular velocity scale (km/s)
- r_h: halo scale radius (kpc)
- q_z: halo vertical flattening
- Omega_p: halo figure rotation rate (km/s/kpc), the novel parameter

## Fixed Parameters
- b = 0.9: in-plane axis ratio (triaxial, from simulations)
- tilt = 18 deg: halo minor axis tilt (Nibauer & Bonaca 2025)
- LMC: M=1.38e11 Msun, a=14.9 kpc Hernquist (Erkal+2019)
- Baryons: MWPotential2014 (Bovy 2015), disk normalize=0.6, bulge normalize=0.05

## Architecture
- `src/potential/` — triaxial NFW + rotation + LMC + baryons (galpy)
- `src/likelihood/` — RC, GD-1 stream, Pal 5 stream likelihoods
- `src/sampling/` — emcee sampler, priors
- `scripts/` — numbered pipeline steps (00-06)

## Conventions
- All potentials use galpy natural units (no ro/vo in constructors)
- Coordinates: Galactocentric frame with R0=8.122 kpc, z_sun=0.0208 kpc
  (matches Eilers+2019 RC; Gravity Collaboration 2018)
- Solar motion: (U,V,W) = (11.1, 241.24, 7.25) km/s (Schoenrich+2010 + v0=229)
- GD-1 frame: gala GD1Koposov10
- Pal 5 frame: gala Pal5PriceWhelan18
- GD-1 anchor: phi1=-15.5, d=10kpc, RV=-183.3 km/s
- Pal 5 anchor: cluster at RA=229.022, d=21.9kpc, RV=-58.4 km/s

## Data Sources
- Rotation curve: Eilers+2019 (32 points, 5-21 kpc)
- GD-1 catalog: Tavangar & Price-Whelan 2025 (Zenodo 15428120)
- GD-1 RVs: DESI DR1 cross-match (234 matches, median err 3.3 km/s)
- Pal 5 catalog: Kuzma+2022 (109 kinematic members)
- LMC orbit: galpy Orbit.from_name('LMC') + ChandrasekharDynamicalFrictionForce
