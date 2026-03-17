# MW Halo Figure Rotation

**First observational constraint on the figure rotation rate of the Milky Way's dark matter halo from GD-1 and rotation curve data.**

LCDM simulations predict that triaxial dark matter halos tumble about their minor axis at ~0.1 km/s/kpc. This has never been constrained observationally. We use the GD-1 stellar stream as a dynamical probe: the stream track on the sky encodes the past orientation of the halo potential, which differs from the present if the halo is rotating.

## Structure

```
src/
  potential/    Triaxial NFW + SolidBodyRotation wrapper (galpy)
  likelihood/   RC and stream chi-squared likelihoods
  sampling/     emcee sampler, priors, convergence diagnostics
  stream/       GD-1 orbit integration and track comparison
  utils/        Coordinate transforms, data loaders
scripts/        Numbered pipeline steps (01_, 02_, ...)
notebooks/      Exploration and plotting
data/           Input data (processed CSVs tracked; large raw files gitignored)
results/        Output chains, plots, tables
paper/          Project plan and manuscript
tests/          Unit tests
```

## Setup

```bash
conda env create -f environment.yml
conda activate mw-halo
pip install -e .
```

## Parameters

| Parameter | Symbol | Prior | Unit |
|-----------|--------|-------|------|
| Halo velocity scale | v_h | U(150, 300) | km/s |
| Halo scale radius | r_h | U(5, 40) | kpc |
| Vertical flattening | q_z | U(0.5, 1.5) | — |
| **Figure rotation rate** | **Omega_p** | **U(0, 0.5)** | **km/s/kpc** |

## Key Reference

Bailin & Steinmetz (2004) predict Omega_p ~ 0.15h km/s/kpc from N-body simulations.
