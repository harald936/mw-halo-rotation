# The First Observational Constraint on the Figure Rotation Rate of the Milky Way's Dark Matter Halo from GD-1 and Rotation Curve Data

## Project Plan

---

## Phase 0: Data Acquisition & Validation

### Step 0a — Rotation Curve Data
- Download Eilers+2019 or Beordo+2024 circular velocity measurements (R, V_circ, sigma)
- Validate: plot V(R), check for outliers, verify error bars are symmetric or handle asymmetry
- Store as `data/rotation_curve/rc_data.csv` with columns: `R_kpc, Vcirc_kms, eVcirc_kms`
- Cross-check against independent RC compilations (Huang+2020, Mroz+2019 Cepheids)

### Step 0b — GD-1 Stream Catalog
- Query Gaia DR3 for GD-1 region using Price-Whelan & Bonaca (2018) footprint
- Apply quality cuts: RUWE < 1.4, finite astrometry, CMD matched filter
- Transform to GD-1 stream coordinates (phi1, phi2) using Koposov+2010 rotation matrix
- Assign membership probabilities via Gaussian Mixture Model (stream + background)
- Validate against Bonaca+2020 published member list — compare overlap fraction
- Store as `data/gd1/gd1_members.csv`

### Step 0c — GD-1 Radial Velocities
- Compile RV measurements from Koposov+2010 and Bonaca+2020
- Cross-match with Gaia DR3 RV catalog (RVS)
- Anchor point RV: use literature value (~-133 km/s), NOT zero
- Store as `data/gd1/gd1_rvs.csv`

### Step 0d — GD-1 Binned Track
- Bin members (p > 0.6) into 40-60 bins in phi1
- Compute weighted median + bootstrap error for: phi2, pm_phi1, pm_phi2
- Require n_eff >= 20 per bin
- Store as `data/gd1/gd1_track.csv`

---

## Phase 1: Unified Potential Model

### Step 1a — Baryonic Components (Fixed)
- Disk: Miyamoto-Nagai potential with parameters from McMillan (2017)
  - M_d = 5.43e10 Msun, a = 6.5 kpc, b = 0.26 kpc (thin) + thick disk component
- Bulge: Hernquist profile
  - M_b = 0.91e10 Msun, a = 0.07 kpc (compact)
- Implement in `src/potential/baryons.py` using galpy
- Validate: plot individual V_circ contributions, compare to published decompositions

### Step 1b — Triaxial NFW Halo
- Use galpy's `TriaxialNFWPotential(a=r_h, amp=v_h, c=q_z, b=1.0)`
  - Free parameters: v_h (normalisation), r_h (scale radius), q_z (vertical axis ratio)
  - Fix q_x = q_y = 1.0 (axisymmetric in-plane for now)
- This replaces the old inconsistency where RC used gNFW and orbits used Logarithmic
- Implement in `src/potential/halo.py`
- Validate: check V_circ(R) matches analytic NFW at q_z=1

### Step 1c — Figure Rotation Wrapper
- Wrap halo in `SolidBodyRotationWrapperPotential(pot=halo, omega=Omega_p, pa=0)`
  - Omega_p in km/s/kpc (the new free parameter)
  - pa = initial position angle of halo major axis (fixed or marginalised)
- At Omega_p = 0, this reduces exactly to the static triaxial case
- Implement in `src/potential/rotating.py`
- Validate: integrate circular orbit, confirm energy is NOT conserved (Jacobi integral IS)

### Step 1d — Composite Potential Builder
- Function `build_potential(v_h, r_h, q_z, Omega_p)` returns full MW potential
- Baryonic components fixed, halo parameters free
- Implement in `src/potential/composite.py`
- Validate: plot total V_circ(R) for fiducial parameters, compare to observed RC

---

## Phase 2: Likelihood Functions

### Step 2a — Rotation Curve Likelihood
- Forward model: compute V_circ(R) from composite potential at each observed R
- For the rotating potential, V_circ is evaluated at t=0 (present-day orientation)
  - Note: RC is measured today, so we use the current halo orientation
- Likelihood: Gaussian chi-squared
  - `ln L_RC = -0.5 * sum[(V_obs - V_model)^2 / sigma^2]`
- Handle asymmetric errors by symmetrising: sigma = (sigma_plus + sigma_minus) / 2
- Implement in `src/likelihood/rotation_curve.py`
- Validate: recover known parameters from mock RC data

### Step 2b — GD-1 Stream Likelihood
- Forward model:
  1. Set initial conditions from GD-1 anchor point (phi1_0, phi2_0, d_0, pm1_0, pm2_0, RV_0)
     - Use literature values: d ~ 10 kpc, RV ~ -133 km/s
  2. Transform anchor to Galactocentric (x, v)
  3. Integrate orbit BACKWARD for 1.2 Gyr in the rotating potential
     - The rotating wrapper automatically handles the changing halo orientation
  4. Sample orbit at observed phi1 values to get model (phi2, pm1, pm2) track
- Likelihood: chi-squared on track residuals
  - `ln L_stream = -0.5 * sum[(obs_i - model_i)^2 / sigma_i^2]`
  - Summed over phi2, pm_phi1, pm_phi2 channels
  - Profile out a global offset per channel (nuisance parameter for anchor uncertainty)
- Implement in `src/likelihood/stream.py`
- Validate: check that fiducial parameters produce a track resembling observed GD-1

### Step 2c — Joint Likelihood
- `ln L_joint = ln L_RC + ln L_stream`
- Assumes RC and stream data are independent (they are — different tracers)
- Implement in `src/likelihood/joint.py`

---

## Phase 3: Priors

### Step 3a — Parameter Priors
- v_h: Uniform(150, 300) km/s — covers all reasonable MW halo masses
- r_h: Uniform(5, 40) kpc — spans literature range
- q_z: Uniform(0.5, 1.5) — oblate to prolate
- Omega_p: Uniform(0, 0.5) km/s/kpc — from zero to well above simulation predictions
  - Simulation prediction: Omega_p ~ 0.15h ~ 0.10 km/s/kpc (Bailin & Steinmetz 2004)
  - Upper bound 0.5 km/s/kpc covers the full range generously
- Implement in `src/sampling/priors.py`

---

## Phase 4: MCMC Sampling

### Step 4a — Sampler Setup
- Use emcee with 128 walkers, 5000 steps, 1000 burn-in
- Moves: StretchMove (70%) + DEMove (30%) for multimodal exploration
- Parallel: multiprocessing pool (all available cores)
- Implement in `src/sampling/sampler.py`

### Step 4b — Initialisation Strategy
- Start walkers in a tight ball around MAP estimate from Phase 2 validation
- Alternatively: run short optimisation (scipy.minimize) first to find MAP
- For the bimodality concern: run multiple independent chains from dispersed starting points

### Step 4c — Convergence Diagnostics
- Compute autocorrelation time (tau) using emcee's built-in
- Require chain length > 50*tau for all parameters
- Compute Gelman-Rubin R-hat across independent chains (target R-hat < 1.01)
- Report effective sample size (ESS) for each parameter
- Implement diagnostics in `src/sampling/diagnostics.py`

### Step 4d — Run Configurations
- **Run A: RC-only** — fit (v_h, r_h) with q_z=1, Omega_p=0 to validate RC model
- **Run B: RC + GD-1, static** — fit (v_h, r_h, q_z) with Omega_p=0 to reproduce old results
- **Run C: RC + GD-1, rotating** — fit (v_h, r_h, q_z, Omega_p) — the main result
- Scripts: `scripts/01_run_rc_only.py`, `scripts/02_run_joint_static.py`, `scripts/03_run_joint_rotating.py`

---

## Phase 5: Post-Processing & Results

### Step 5a — Chain Analysis
- Thin chains by autocorrelation time
- Produce corner plots for all parameter combinations
- Compute marginal posteriors: median, 68% CI, 95% CI for each parameter
- Save summary tables to `results/tables/`

### Step 5b — Omega_p Constraint (The Main Result)
- Extract 1D marginal posterior on Omega_p
- If peaked away from zero: report median + 68% CI as a detection
- If monotonically decreasing from zero: report 95% upper limit
- Compare to Bailin & Steinmetz (2004) prediction: Omega_p ~ 0.15h km/s/kpc
  - Overlay simulation prediction distribution on posterior plot

### Step 5c — Goodness of Fit
- Posterior predictive checks:
  - Draw 100 samples from posterior, compute model RC and stream track for each
  - Plot data with 68%/95% model bands
- Compute chi-squared per degree of freedom for best-fit model
- Check residuals for structure (run test, autocorrelation)

### Step 5d — Sensitivity Tests
- Fix Omega_p = 0 vs free: does q_z constraint change? (tests degeneracy)
- Vary disk/bulge parameters within literature uncertainties
- Vary GD-1 anchor distance (9, 10, 11 kpc) and RV (-120, -133, -145 km/s)
- Test with Eilers+2019 RC vs Beordo+2024 RC

---

## Phase 6: Figures

### Figure 1 — Data Overview
- Panel A: Rotation curve data with best-fit model and component decomposition
- Panel B: GD-1 stream on sky (phi1 vs phi2) with membership coloring
- Panel C: GD-1 proper motion track (pm_phi1 vs phi1, pm_phi2 vs phi1)

### Figure 2 — RC-Only Fit
- Best-fit V_circ(R) with 68% band
- Residuals panel below
- Component decomposition (disk, bulge, halo)

### Figure 3 — Stream Track Comparison
- GD-1 track for different Omega_p values (0, 0.1, 0.2, 0.3 km/s/kpc)
- Show how rotation shifts the track — this is the observable signature

### Figure 4 — Corner Plot
- Full 4D posterior: (v_h, r_h, q_z, Omega_p)
- Highlight degeneracies between parameters

### Figure 5 — The Money Plot
- 1D marginal posterior on Omega_p
- Vertical band: Bailin & Steinmetz 2004 simulation prediction
- 95% upper limit or detection marked
- This is the headline result

### Figure 6 — Posterior Predictive Checks
- RC data vs model band
- Stream track data vs model band
- Demonstrates the fit quality

---

## Phase 7: Paper Writing

### Sections
1. **Introduction** — LCDM predicts triaxial, tumbling halos; never observed; GD-1 as a probe
2. **Data** — RC source, GD-1 selection, membership, binning
3. **Method** — Potential model, likelihood functions, MCMC setup
4. **Results** — RC-only fit, joint static fit, joint rotating fit, Omega_p constraint
5. **Discussion** — Comparison to simulations, caveats, future work (Pal 5, Sagittarius)
6. **Conclusions** — First constraint on MW halo tumbling rate

---

## Execution Order & Dependencies

```
Phase 0 (data)         ← do first, everything depends on this
  ↓
Phase 1 (potential)    ← needs data for validation
  ↓
Phase 2 (likelihood)   ← needs potential + data
  ↓
Phase 3 (priors)       ← independent, quick
  ↓
Phase 4a-c (sampler)   ← needs likelihood + priors
  ↓
Phase 4d Run A (RC-only)      ← validates pipeline
  ↓
Phase 4d Run B (joint static) ← reproduces old results
  ↓
Phase 4d Run C (joint rotating) ← THE main run
  ↓
Phase 5 (results)      ← needs chains
  ↓
Phase 6 (figures)      ← needs results
  ↓
Phase 7 (paper)        ← needs everything
```
