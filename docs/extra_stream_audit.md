# Extra Stream Audit for Omega_p Paper

## Decision: Do NOT add a 5th stream

The current 4-stream set (GD-1, Pal 5, Jhelum, Orphan-Chenab) with 314 data points
across 6 observable channels is already the strongest dataset assembled for an Omega_p
constraint. No candidate 5th stream adds sufficient leverage to justify the risk.

## Current stream ranking (Omega_p sensitivity)

1. **Orphan-Chenab** (86 pts) — PRIMARY PROBE. Longest stream (160 deg), distance
   track (17-77 kpc from RR Lyrae), 3 Gyr lookback, LMC modeled. Carries most of
   the Omega_p constraint.
2. **GD-1** (144 pts) — Most data points, best PM coverage. Constrains halo mass/shape
   that makes Omega_p measurable. But only 1.2 Gyr lookback limits rotation signal.
3. **Pal 5** (28 pts) — Known progenitor, 3 Gyr lookback, probes high-z region.
   Complements GD-1 for q_z. Limited by small sample (109 members).
4. **Jhelum** (24 pts) — Weakest. Now has Gaia PMs but still only 6 bins. Narrow/broad
   component issue (Awad+2023). Kept for geometric coverage (southern sky).

## 5th stream candidates evaluated

### ATLAS-Aliqa Uma (BEST CANDIDATE — still not worth adding)
- d~20 kpc, ~30 deg, S5 RVs + Gaia PM available
- Li+2020: ATLAS and Aliqa Uma are one stream
- Geometrically similar to GD-1 (similar distance, similar halo region)
- Would add ~20-30 data points but mostly redundant information
- **Reject:** marginal gain, adds implementation complexity

### Indus
- d~16 kpc, ~20 deg, S5 RVs + Gaia PM
- Yang+2026: epicyclic density variations
- Similar distance to GD-1/Jhelum — redundant
- **Reject:** no new Omega_p leverage

### Phoenix
- d~18 kpc, ~5 deg — too short for Omega_p sensitivity
- **Reject:** too short

### Tucana III
- d~25 kpc, has S5 RVs
- Close LMC passage (~15 kpc) — massive perturbation
- Would require LMC modeling like Orphan, for a tiny ~3 deg stream
- **Reject:** LMC-contaminated, too short

### Fimbulthul (Omega Cen stream)
- d~5 kpc, inner halo, associated with Omega Centauri
- Hot stream (large velocity dispersion), massive progenitor
- **Reject:** too close, too hot, complex progenitor

## Recommended upgrades instead

1. Run signed Omega_p scan (script 10, ready, 30 min)
2. Run injection-recovery (script 11, ready, 1-2 hrs)
3. Run signed dynesty if scan shows signal (script 12, ready, ~15 hrs)
4. Add Pal 5 RR Lyrae distances from Price-Whelan+2019
5. Bayes factor comparison (static vs rotating halo)
