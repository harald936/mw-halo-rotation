"""
Injection-recovery test for Omega_p.
====================================
Determines the minimum |Omega_p| recoverable with current data quality.

For each injected Omega_p:
1. Generate synthetic mock stream tracks at that Omega_p
2. Add Gaussian noise using real measurement errors
3. Recover Omega_p via 1D grid scan against the noisy synthetic data
"""
import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.potential.composite import build_potential
from src.potential.lmc import build_lmc_potential
from src.likelihood.rotation_curve import ln_likelihood_rc, compute_model_vcirc, _RC_R, _RC_V, _RC_SIGMA
from src.likelihood.stream_mock import (
    mock_stream_likelihood_single, _extract_mock_particles,
    _interp_track, _ALL_TRACKS, STREAMS, SYS_PM, SYS_RV,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------
# Fixed params (from dynesty or fallback)
# -----------------------------------------------------------------------
results_file = os.path.join(REPO, "results", "dynesty_final.npz")
if os.path.exists(results_file):
    print("Loading best-fit from dynesty_final.npz...")
    data = np.load(results_file)
    samples = data['samples']
    logwt = data['logwt']
    logz = data['logz']
    weights = np.exp(logwt - logz[-1])
    weights /= weights.sum()
    def wmedian(x, w):
        idx = np.argsort(x)
        cs = np.cumsum(w[idx])
        return x[idx][np.searchsorted(cs, 0.5)]
    v_h = wmedian(samples[:, 0], weights)
    r_h = wmedian(samples[:, 1], weights)
    q_z = wmedian(samples[:, 2], weights)
    sigma_sys = wmedian(samples[:, 4], weights)
else:
    v_h, r_h, q_z, sigma_sys = 160.0, 16.0, 0.93, 0.3

print(f"Fixed params: v_h={v_h:.1f}, r_h={r_h:.1f}, q_z={q_z:.3f}, sigma_sys={sigma_sys:.3f}")

INJECT_VALUES = [-0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20]
RECOVERY_GRID = np.linspace(-0.5, 0.5, 41)


def generate_synthetic_tracks(omega_inj):
    """Generate synthetic observables at a given Omega_p.

    Returns dict of {stream_name: {phi1, phi2, pm1, pm2, rv}} mock tracks,
    plus synthetic RC velocities.
    """
    pot = build_potential(v_h, r_h, q_z, omega_inj, include_lmc=False)

    # Synthetic RC
    V_synth = compute_model_vcirc(pot, _RC_R)

    # Synthetic stream tracks
    tracks = {}
    for name in STREAMS:
        if name == 'orphan':
            lmc_pot, _ = build_lmc_potential(pot)
            pot_use = pot + [lmc_pot]
        else:
            pot_use = pot

        result = _extract_mock_particles(pot_use, name)
        if result is None:
            tracks[name] = None
            continue

        phi1s, phi2s, pm1s, pm2s, rvs, dists_kpc = result
        tracks[name] = {'phi1': phi1s, 'phi2': phi2s, 'pm1': pm1s, 'pm2': pm2s, 'rv': rvs, 'dist': dists_kpc}

    return V_synth, tracks


def compute_synthetic_lnL(omega_eval, V_synth_noisy, synth_tracks_noisy):
    """Evaluate lnL of a trial Omega_p against synthetic noisy data.

    Uses the same chi-squared structure as the real likelihood but
    compares model predictions against synthetic (not real) data.
    """
    pot = build_potential(v_h, r_h, q_z, omega_eval, include_lmc=False)

    # RC contribution against synthetic RC
    V_model = compute_model_vcirc(pot, _RC_R)
    chi2_rc = np.sum(((V_synth_noisy - V_model) / _RC_SIGMA) ** 2)
    lnL = -0.5 * chi2_rc

    # Stream contributions
    for name in STREAMS:
        if name == 'orphan':
            try:
                lmc_pot, _ = build_lmc_potential(pot)
                pot_use = pot + [lmc_pot]
            except (RuntimeError, ValueError):
                return -1e10
        else:
            pot_use = pot

        try:
            result = _extract_mock_particles(pot_use, name)
        except (RuntimeError, ValueError):
            return -1e10

        if result is None:
            return -1e10

        phi1s_model, phi2s_model, pm1s_model, pm2s_model, rvs_model, dists_model = result
        synth = synth_tracks_noisy.get(name)
        if synth is None:
            return -1e10

        data_track = _ALL_TRACKS[name]['track']
        rv_track = _ALL_TRACKS[name]['rv_track']
        phi1_data = data_track['phi1_deg'].values

        chi2 = 0.0

        # phi2
        phi2_mod, v = _interp_track(phi1s_model, phi2s_model, phi1_data)
        if v.sum() >= 3:
            sigma2 = data_track['phi2_err'].values[v] ** 2 + sigma_sys ** 2
            chi2 += np.sum((synth['phi2_interp'][v] - phi2_mod[v]) ** 2 / sigma2)

        # PM
        if 'pm1_med' in data_track.columns:
            pm1_mod, vpm1 = _interp_track(phi1s_model, pm1s_model, phi1_data)
            valid_pm1 = v & vpm1
            if valid_pm1.sum() >= 2:
                sigma_pm1 = np.sqrt(data_track['pm1_err'].values[valid_pm1] ** 2 + SYS_PM ** 2)
                chi2 += np.sum((synth['pm1_interp'][valid_pm1] - pm1_mod[valid_pm1]) ** 2 / sigma_pm1 ** 2)

        if 'pm2_med' in data_track.columns:
            pm2_mod, vpm2 = _interp_track(phi1s_model, pm2s_model, phi1_data)
            valid_pm2 = v & vpm2
            if valid_pm2.sum() >= 2:
                sigma_pm2 = np.sqrt(data_track['pm2_err'].values[valid_pm2] ** 2 + SYS_PM ** 2)
                chi2 += np.sum((synth['pm2_interp'][valid_pm2] - pm2_mod[valid_pm2]) ** 2 / sigma_pm2 ** 2)

        # RV
        if 'rv_med' in data_track.columns:
            rv_mod, vrv = _interp_track(phi1s_model, rvs_model, phi1_data)
            valid_rv = v & vrv
            if valid_rv.sum() >= 2:
                sigma2_rv = data_track['rv_err'].values[valid_rv] ** 2 + SYS_RV ** 2
                chi2 += np.sum((synth['rv_interp'][valid_rv] - rv_mod[valid_rv]) ** 2 / sigma2_rv)

        if rv_track is not None:
            phi1_rv = rv_track['phi1_deg'].values
            rv_mod2, vrv2 = _interp_track(phi1s_model, rvs_model, phi1_rv)
            if vrv2.sum() >= 2:
                sigma2_rv2 = rv_track['rv_err'].values[vrv2] ** 2 + SYS_RV ** 2
                chi2 += np.sum((synth['rv_interp2'][vrv2] - rv_mod2[vrv2]) ** 2 / sigma2_rv2)

        lnL += -0.5 * chi2

    return lnL


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
print(f"\nInjecting {len(INJECT_VALUES)} values, recovering on {len(RECOVERY_GRID)}-point grid\n")

results = []
for inj_omega in INJECT_VALUES:
    print(f"Injecting Omega_p = {inj_omega:+.3f}...")

    # Step 1: generate synthetic observables at injected Omega_p
    try:
        V_synth, synth_tracks = generate_synthetic_tracks(inj_omega)
    except (RuntimeError, ValueError) as e:
        print(f"  FAILED to generate synthetic data: {e}")
        results.append({'injected': inj_omega, 'recovered': np.nan, 'bias': np.nan})
        continue

    # Step 2: add noise using real error bars
    rng = np.random.RandomState(seed=12345)
    V_synth_noisy = V_synth + rng.randn(len(V_synth)) * _RC_SIGMA

    synth_noisy = {}
    for name in STREAMS:
        if synth_tracks[name] is None:
            synth_noisy[name] = None
            continue

        st = synth_tracks[name]
        data_track = _ALL_TRACKS[name]['track']
        rv_track = _ALL_TRACKS[name]['rv_track']
        phi1_data = data_track['phi1_deg'].values

        # Interpolate synthetic track at data positions
        phi2_interp, _ = _interp_track(st['phi1'], st['phi2'], phi1_data)
        pm1_interp, _ = _interp_track(st['phi1'], st['pm1'], phi1_data)
        pm2_interp, _ = _interp_track(st['phi1'], st['pm2'], phi1_data)
        rv_interp, _ = _interp_track(st['phi1'], st['rv'], phi1_data)

        # Add noise
        phi2_noisy = phi2_interp + rng.randn(len(phi1_data)) * data_track['phi2_err'].values
        pm1_noisy = pm1_interp.copy()
        pm2_noisy = pm2_interp.copy()
        if 'pm1_err' in data_track.columns:
            pm1_noisy += rng.randn(len(phi1_data)) * data_track['pm1_err'].values
        if 'pm2_err' in data_track.columns:
            pm2_noisy += rng.randn(len(phi1_data)) * data_track['pm2_err'].values
        rv_noisy = rv_interp + rng.randn(len(phi1_data)) * (data_track['rv_err'].values if 'rv_err' in data_track.columns else 5.0)

        entry = {
            'phi2_interp': phi2_noisy,
            'pm1_interp': pm1_noisy,
            'pm2_interp': pm2_noisy,
            'rv_interp': rv_noisy,
        }

        # Separate RV track
        if rv_track is not None:
            phi1_rv = rv_track['phi1_deg'].values
            rv_interp2, _ = _interp_track(st['phi1'], st['rv'], phi1_rv)
            rv_noisy2 = rv_interp2 + rng.randn(len(phi1_rv)) * rv_track['rv_err'].values
            entry['rv_interp2'] = rv_noisy2
        else:
            entry['rv_interp2'] = np.array([])

        synth_noisy[name] = entry

    # Step 3: recover Omega_p on grid
    lnL_grid = []
    for omega_r in RECOVERY_GRID:
        lnL = compute_synthetic_lnL(omega_r, V_synth_noisy, synth_noisy)
        lnL_grid.append(lnL)
    lnL_grid = np.array(lnL_grid)

    valid = lnL_grid > -1e9
    if valid.sum() < 3:
        print(f"  WARNING: most grid points failed")
        best_omega = np.nan
    else:
        best_omega = RECOVERY_GRID[valid][np.argmax(lnL_grid[valid])]

    bias = best_omega - inj_omega if np.isfinite(best_omega) else np.nan
    results.append({'injected': inj_omega, 'recovered': best_omega, 'bias': bias})
    print(f"  Recovered: {best_omega:+.3f}, Bias: {bias:+.3f}")

df = pd.DataFrame(results)

# Save
tables_dir = os.path.join(REPO, "results", "tables")
os.makedirs(tables_dir, exist_ok=True)
csv_path = os.path.join(tables_dir, "omega_injection_recovery.csv")
df.to_csv(csv_path, index=False, float_format="%.4f")
print(f"\nSaved to {csv_path}")

# Analysis
print(f"\n{'='*55}")
print(f"INJECTION-RECOVERY RESULTS")
print(f"{'='*55}")
print(f"{'Injected':>10s} {'Recovered':>10s} {'Bias':>10s}")
for _, r in df.iterrows():
    print(f"{r.injected:+10.3f} {r.recovered:+10.3f} {r.bias:+10.3f}")

recoverable = []
for _, r in df.iterrows():
    if r.injected != 0 and np.isfinite(r.bias) and abs(r.bias) < 0.5 * abs(r.injected):
        recoverable.append(abs(r.injected))

if recoverable:
    min_recoverable = min(recoverable)
    print(f"\nMinimum recoverable |Omega_p|: ~{min_recoverable:.2f} km/s/kpc")
else:
    print(f"\nNo injected values were cleanly recovered.")
    print(f"Current data may not support a signed measurement.")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([-0.3, 0.3], [-0.3, 0.3], 'k--', lw=1, alpha=0.5, label='Perfect recovery')
valid_mask = df.recovered.notna()
ax.scatter(df.injected[valid_mask], df.recovered[valid_mask], s=100, color='steelblue',
           edgecolors='black', zorder=3)
for _, r in df[valid_mask].iterrows():
    ax.annotate(f"bias={r.bias:+.3f}", (r.injected, r.recovered),
                xytext=(8, -12), textcoords='offset points', fontsize=8)
ax.set_xlabel(r'Injected $\Omega_p$ (km/s/kpc)', fontsize=12)
ax.set_ylabel(r'Recovered $\Omega_p$ (km/s/kpc)', fontsize=12)
ax.set_title('Injection-Recovery Test for Omega_p', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
ax.set_xlim(-0.25, 0.25)
ax.set_ylim(-0.25, 0.25)
plt.tight_layout()
plot_path = os.path.join(REPO, "results", "plots", "omega_injection_recovery.png")
fig.savefig(plot_path, dpi=200)
print(f"Saved plot to {plot_path}")
print("\nDone!")
