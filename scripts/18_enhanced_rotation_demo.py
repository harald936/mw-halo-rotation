"""
Enhanced demonstration of halo figure rotation effects.
=========================================================
Two publication-quality figures:

Figure 1: Orbit precession demo (enhanced version of the original)
    - 1×3 panels showing face-on and side views
    - Multiple Omega_p values as colored streams of particles
    - Time-colored trajectories with clear annotations

Figure 2: GD-1 stream observables vs Omega_p (THE key figure)
    - 2×2 panels: phi2, pm_phi1, pm_phi2, rv vs phi1
    - Model tracks at 5 different Omega_p values
    - Real GD-1 data with error bars overlaid
    - Directly demonstrates what halo rotation does to observables

Uses the EXACT potential model and stream generator from this project.
"""
import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.potential.composite import build_potential
from src.likelihood.stream_mock import (
    STREAMS, _extract_mock_particles, _ALL_TRACKS
)
from galpy.orbit import Orbit
from galpy.util.conversion import time_in_Gyr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec

RO, VO = 8.122, 229.0
PLOTS = os.path.join(REPO, "results", "plots")

# Best-fit halo parameters from the production dynesty run
VH_BEST = 193.25
RH_BEST = 23.80
QZ_BEST = 0.713

# Omega_p values to demonstrate
OMEGA_VALUES = [-0.30, -0.10, 0.0, 0.10, 0.30]
OMEGA_COLORS = ['#2166ac', '#67a9cf', '#1a1a1a', '#ef8a62', '#b2182b']
OMEGA_LABELS = [
    r'$\Omega_p = -0.30$ (retrograde)',
    r'$\Omega_p = -0.10$',
    r'$\Omega_p = 0$ (static)',
    r'$\Omega_p = +0.10$ (Bailin+04)',
    r'$\Omega_p = +0.30$ (prograde)',
]

# =========================================================================
# FIGURE 1: Enhanced orbit precession demo
# =========================================================================
print("=" * 60)
print("FIGURE 1: Enhanced orbit precession demo")
print("=" * 60)

# Orbit initial conditions: halo-crossing orbit at ~20 kpc
R0_orb = 20.0 / RO
vT0 = 180.0 / VO
z0 = 5.0 / RO
vR0 = 0.05
vz0 = 0.3
phi0 = 0.5
orb_init = [R0_orb, vR0, vT0, z0, vz0, phi0]

t_gyr = 4.0
t_conv = time_in_Gyr(VO, RO)
t_nat = t_gyr / t_conv
ts = np.linspace(0, -t_nat, 10000)
t_gyr_arr = ts * t_conv

print("Integrating orbits at 5 Omega_p values...")
orbit_data = {}
for omega in OMEGA_VALUES:
    pot = build_potential(VH_BEST, RH_BEST, QZ_BEST, omega)
    o = Orbit(orb_init)
    o.integrate(ts, pot)
    orbit_data[omega] = {
        'x': o.x(ts) * RO,
        'y': o.y(ts) * RO,
        'z': o.z(ts) * RO,
    }
    print(f"  Omega_p = {omega:+.2f}: done")

# --- Plot ---
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5.5))

projections = [
    ('x', 'y', 'Face-on (Galactic plane)'),
    ('x', 'z', 'Edge-on (x-z)'),
    ('y', 'z', 'Edge-on (y-z)'),
]

for col, (ax_x, ax_y, title) in enumerate(projections):
    ax = axes1[col]

    # Plot each Omega_p as a thin colored line
    for omega, color, label in zip(OMEGA_VALUES, OMEGA_COLORS, OMEGA_LABELS):
        orb = orbit_data[omega]
        xd = orb[ax_x]
        yd = orb[ax_y]
        lw = 1.5 if omega == 0.0 else 0.7
        alpha = 1.0 if omega == 0.0 else 0.7
        ax.plot(xd, yd, '-', color=color, lw=lw, alpha=alpha,
                label=label if col == 0 else None)

    # Mark Sun and Galactic Center
    if ax_x == 'x' and ax_y == 'y':
        ax.plot(-RO, 0, '*', color='gold', ms=12, markeredgecolor='darkorange',
                mew=1.0, zorder=10, label='Sun')
    ax.plot(0, 0, '+', color='black', ms=10, mew=2, zorder=10)

    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)
    ax.set_aspect('equal')
    ax.set_xlabel(f'{ax_x} (kpc)', fontsize=13)
    ax.set_ylabel(f'{ax_y} (kpc)', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.15, ls='--')
    ax.tick_params(labelsize=11)

axes1[0].legend(fontsize=9.5, loc='lower left', framealpha=0.9,
                edgecolor='gray', fancybox=False)

fig1.suptitle(
    'Effect of Dark Matter Halo Figure Rotation on Stellar Orbits\n'
    r'Same initial conditions, same halo mass — only $\Omega_p$ varies'
    f'\n(v$_h$ = {VH_BEST:.0f} km/s, r$_h$ = {RH_BEST:.0f} kpc, '
    f'q$_z$ = {QZ_BEST:.2f}, b = 0.9, tilt = 18°)',
    fontsize=13, fontweight='bold', y=1.08
)

plt.tight_layout()
out1 = os.path.join(PLOTS, "orbit_rotation_enhanced.png")
fig1.savefig(out1, dpi=250, bbox_inches='tight', facecolor='white')
print(f"Saved: {out1}")


# =========================================================================
# FIGURE 2: GD-1 stream observables vs Omega_p with real data
# =========================================================================
print()
print("=" * 60)
print("FIGURE 2: GD-1 stream observables vs Omega_p")
print("=" * 60)

# Generate mock GD-1 streams at each Omega_p
n_part = 200  # more particles for smoother tracks
gd1_mocks = {}

for omega in OMEGA_VALUES:
    print(f"  Generating GD-1 mock stream at Omega_p = {omega:+.2f}...")
    pot = build_potential(VH_BEST, RH_BEST, QZ_BEST, omega)
    try:
        result = _extract_mock_particles(pot, 'gd1', n_particles=n_part)
        if result is not None:
            phi1s, phi2s, pm1s, pm2s, rvs, dists = result
            # Sort by phi1 for clean plotting
            idx = np.argsort(phi1s)
            gd1_mocks[omega] = {
                'phi1': phi1s[idx], 'phi2': phi2s[idx],
                'pm1': pm1s[idx], 'pm2': pm2s[idx],
                'rv': rvs[idx], 'dist': dists[idx],
            }
            print(f"    -> {len(phi1s)} particles survived")
        else:
            print(f"    -> FAILED (too few particles)")
    except Exception as e:
        print(f"    -> FAILED: {e}")

# Load real GD-1 data
gd1_track = _ALL_TRACKS['gd1']['track']
gd1_rv = _ALL_TRACKS['gd1']['rv_track']

# --- Plot ---
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))

channels = [
    (0, 0, 'phi2', 'phi2_med', 'phi2_err', gd1_track,
     r'$\phi_2$ (deg)', 'Stream track (sky position)'),
    (0, 1, 'pm1', 'pm1_med', 'pm1_err', gd1_track,
     r'$\mu_{\phi_1}$ (mas/yr)', 'Proper motion along stream'),
    (1, 0, 'pm2', 'pm2_med', 'pm2_err', gd1_track,
     r'$\mu_{\phi_2}$ (mas/yr)', 'Proper motion across stream'),
    (1, 1, 'rv', 'rv_med', 'rv_err', gd1_rv,
     r'$v_r$ (km/s)', 'Radial velocity'),
]

for row, col_idx, key, med_col, err_col, data_df, ylabel, subtitle in channels:
    ax = axes2[row, col_idx]

    # Plot model tracks for each Omega_p
    for omega, color, label in zip(OMEGA_VALUES, OMEGA_COLORS, OMEGA_LABELS):
        if omega not in gd1_mocks:
            continue
        mock = gd1_mocks[omega]
        phi1 = mock['phi1']
        vals = mock[key]

        # Smooth by binning (running median in phi1 bins)
        phi1_grid = np.linspace(phi1.min() + 1, phi1.max() - 1, 80)
        smoothed = np.full_like(phi1_grid, np.nan)
        for i, p1 in enumerate(phi1_grid):
            mask = np.abs(phi1 - p1) < 2.0
            if mask.sum() >= 3:
                smoothed[i] = np.median(vals[mask])
        valid = np.isfinite(smoothed)

        lw = 2.5 if omega == 0.0 else 1.8
        ls = '-' if omega == 0.0 else '-'
        alpha = 1.0 if omega == 0.0 else 0.8
        ax.plot(phi1_grid[valid], smoothed[valid], ls, color=color,
                lw=lw, alpha=alpha, label=label, zorder=3)

        # Also scatter individual particles lightly
        ax.scatter(phi1, vals, c=color, s=3, alpha=0.08, zorder=1,
                   rasterized=True)

    # Overlay real data with error bars
    if data_df is not None and med_col in data_df.columns:
        phi1_data = data_df['phi1_deg'].values
        y_data = data_df[med_col].values
        y_err = data_df[err_col].values
        ax.errorbar(phi1_data, y_data, yerr=y_err,
                    fmt='o', color='black', ms=4, elinewidth=1.0,
                    capsize=2, capthick=0.8, zorder=5,
                    label='GD-1 data' if row == 0 and col_idx == 0 else None,
                    markeredgewidth=0.5)

    ax.set_xlabel(r'$\phi_1$ (deg)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(subtitle, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.12, ls='--')
    ax.tick_params(labelsize=11)

    # Set consistent phi1 range
    ax.set_xlim(-75, 10)

# Add legend to first panel
axes2[0, 0].legend(fontsize=8.5, loc='best', framealpha=0.9,
                   edgecolor='gray', fancybox=False, ncol=1)

fig2.suptitle(
    'How Dark Matter Halo Rotation Changes the GD-1 Stellar Stream\n'
    r'Model tracks at different $\Omega_p$ values vs observed data (black points)',
    fontsize=14, fontweight='bold', y=1.03
)

plt.tight_layout()
out2 = os.path.join(PLOTS, "gd1_omega_sensitivity.png")
fig2.savefig(out2, dpi=250, bbox_inches='tight', facecolor='white')
print(f"Saved: {out2}")


# =========================================================================
# FIGURE 3: All 4 streams — sky tracks at different Omega_p
# =========================================================================
print()
print("=" * 60)
print("FIGURE 3: All 4 streams — sky tracks at different Omega_p")
print("=" * 60)

# Generate mock streams for Pal5, Jhelum, Orphan at static + 2 extremes
omega_subset = [-0.30, 0.0, 0.30]
color_subset = ['#2166ac', '#1a1a1a', '#b2182b']
label_subset = [r'$\Omega_p = -0.30$', r'$\Omega_p = 0$ (static)', r'$\Omega_p = +0.30$']

all_stream_mocks = {'gd1': {}}
# Reuse GD-1 from above
for omega in omega_subset:
    if omega in gd1_mocks:
        all_stream_mocks['gd1'][omega] = gd1_mocks[omega]

for stream_name in ['pal5', 'jhelum', 'orphan']:
    all_stream_mocks[stream_name] = {}
    for omega in omega_subset:
        print(f"  Generating {stream_name} at Omega_p = {omega:+.2f}...")
        # For Orphan, include LMC
        include_lmc = (stream_name == 'orphan')
        pot = build_potential(VH_BEST, RH_BEST, QZ_BEST, omega,
                              include_lmc=include_lmc)
        try:
            result = _extract_mock_particles(pot, stream_name, n_particles=n_part)
            if result is not None:
                phi1s, phi2s, pm1s, pm2s, rvs, dists = result
                idx = np.argsort(phi1s)
                all_stream_mocks[stream_name][omega] = {
                    'phi1': phi1s[idx], 'phi2': phi2s[idx],
                    'pm1': pm1s[idx], 'pm2': pm2s[idx],
                    'rv': rvs[idx], 'dist': dists[idx],
                }
                print(f"    -> {len(phi1s)} particles")
            else:
                print(f"    -> FAILED")
        except Exception as e:
            print(f"    -> FAILED: {e}")

# --- Plot: 4×2 grid: phi2 and rv for each stream ---
fig3, axes3 = plt.subplots(4, 2, figsize=(16, 16))

stream_order = ['gd1', 'pal5', 'jhelum', 'orphan']
stream_titles = ['GD-1', 'Pal 5', 'Jhelum', 'Orphan-Chenab']
phi1_ranges = [(-75, 10), (-15, 10), (-10, 10), (-100, 50)]

for row, (sname, stitle, phi1_range) in enumerate(
        zip(stream_order, stream_titles, phi1_ranges)):
    mocks = all_stream_mocks.get(sname, {})
    track_data = _ALL_TRACKS[sname]['track']
    rv_data = _ALL_TRACKS[sname].get('rv_track')

    for col_idx, (key, med_col, err_col, use_rv, ylabel) in enumerate([
        ('phi2', 'phi2_med', 'phi2_err', False, r'$\phi_2$ (deg)'),
        ('rv', 'rv_med', 'rv_err', True, r'$v_r$ (km/s)'),
    ]):
        ax = axes3[row, col_idx]
        data_src = rv_data if (use_rv and rv_data is not None) else track_data

        # Plot model tracks
        for omega, color, label in zip(omega_subset, color_subset, label_subset):
            if omega not in mocks:
                continue
            mock = mocks[omega]
            phi1 = mock['phi1']
            vals = mock[key]

            # Running median smooth
            p1_min, p1_max = phi1_range
            phi1_grid = np.linspace(max(phi1.min() + 1, p1_min + 2),
                                     min(phi1.max() - 1, p1_max - 2), 60)
            smoothed = np.full_like(phi1_grid, np.nan)
            hw = 3.0 if sname == 'orphan' else 2.0
            for i, p1 in enumerate(phi1_grid):
                mask = np.abs(phi1 - p1) < hw
                if mask.sum() >= 3:
                    smoothed[i] = np.median(vals[mask])
            valid = np.isfinite(smoothed)

            lw = 2.5 if omega == 0.0 else 1.5
            ax.plot(phi1_grid[valid], smoothed[valid], '-', color=color,
                    lw=lw, alpha=0.85, label=label if row == 0 else None,
                    zorder=3)

        # Overlay data
        if data_src is not None and med_col in data_src.columns:
            phi1_d = data_src['phi1_deg'].values
            y_d = data_src[med_col].values
            y_e = data_src[err_col].values
            ax.errorbar(phi1_d, y_d, yerr=y_e, fmt='o', color='black',
                        ms=3.5, elinewidth=0.8, capsize=1.5, capthick=0.6,
                        zorder=5)

        ax.set_xlim(phi1_range)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.12, ls='--')
        ax.tick_params(labelsize=10)

        if row == 3:
            ax.set_xlabel(r'$\phi_1$ (deg)', fontsize=12)

        # Stream name on left column
        if col_idx == 0:
            ax.text(-0.18, 0.5, stitle, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', rotation=90,
                    va='center', ha='center')

# Column titles
axes3[0, 0].set_title('Sky Track', fontsize=13, fontweight='bold')
axes3[0, 1].set_title('Radial Velocity', fontsize=13, fontweight='bold')

axes3[0, 0].legend(fontsize=9, loc='best', framealpha=0.9)

fig3.suptitle(
    'Halo Figure Rotation Effect on All Four Stellar Streams\n'
    r'Retrograde ($\Omega_p < 0$) vs Static vs Prograde ($\Omega_p > 0$) — '
    'data shown as black points',
    fontsize=14, fontweight='bold', y=1.02
)

plt.tight_layout()
out3 = os.path.join(PLOTS, "all_streams_omega_sensitivity.png")
fig3.savefig(out3, dpi=250, bbox_inches='tight', facecolor='white')
print(f"Saved: {out3}")

print("\nAll figures complete!")
