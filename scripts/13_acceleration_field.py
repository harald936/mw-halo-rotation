"""
Gravitational Acceleration Field + Complete Stream Data Atlas
=============================================================
Two publication figures:
1. Dense vector field of the MW acceleration from the triaxial
   rotating halo, with all 4 streams overlaid in physical space
2. Complete 6D data atlas: every data point across all streams
"""
import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import pandas as pd
import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc
from galpy.potential import evaluateRforces, evaluatezforces, evaluatephitorques

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

from src.potential.composite import build_potential

PLOTS = os.path.join(REPO, "results", "plots")
RO, VO, Z_SUN = 8.122, 229.0, 0.0208

galcen = coord.Galactocentric(
    galcen_distance=RO*u.kpc, z_sun=Z_SUN*u.kpc,
    galcen_v_sun=coord.CartesianDifferential([11.1, 12.24+VO, 7.25]*u.km/u.s),
)

STREAM_CFG = {
    'GD-1': {
        'track': 'data/gd1/gd1_track.csv',
        'rv_track': 'data/gd1/gd1_track_rv_desi.csv',
        'dist_track': None,
        'frame': gc.GD1Koposov10,
        'color': '#1565C0', 'dist_kpc': 10.0,
    },
    'Pal 5': {
        'track': 'data/pal5/pal5_track.csv',
        'rv_track': None, 'dist_track': None,
        'frame': gc.Pal5PriceWhelan18,
        'color': '#E65100', 'dist_kpc': 21.9,
    },
    'Jhelum': {
        'track': 'data/jhelum/jhelum_track.csv',
        'rv_track': None, 'dist_track': None,
        'frame': gc.JhelumBonaca19,
        'color': '#2E7D32', 'dist_kpc': 12.0,
    },
    'Orphan-Chenab': {
        'track': 'data/orphan/orphan_track.csv',
        'rv_track': 'data/orphan/orphan_rv_track.csv',
        'dist_track': 'data/orphan/orphan_dist_track.csv',
        'frame': gc.OrphanKoposov19,
        'color': '#C62828', 'dist_kpc': None,
    },
}

# Load all data
streams = {}
for name, cfg in STREAM_CFG.items():
    tr = pd.read_csv(os.path.join(REPO, cfg['track']))
    rv_tr = pd.read_csv(os.path.join(REPO, cfg['rv_track'])) if cfg['rv_track'] else None
    dist_tr = pd.read_csv(os.path.join(REPO, cfg['dist_track'])) if cfg['dist_track'] else None

    phi1 = tr['phi1_deg'].values
    phi2 = tr['phi2_med'].values
    if cfg.get('dist_track') and dist_tr is not None:
        dist = np.interp(phi1, dist_tr['phi1_deg'].values, dist_tr['dist_med'].values)
    else:
        dist = np.full(len(phi1), cfg['dist_kpc'])

    # Transform to Galactocentric (positions only for spatial plot)
    try:
        sc = cfg['frame'](
            phi1=phi1*u.deg, phi2=phi2*u.deg, distance=dist*u.kpc,
            pm_phi1_cosphi2=np.zeros(len(phi1))*u.mas/u.yr,
            pm_phi2=np.zeros(len(phi1))*u.mas/u.yr,
            radial_velocity=np.zeros(len(phi1))*u.km/u.s,
        )
        gc_c = sc.transform_to(galcen)
        streams[name] = {
            'track': tr, 'rv_track': rv_tr, 'dist_track': dist_tr,
            'x': gc_c.x.to(u.kpc).value, 'y': gc_c.y.to(u.kpc).value,
            'z': gc_c.z.to(u.kpc).value,
            'R': np.sqrt(gc_c.x.to(u.kpc).value**2 + gc_c.y.to(u.kpc).value**2),
            'color': cfg['color'], 'dist': dist,
        }
    except Exception as e:
        print(f"  {name}: {e}")

# -----------------------------------------------------------------------
# FIGURE 1: Dense acceleration vector field with streams
# -----------------------------------------------------------------------
print("Building acceleration field...")
pot = build_potential(165, 16, 0.93, 0.10)  # rotating halo
pot_static = build_potential(165, 16, 0.93, 0.0)

# Dense grid
N = 60
x_g = np.linspace(-55, 55, N)
y_g = np.linspace(-55, 55, N)
X, Y = np.meshgrid(x_g, y_g)
R_g = np.sqrt(X**2 + Y**2)
phi_g = np.arctan2(Y, X)

conv = VO**2 / RO  # natural to (km/s)^2/kpc

aR = np.zeros_like(R_g)
aphi = np.zeros_like(R_g)
aR_s = np.zeros_like(R_g)

for i in range(N):
    for j in range(N):
        rn = R_g[i,j] / RO
        if rn < 0.3:
            continue
        p = phi_g[i,j]
        aR[i,j] = evaluateRforces(pot, rn, 0, phi=p) * conv
        aphi[i,j] = evaluatephitorques(pot, rn, 0, phi=p) * conv / R_g[i,j]
        aR_s[i,j] = evaluateRforces(pot_static, rn, 0, phi=p) * conv

cos_p = np.cos(phi_g)
sin_p = np.sin(phi_g)
R_safe = np.where(R_g > 1, R_g, 1)

# Cartesian acceleration
ax_f = aR * cos_p - aphi * sin_p * R_safe
ay_f = aR * sin_p + aphi * cos_p * R_safe
a_mag = np.sqrt(ax_f**2 + ay_f**2)

# Normalize arrows for uniform visual length
a_mag_safe = np.where(a_mag > 0, a_mag, 1)
ax_norm = ax_f / a_mag_safe
ay_norm = ay_f / a_mag_safe

# Tangential component (non-axisymmetric signature)
aphi_mag = np.abs(aphi * R_safe)

# Edge-on grid
N_e = 50
R_e = np.linspace(1, 55, N_e)
z_e = np.linspace(-45, 45, N_e)
RE, ZE = np.meshgrid(R_e, z_e)

aR_e = np.zeros_like(RE)
az_e = np.zeros_like(RE)

for i in range(N_e):
    for j in range(N_e):
        rn = RE[i,j] / RO
        zn = ZE[i,j] / RO
        aR_e[i,j] = evaluateRforces(pot, rn, zn, phi=0) * conv
        az_e[i,j] = evaluatezforces(pot, rn, zn, phi=0) * conv

a_mag_e = np.sqrt(aR_e**2 + az_e**2)
aR_e_norm = aR_e / np.where(a_mag_e > 0, a_mag_e, 1)
az_e_norm = az_e / np.where(a_mag_e > 0, a_mag_e, 1)

print("Plotting Figure 1...")

fig, axes = plt.subplots(1, 2, figsize=(20, 9.5))

# --- Face-on ---
ax = axes[0]

# Color background by acceleration magnitude
im = ax.pcolormesh(X, Y, np.log10(np.where(R_g > 1, a_mag, np.nan)),
                    cmap='bone_r', shading='auto', alpha=0.5, zorder=0,
                    vmin=1.5, vmax=4.5)

# Dense arrows — uniform length, colored by magnitude
skip = 3
mask = R_g[::skip, ::skip] > 2
q = ax.quiver(X[::skip,::skip][mask], Y[::skip,::skip][mask],
              ax_norm[::skip,::skip][mask], ay_norm[::skip,::skip][mask],
              np.log10(a_mag[::skip,::skip][mask]),
              cmap='YlOrRd', clim=(2.0, 4.0),
              scale=35, width=0.003, headwidth=3.5, headlength=4,
              alpha=0.55, zorder=1)

# Halo ellipse
theta = np.linspace(0, 2*np.pi, 200)
tilt = np.radians(18)
ah, bh = 40, 40*0.9
xh = ah*np.cos(theta)*np.cos(tilt) - bh*np.sin(theta)*np.sin(tilt)
yh = ah*np.cos(theta)*np.sin(tilt) + bh*np.sin(theta)*np.cos(tilt)
ax.plot(xh, yh, '-', color='purple', lw=1.5, alpha=0.3)

# Streams as thick colored tracks
for name, s in streams.items():
    # Color track by distance
    points = np.array([s['x'], s['y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=s['color'], linewidths=5, alpha=0.95, zorder=4,
                        path_effects=[pe.Stroke(linewidth=7, foreground='white'), pe.Normal()])
    ax.add_collection(lc)
    ax.scatter(s['x'], s['y'], s=30, c=s['color'], edgecolors='white',
               linewidths=0.8, zorder=5)
    mid = len(s['x']) // 2
    ax.annotate(name, (s['x'][mid], s['y'][mid]), fontsize=12, fontweight='bold',
                color=s['color'], xytext=(10, 10), textcoords='offset points',
                path_effects=[pe.withStroke(linewidth=3, foreground='white')], zorder=6)

ax.plot(-RO, 0, '*', color='#FFB300', ms=18, markeredgecolor='#E65100', mew=1.5, zorder=7)
ax.text(-RO-1, -4, 'Sun', fontsize=11, fontweight='bold', color='#E65100',
        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
ax.plot(0, 0, '+', color='black', ms=12, mew=2.5, zorder=7)
ax.text(1.5, -3, 'GC', fontsize=11, fontweight='bold')

ax.set_xlim(-52, 52)
ax.set_ylim(-52, 52)
ax.set_aspect('equal')
ax.set_xlabel('x (kpc)', fontsize=14)
ax.set_ylabel('y (kpc)', fontsize=14)
ax.set_title('Face-on: Gravitational Acceleration Field\n(Triaxial Rotating Halo, $\\Omega_p=0.10$ km/s/kpc)',
             fontsize=14, fontweight='bold')

# Colorbar
cb = plt.colorbar(q, ax=ax, shrink=0.6, pad=0.02, aspect=25)
cb.set_label(r'log$_{10}$ |a| (km$^2$ s$^{-2}$ kpc$^{-1}$)', fontsize=11)

# --- Edge-on ---
ax = axes[1]

im_e = ax.pcolormesh(RE, ZE, np.log10(np.where(a_mag_e > 0, a_mag_e, np.nan)),
                      cmap='bone_r', shading='auto', alpha=0.5, zorder=0,
                      vmin=1.5, vmax=4.5)

skip_e = 3
mask_e = (RE[::skip_e,::skip_e] > 1) & (a_mag_e[::skip_e,::skip_e] > 0)
ax.quiver(RE[::skip_e,::skip_e][mask_e], ZE[::skip_e,::skip_e][mask_e],
          aR_e_norm[::skip_e,::skip_e][mask_e], az_e_norm[::skip_e,::skip_e][mask_e],
          np.log10(a_mag_e[::skip_e,::skip_e][mask_e]),
          cmap='YlOrRd', clim=(2.0, 4.0),
          scale=35, width=0.003, headwidth=3.5, headlength=4,
          alpha=0.55, zorder=1)

# Disk
ax.fill_between([0, 55], -0.5, 0.5, color='#1565C0', alpha=0.25, zorder=2)

# Tilted minor axis
ax.plot([0, 25*np.sin(tilt)], [0, 25*np.cos(tilt)], '-', color='purple', lw=2, alpha=0.5)
ax.plot([0, -25*np.sin(tilt)], [0, -25*np.cos(tilt)], '-', color='purple', lw=2, alpha=0.5)
ax.text(25*np.sin(tilt)+1, 25*np.cos(tilt)+1, '18° tilt', fontsize=10, color='purple',
        fontweight='bold', path_effects=[pe.withStroke(linewidth=2, foreground='white')])

# Streams
for name, s in streams.items():
    points = np.array([s['R'], s['z']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=s['color'], linewidths=5, alpha=0.95, zorder=4,
                        path_effects=[pe.Stroke(linewidth=7, foreground='white'), pe.Normal()])
    ax.add_collection(lc)
    ax.scatter(s['R'], s['z'], s=30, c=s['color'], edgecolors='white',
               linewidths=0.8, zorder=5)
    mid = len(s['R']) // 2
    ax.annotate(name, (s['R'][mid], s['z'][mid]), fontsize=12, fontweight='bold',
                color=s['color'], xytext=(8, 8), textcoords='offset points',
                path_effects=[pe.withStroke(linewidth=3, foreground='white')], zorder=6)

ax.plot(RO, 0, '*', color='#FFB300', ms=18, markeredgecolor='#E65100', mew=1.5, zorder=7)
ax.plot(0, 0, '+', color='black', ms=12, mew=2.5, zorder=7)

ax.set_xlim(0, 52)
ax.set_ylim(-42, 42)
ax.set_xlabel('R (kpc)', fontsize=14)
ax.set_ylabel('z (kpc)', fontsize=14)
ax.set_title('Edge-on: Acceleration Field + Halo Tilt\n4 streams probing R = 7$-$43 kpc, |z| = 0$-$35 kpc',
             fontsize=14, fontweight='bold')

cb_e = plt.colorbar(im_e, ax=ax, shrink=0.6, pad=0.02, aspect=25)
cb_e.set_label(r'log$_{10}$ |a| (km$^2$ s$^{-2}$ kpc$^{-1}$)', fontsize=11)

plt.tight_layout()
fig.savefig(os.path.join(PLOTS, "acceleration_field.png"), dpi=250, bbox_inches='tight')
print("Saved acceleration_field.png")

# -----------------------------------------------------------------------
# FIGURE 2: Complete 6D Stream Data Atlas
# -----------------------------------------------------------------------
print("\nPlotting data atlas...")

channels = [
    ('phi2_med', 'phi2_err', r'$\phi_2$ (deg)', 'Sky track'),
    ('pm1_med', 'pm1_err', r'$\mu_{\phi_1}$ (mas/yr)', 'PM along stream'),
    ('pm2_med', 'pm2_err', r'$\mu_{\phi_2}$ (mas/yr)', 'PM perpendicular'),
]

stream_names = ['GD-1', 'Pal 5', 'Jhelum', 'Orphan-Chenab']
n_rows = len(stream_names)
n_cols = 5  # phi2, pm1, pm2, rv, distance

fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 14))

for i, name in enumerate(stream_names):
    s = streams[name]
    tr = s['track']
    phi1 = tr['phi1_deg'].values
    color = s['color']

    # Col 0: phi2
    ax = axes[i, 0]
    ax.errorbar(phi1, tr['phi2_med'], yerr=tr['phi2_err'], fmt='o', ms=4,
                color=color, capsize=2, elinewidth=0.8, zorder=3)
    ax.set_ylabel(r'$\phi_2$ (deg)', fontsize=10)
    if i == 0: ax.set_title('Sky Track', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.12)
    ax.text(0.03, 0.92, name, transform=ax.transAxes, fontsize=12, fontweight='bold',
            color=color, va='top', bbox=dict(fc='white', ec=color, alpha=0.8, boxstyle='round,pad=0.3'))

    # Col 1: pm1
    ax = axes[i, 1]
    if 'pm1_med' in tr.columns and tr['pm1_med'].notna().any():
        m = tr['pm1_med'].notna()
        ax.errorbar(phi1[m], tr['pm1_med'][m], yerr=tr['pm1_err'][m], fmt='o', ms=4,
                    color=color, capsize=2, elinewidth=0.8)
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='#ccc')
    if i == 0: ax.set_title(r'PM $\mu_{\phi_1}$', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.12)

    # Col 2: pm2
    ax = axes[i, 2]
    if 'pm2_med' in tr.columns and tr['pm2_med'].notna().any():
        m = tr['pm2_med'].notna()
        ax.errorbar(phi1[m], tr['pm2_med'][m], yerr=tr['pm2_err'][m], fmt='o', ms=4,
                    color=color, capsize=2, elinewidth=0.8)
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='#ccc')
    if i == 0: ax.set_title(r'PM $\mu_{\phi_2}$', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.12)

    # Col 3: RV
    ax = axes[i, 3]
    if 'rv_med' in tr.columns and tr['rv_med'].notna().any():
        ax.errorbar(phi1, tr['rv_med'], yerr=tr['rv_err'], fmt='o', ms=4,
                    color=color, capsize=2, elinewidth=0.8)
    elif s.get('rv_track') is not None:
        rv_t = s['rv_track']
        ax.errorbar(rv_t['phi1_deg'], rv_t['rv_med'], yerr=rv_t['rv_err'], fmt='o', ms=4,
                    color=color, capsize=2, elinewidth=0.8)
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='#ccc')
    if i == 0: ax.set_title('Radial Velocity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.12)

    # Col 4: Distance
    ax = axes[i, 4]
    if s.get('dist_track') is not None:
        dt = s['dist_track']
        ax.errorbar(dt['phi1_deg'], dt['dist_med'], yerr=dt['dist_err'], fmt='o', ms=4,
                    color=color, capsize=2, elinewidth=0.8)
        ax.set_ylabel('d (kpc)', fontsize=10)
    elif s.get('dist') is not None and not np.all(s['dist'] == s['dist'][0]):
        ax.plot(phi1, s['dist'], 'o', ms=4, color=color)
    else:
        d_val = s['dist'][0] if s.get('dist') is not None else 0
        ax.axhline(d_val, color=color, lw=2, ls='--', alpha=0.6)
        ax.text(0.5, 0.65, f'd ≈ {d_val:.0f} kpc\n(fixed)', transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color=color)
    if i == 0: ax.set_title('Distance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.12)

    # x-axis label on bottom row
    if i == n_rows - 1:
        for j in range(n_cols):
            axes[i, j].set_xlabel(r'$\phi_1$ (deg)', fontsize=10)

# Count data points
total = 0
for name, s in streams.items():
    tr = s['track']
    n = len(tr)
    if 'pm1_med' in tr.columns: n += tr['pm1_med'].notna().sum()
    if 'pm2_med' in tr.columns: n += tr['pm2_med'].notna().sum()
    if 'rv_med' in tr.columns: n += tr['rv_med'].notna().sum()
    if s.get('rv_track') is not None: n += len(s['rv_track'])
    if s.get('dist_track') is not None: n += len(s['dist_track'])
    total += n
total += 32  # RC

fig.suptitle(f'Complete Stream Data Atlas — {total} Data Points Across 6 Observable Channels',
             fontsize=17, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS, "stream_data_atlas.png"), dpi=200, bbox_inches='tight')
print("Saved stream_data_atlas.png")

print("\nDone!")
