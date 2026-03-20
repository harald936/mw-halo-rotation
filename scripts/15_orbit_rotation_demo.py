"""
Demonstration: How halo rotation affects stellar orbits.
========================================================
Shows the same orbit integrated in a static vs rotating triaxial halo,
using the EXACT potential model from this project. This proves that
even small Omega_p produces observable orbit distortions.

Inspired by Valluri, Price-Whelan & Snyder (2021) Figure 3,
but using our triaxial NFW + MWPotential2014 baryons model.
"""
import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.potential.composite import build_potential
from galpy.orbit import Orbit
from galpy.util.conversion import time_in_Gyr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

RO, VO = 8.122, 229.0
PLOTS = os.path.join(REPO, "results", "plots")

# -----------------------------------------------------------------------
# Set up a test orbit (Sgr-like polar orbit at ~20 kpc)
# -----------------------------------------------------------------------
# Start at R=20 kpc, z=0, with tangential velocity ~180 km/s
# This gives a roughly polar orbit that samples the triaxial shape
R0_orb = 20.0 / RO
vT0 = 180.0 / VO
z0 = 5.0 / RO
vR0 = 0.05
vz0 = 0.3
phi0 = 0.5

orb_init = [R0_orb, vR0, vT0, z0, vz0, phi0]

# Integration time: 4 Gyr backward
t_gyr = 4.0
t_conv = time_in_Gyr(VO, RO)
t_nat = t_gyr / t_conv
ts = np.linspace(0, -t_nat, 8000)
t_gyr_arr = ts * t_conv  # for coloring

# -----------------------------------------------------------------------
# Integrate in static vs rotating halos
# -----------------------------------------------------------------------
omega_values = [0.0, 0.05, 0.10, 0.20]
labels = [
    r'Static ($\Omega_p = 0$)',
    r'$\Omega_p = 0.05$ km/s/kpc',
    r'$\Omega_p = 0.10$ km/s/kpc (predicted)',
    r'$\Omega_p = 0.20$ km/s/kpc',
]

print("Integrating orbits...")
orbits = {}
for omega in omega_values:
    pot = build_potential(165, 16, 0.93, omega)
    o = Orbit(orb_init)
    o.integrate(ts, pot)

    # Get Cartesian coordinates
    x = o.x(ts) * RO
    y = o.y(ts) * RO
    z_arr = o.z(ts) * RO
    orbits[omega] = {'x': x, 'y': y, 'z': z_arr}
    print(f"  Omega_p = {omega}: done")

# -----------------------------------------------------------------------
# Plot — Valluri+2021 style
# -----------------------------------------------------------------------
print("Plotting...")

fig, axes = plt.subplots(4, 3, figsize=(15, 18))

projections = [
    ('x', 'y', 'Face-on (x-y)'),
    ('x', 'z', 'Side (x-z)'),
    ('y', 'z', 'Side (y-z)'),
]

cmap = plt.cm.coolwarm_r
norm = Normalize(vmin=-t_gyr, vmax=0)

for row, (omega, label) in enumerate(zip(omega_values, labels)):
    orb = orbits[omega]

    for col, (ax_x, ax_y, proj_title) in enumerate(projections):
        ax = axes[row, col]

        xdata = orb[ax_x]
        ydata = orb[ax_y]

        # Create colored line segments
        points = np.array([xdata, ydata]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=0.8, alpha=0.9)
        lc.set_array(t_gyr_arr[:-1])
        ax.add_collection(lc)

        # Plot Sun and GC
        if ax_x == 'x' and ax_y == 'y':
            ax.plot(-RO, 0, '*', color='gold', ms=8, markeredgecolor='orange', mew=0.8, zorder=5)

        ax.plot(0, 0, '+', color='black', ms=6, mew=1.5, zorder=5)

        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.1)

        if row == 0:
            ax.set_title(proj_title, fontsize=13, fontweight='bold')
        if col == 0:
            ax.set_ylabel(f'{label}\n{ax_y} (kpc)', fontsize=11)
        else:
            ax.set_ylabel(f'{ax_y} (kpc)', fontsize=10)
        if row == 3:
            ax.set_xlabel(f'{ax_x} (kpc)', fontsize=10)

        # Add colorbar only to rightmost column of each row
        if col == 2:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
            cb.set_label('Lookback time (Gyr)', fontsize=9)

fig.suptitle(
    'Effect of Halo Figure Rotation on Stellar Orbits\n'
    'Same initial conditions in our triaxial NFW + MWPotential2014 model',
    fontsize=16, fontweight='bold', y=1.01
)

plt.tight_layout()
outpath = os.path.join(PLOTS, "orbit_rotation_demo.png")
fig.savefig(outpath, dpi=200, bbox_inches='tight')
print(f"Saved to {outpath}")

# -----------------------------------------------------------------------
# Also make a cleaner 2-panel version (static vs predicted)
# -----------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

for panel, (omega, title, color) in enumerate([
    (0.0, r'Static Halo ($\Omega_p = 0$)', '#1565C0'),
    (0.10, r'Rotating Halo ($\Omega_p = 0.10$ — predicted)', '#C62828'),
]):
    orb = orbits[omega]
    for col, (ax_x, ax_y, proj) in enumerate(projections):
        ax = axes2[col]
        xdata = orb[ax_x]
        ydata = orb[ax_y]

        ax.plot(xdata, ydata, '-', color=color, lw=0.5, alpha=0.6,
                label=title if col == 0 else None)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_aspect('equal')
        ax.set_xlabel(f'{ax_x} (kpc)', fontsize=12)
        ax.set_ylabel(f'{ax_y} (kpc)', fontsize=12)
        ax.set_title(proj, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.15)
        ax.plot(0, 0, '+', color='black', ms=8, mew=2, zorder=5)

axes2[0].legend(fontsize=10, loc='lower left')

fig2.suptitle(
    'Static vs Rotating Halo: Same Orbit, Same Potential, Different $\\Omega_p$\n'
    'Rotation causes orbit precession visible in all projections',
    fontsize=14, fontweight='bold', y=1.05
)
plt.tight_layout()
outpath2 = os.path.join(PLOTS, "orbit_static_vs_rotating.png")
fig2.savefig(outpath2, dpi=200, bbox_inches='tight')
print(f"Saved to {outpath2}")

print("\nDone!")
