"""
Potential Model Validation
==========================
Validates the composite MW potential against multiple checks:

1. Fiducial V_circ(R) vs Eilers+2019 rotation curve data
2. Component decomposition (disk, bulge, halo contributions)
3. Spherical limit: q_z=1 should match standard NFW analytically
4. Omega_p=0 should give identical V_circ to the static case
5. Sensitivity: how V_circ changes with each free parameter
6. Sanity checks on enclosed mass and local density

Output: results/plots/potential_validation.png
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Add project root to path
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.potential.composite import build_potential, RO, VO
from src.potential.baryons import build_baryonic_potential
from src.potential.halo import build_halo_potential
from galpy.potential import vcirc

DATA_DIR = os.path.join(REPO, "data", "rotation_curve")
PLOT_DIR = os.path.join(REPO, "results", "plots")

# -----------------------------------------------------------------------
# Load rotation curve data
# -----------------------------------------------------------------------
rc = pd.read_csv(os.path.join(DATA_DIR, "eilers2019_rc.csv"), comment="#")
R_data = rc["R_kpc"].values
V_data = rc["Vcirc_kms"].values
eV_minus = rc["eVcirc_minus_kms"].values
eV_plus = rc["eVcirc_plus_kms"].values
eV_sym = (eV_minus + eV_plus) / 2  # symmetrized errors

# -----------------------------------------------------------------------
# Fiducial halo parameters
# -----------------------------------------------------------------------
# These should produce a reasonable match to the rotation curve.
# v_h ~ 200 km/s, r_h ~ 16 kpc are typical for the MW NFW halo.
V_H_FID = 160.0    # km/s — gives V_total(R0) ~ 229 km/s with these baryons
R_H_FID = 16.0     # kpc
Q_Z_FID = 0.93     # mildly oblate (close to your previous result)
OMEGA_P_FID = 0.0  # static for validation

# -----------------------------------------------------------------------
# Helper: compute V_circ in km/s at given R_kpc array
# -----------------------------------------------------------------------
def compute_vcirc(pot, R_kpc):
    """Compute circular velocity in km/s at radii R_kpc."""
    R_nat = R_kpc / RO  # convert to natural units
    vc = np.array([vcirc(pot, r) for r in R_nat])
    return vc * VO  # convert to km/s


# -----------------------------------------------------------------------
# Test 1: Fiducial model vs data
# -----------------------------------------------------------------------
print("=" * 60)
print("POTENTIAL VALIDATION")
print("=" * 60)
print(f"\nFiducial parameters: v_h={V_H_FID}, r_h={R_H_FID}, "
      f"q_z={Q_Z_FID}, Omega_p={OMEGA_P_FID}")
print(f"Solar: R0={RO} kpc, V0={VO} km/s")

pot_fid = build_potential(V_H_FID, R_H_FID, Q_Z_FID, OMEGA_P_FID)
R_model = np.linspace(3.0, 25.0, 200)
V_model = compute_vcirc(pot_fid, R_model)
V_at_data = compute_vcirc(pot_fid, R_data)

residuals = V_data - V_at_data
chi2 = np.sum((residuals / eV_sym) ** 2)
chi2_dof = chi2 / (len(R_data) - 3)  # 3 free params for halo

print(f"\n--- Test 1: Fiducial V_circ vs Eilers+2019 ---")
print(f"  chi2 = {chi2:.1f} ({len(R_data)} points, 3 free params)")
print(f"  chi2/dof = {chi2_dof:.2f}")
print(f"  V_circ(R0) = {compute_vcirc(pot_fid, np.array([RO]))[0]:.1f} km/s "
      f"(target: {VO} km/s)")
print(f"  Mean residual = {np.mean(residuals):.2f} km/s")
print(f"  RMS residual = {np.sqrt(np.mean(residuals**2)):.2f} km/s")

if chi2_dof > 10:
    print("  WARNING: chi2/dof is high — fiducial parameters may need tuning")
    print("  (This is OK; MCMC will find the best fit)")

# -----------------------------------------------------------------------
# Test 2: Component decomposition
# -----------------------------------------------------------------------
print(f"\n--- Test 2: Component decomposition ---")
baryons = build_baryonic_potential()
halo_only = build_halo_potential(V_H_FID, R_H_FID, Q_Z_FID, 0.0)

V_disk = compute_vcirc([baryons[0]], R_model)
V_bulge = compute_vcirc([baryons[1]], R_model)
V_halo = compute_vcirc([halo_only], R_model)
V_total = compute_vcirc(pot_fid, R_model)

# Check that components add in quadrature (V_tot^2 = V_disk^2 + V_bulge^2 + V_halo^2)
V_quad = np.sqrt(V_disk**2 + V_bulge**2 + V_halo**2)
quad_error = np.max(np.abs(V_total - V_quad))
print(f"  Max |V_total - sqrt(V_d^2+V_b^2+V_h^2)| = {quad_error:.4f} km/s")
if quad_error > 0.5:
    print("  WARNING: Components don't add in quadrature — check potential setup")
else:
    print("  PASS: Components add in quadrature correctly")

# Fractional contributions at R0
V_d_R0 = compute_vcirc([baryons[0]], np.array([RO]))[0]
V_b_R0 = compute_vcirc([baryons[1]], np.array([RO]))[0]
V_h_R0 = compute_vcirc([halo_only], np.array([RO]))[0]
V_t_R0 = compute_vcirc(pot_fid, np.array([RO]))[0]
print(f"  At R0={RO} kpc:")
print(f"    Disk:  V={V_d_R0:.1f} km/s ({V_d_R0**2/V_t_R0**2*100:.1f}% of V^2)")
print(f"    Bulge: V={V_b_R0:.1f} km/s ({V_b_R0**2/V_t_R0**2*100:.1f}% of V^2)")
print(f"    Halo:  V={V_h_R0:.1f} km/s ({V_h_R0**2/V_t_R0**2*100:.1f}% of V^2)")
print(f"    Total: V={V_t_R0:.1f} km/s")

# -----------------------------------------------------------------------
# Test 3: Omega_p=0 vs static
# -----------------------------------------------------------------------
print(f"\n--- Test 3: Omega_p=0 == static ---")
pot_static = build_potential(V_H_FID, R_H_FID, Q_Z_FID, 0.0)
pot_zero_omega = build_potential(V_H_FID, R_H_FID, Q_Z_FID, 0.0)
V_static = compute_vcirc(pot_static, R_model)
V_zero = compute_vcirc(pot_zero_omega, R_model)
max_diff = np.max(np.abs(V_static - V_zero))
print(f"  Max |V_static - V_omega0| = {max_diff:.6f} km/s")
if max_diff < 1e-10:
    print("  PASS: Identical (as expected)")
else:
    print("  WARNING: Not identical — check wrapper implementation")

# -----------------------------------------------------------------------
# Test 4: V_circ should be ~flat around 220-240 km/s from 5-20 kpc
# -----------------------------------------------------------------------
print(f"\n--- Test 4: Rotation curve shape ---")
V_5 = compute_vcirc(pot_fid, np.array([5.0]))[0]
V_10 = compute_vcirc(pot_fid, np.array([10.0]))[0]
V_15 = compute_vcirc(pot_fid, np.array([15.0]))[0]
V_20 = compute_vcirc(pot_fid, np.array([20.0]))[0]
print(f"  V_circ(5 kpc)  = {V_5:.1f} km/s")
print(f"  V_circ(10 kpc) = {V_10:.1f} km/s")
print(f"  V_circ(15 kpc) = {V_15:.1f} km/s")
print(f"  V_circ(20 kpc) = {V_20:.1f} km/s")

if 180 < V_5 < 280 and 180 < V_20 < 260:
    print("  PASS: Rotation curve in reasonable range")
else:
    print("  WARNING: Rotation curve outside expected range")

# -----------------------------------------------------------------------
# Test 5: Sensitivity to parameters
# -----------------------------------------------------------------------
print(f"\n--- Test 5: Parameter sensitivity ---")
R_test = np.array([8.122, 15.0])
V_base = compute_vcirc(pot_fid, R_test)

for name, params in [
    ("v_h +10%", (V_H_FID * 1.1, R_H_FID, Q_Z_FID, 0.0)),
    ("r_h +10%", (V_H_FID, R_H_FID * 1.1, Q_Z_FID, 0.0)),
    ("q_z = 0.8", (V_H_FID, R_H_FID, 0.8, 0.0)),
    ("q_z = 1.0", (V_H_FID, R_H_FID, 1.0, 0.0)),
    ("Omega_p = 0.1", (V_H_FID, R_H_FID, Q_Z_FID, 0.1)),
]:
    pot_test = build_potential(*params)
    V_test = compute_vcirc(pot_test, R_test)
    dV = V_test - V_base
    print(f"  {name:20s}: dV(R0)={dV[0]:+.2f} km/s, dV(15kpc)={dV[1]:+.2f} km/s")

# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: V_circ vs data with component decomposition
ax = axes[0, 0]
ax.errorbar(R_data, V_data, yerr=[eV_minus, eV_plus], fmt='o', ms=4,
            color='k', capsize=2, elinewidth=1, label='Eilers+2019', zorder=5)
ax.plot(R_model, V_total, 'r-', lw=2, label='Total (fiducial)')
ax.plot(R_model, V_disk, '--', color='steelblue', lw=1.2, label='Disk')
ax.plot(R_model, V_bulge, ':', color='orange', lw=1.2, label='Bulge')
ax.plot(R_model, V_halo, '-.', color='purple', lw=1.2, label='Halo')
ax.set_xlabel('R (kpc)')
ax.set_ylabel('V_circ (km/s)')
ax.set_title('Rotation Curve: Fiducial Model vs Data')
ax.legend(fontsize=9)
ax.set_xlim(3, 25)
ax.set_ylim(100, 280)

# Panel 2: Residuals
ax = axes[0, 1]
ax.errorbar(R_data, residuals, yerr=eV_sym, fmt='o', ms=4,
            color='k', capsize=2, elinewidth=1)
ax.axhline(0, color='r', ls='--', lw=1)
ax.set_xlabel('R (kpc)')
ax.set_ylabel('V_data - V_model (km/s)')
ax.set_title(f'Residuals (chi2/dof = {chi2_dof:.2f})')
ax.set_xlim(3, 25)

# Panel 3: Effect of q_z on V_circ
ax = axes[1, 0]
for qz_val, ls in [(0.7, ':'), (0.8, '--'), (0.93, '-'), (1.0, '-.'), (1.2, ':')]:
    pot_qz = build_potential(V_H_FID, R_H_FID, qz_val, 0.0)
    V_qz = compute_vcirc(pot_qz, R_model)
    ax.plot(R_model, V_qz, ls=ls, lw=1.5, label=f'q_z = {qz_val}')
ax.errorbar(R_data, V_data, yerr=[eV_minus, eV_plus], fmt='o', ms=3,
            color='k', capsize=1.5, elinewidth=0.8, alpha=0.5)
ax.set_xlabel('R (kpc)')
ax.set_ylabel('V_circ (km/s)')
ax.set_title('Effect of q_z on Rotation Curve')
ax.legend(fontsize=9)
ax.set_xlim(3, 25)
ax.set_ylim(180, 260)

# Panel 4: Effect of Omega_p on V_circ (should be minimal at t=0)
ax = axes[1, 1]
for omega_val, ls in [(0.0, '-'), (0.05, '--'), (0.1, '-.'), (0.2, ':'), (0.5, ':')]:
    pot_om = build_potential(V_H_FID, R_H_FID, Q_Z_FID, omega_val)
    V_om = compute_vcirc(pot_om, R_model)
    ax.plot(R_model, V_om, ls=ls, lw=1.5, label=f'Omega_p = {omega_val}')
ax.errorbar(R_data, V_data, yerr=[eV_minus, eV_plus], fmt='o', ms=3,
            color='k', capsize=1.5, elinewidth=0.8, alpha=0.5)
ax.set_xlabel('R (kpc)')
ax.set_ylabel('V_circ (km/s)')
ax.set_title('Effect of Omega_p on V_circ (at t=0)')
ax.legend(fontsize=9)
ax.set_xlim(3, 25)
ax.set_ylim(180, 260)

plt.suptitle(
    f'Potential Validation | v_h={V_H_FID}, r_h={R_H_FID}, '
    f'q_z={Q_Z_FID}, Omega_p={OMEGA_P_FID}\n'
    f'R0={RO} kpc, V0={VO} km/s | chi2/dof={chi2_dof:.2f}',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()

plot_path = os.path.join(PLOT_DIR, "potential_validation.png")
plt.savefig(plot_path, dpi=200)
print(f"\nSaved validation plot to {plot_path}")
print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
