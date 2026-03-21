"""
Benchmark exact vs fast LMC orbit construction.

This script is intentionally low-cost and is meant for validating the
next-run optimization path before launching a new sampler. It compares
the exact LMC builder against the fast static-density-proxy builder and
reports both timing and orbit differences.
"""
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.potential.composite import build_potential
from src.potential.lmc import RO, VO, build_lmc_potential, build_lmc_potential_fast


PARAM_SETS = [
    (160.0, 16.0, 0.93, 0.10),
    (200.0, 20.0, 0.80, -0.20),
    (130.0, 12.0, 1.20, 0.30),
]


def _orbit_diff(exact, fast):
    ts = exact.t
    dx = (fast.x(ts) - exact.x(ts)) * RO
    dy = (fast.y(ts) - exact.y(ts)) * RO
    dz = (fast.z(ts) - exact.z(ts)) * RO
    dr = np.sqrt(dx**2 + dy**2 + dz**2)

    dvx = (fast.vx(ts) - exact.vx(ts)) * VO
    dvy = (fast.vy(ts) - exact.vy(ts)) * VO
    dvz = (fast.vz(ts) - exact.vz(ts)) * VO
    dv = np.sqrt(dvx**2 + dvy**2 + dvz**2)
    return dr, dv


if __name__ == "__main__":
    print("Benchmarking LMC builders")
    print("  exact: rotating-halo density inside Chandrasekhar friction")
    print("  fast : static halo density proxy + cdf_nr=101 + dopr54_c")

    for params in PARAM_SETS:
        print(f"\nparams={params}")
        pot = build_potential(*params, include_lmc=False)

        t0 = time.time()
        _, exact_orbit = build_lmc_potential(pot)
        t_exact = time.time() - t0

        t1 = time.time()
        _, fast_orbit = build_lmc_potential_fast(pot)
        t_fast = time.time() - t1

        dr, dv = _orbit_diff(exact_orbit, fast_orbit)
        print(f"  build_sec exact/fast : {t_exact:.3f} / {t_fast:.3f}")
        print(f"  speedup              : {t_exact / t_fast:.2f}x")
        print(f"  dr_kpc  median/max   : {np.median(dr):.4f} / {np.max(dr):.4f}")
        print(f"  dv_kms  median/max   : {np.median(dv):.4f} / {np.max(dv):.4f}")
