"""
Jhelum stream likelihood.

Anchor: median position of Jhelum members
  RA ~ 5 deg (phi1), d ~ 12 kpc (Li+2022)
  RV ~ 20 km/s, PM from Gaia

Uses phi2 + RV (2 channels, no PM — S5 has no PMs).
"""

import numpy as np
import pandas as pd
import os

import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc

from galpy.orbit import Orbit
from galpy.util.conversion import time_in_Gyr

RO = 8.122
VO = 229.0
Z_SUN = 0.0208

# Jhelum anchor — median of cleaned members
# RA/Dec from S5, d=12 kpc (Li+2022), RV=20 km/s
# PM from Gaia: pmra~6 mas/yr, pmdec~-5 mas/yr (Li+2022 Table 2)
# Correct ICRS at phi1=10, phi2=0 in JhelumBonaca19
ANCHOR_RA = 343.2
ANCHOR_DEC = -50.8
ANCHOR_DIST = 12.0
ANCHOR_PMRA = 6.0
ANCHOR_PMDEC = -5.0
ANCHOR_RV = 20.0

T_INTEG_GYR = 2.0


def _load_track(data_dir=None):
    if data_dir is None:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        data_dir = os.path.join(repo, "data", "jhelum")
    return pd.read_csv(os.path.join(data_dir, "jhelum_track.csv"))


_TRACK = _load_track()


def _interpolate(phi1_orb, obs_orb, phi1_data):
    if len(phi1_orb) < 5:
        return np.full_like(phi1_data, np.nan, dtype=float), np.zeros(len(phi1_data), dtype=bool)
    idx = np.argsort(phi1_orb)
    p1s, obs_s = phi1_orb[idx], obs_orb[idx]
    umask = np.diff(p1s, prepend=-999) > 0.001
    p1s, obs_s = p1s[umask], obs_s[umask]
    if len(p1s) < 3:
        return np.full_like(phi1_data, np.nan, dtype=float), np.zeros(len(phi1_data), dtype=bool)
    valid = (phi1_data >= p1s.min()) & (phi1_data <= p1s.max())
    result = np.full_like(phi1_data, np.nan, dtype=float)
    if valid.sum() > 0:
        result[valid] = np.interp(phi1_data[valid], p1s, obs_s)
    return result, valid


def ln_likelihood_jhelum(pot):
    try:
        sc = coord.SkyCoord(ra=ANCHOR_RA*u.deg, dec=ANCHOR_DEC*u.deg,
                            distance=ANCHOR_DIST*u.kpc,
                            pm_ra_cosdec=ANCHOR_PMRA*u.mas/u.yr,
                            pm_dec=ANCHOR_PMDEC*u.mas/u.yr,
                            radial_velocity=ANCHOR_RV*u.km/u.s)
        o = Orbit(sc, ro=RO, vo=VO, zo=Z_SUN, solarmotion=[11.1, 12.24, 7.25])
        t_nat = np.linspace(0, -T_INTEG_GYR / time_in_Gyr(VO, RO), 2000)
        o.integrate(t_nat, pot)

        sc_out = o.SkyCoord(t_nat, ro=RO, vo=VO, zo=Z_SUN, solarmotion=[11.1, 12.24, 7.25])
        jh = sc_out.transform_to(gc.JhelumBonaca19())
        phi1 = jh.phi1.deg
        phi2 = jh.phi2.deg
        rv = jh.radial_velocity.to(u.km/u.s).value

        near = np.abs(phi2) < 10
        phi1, phi2, rv = phi1[near], phi2[near], rv[near]
    except (RuntimeError, ValueError):
        return -1e10

    phi1_data = _TRACK["phi1_deg"].values
    phi2_mod, v2 = _interpolate(phi1, phi2, phi1_data)
    rv_mod, vrv = _interpolate(phi1, rv, phi1_data)
    valid = v2 & vrv
    if valid.sum() < 3:
        return -1e10

    SYS_PHI2 = 0.5
    SYS_RV = 10.0

    chi2 = 0.0
    chi2 += np.sum((_TRACK["phi2_med"].values[valid] - phi2_mod[valid])**2
                   / (_TRACK["phi2_err"].values[valid]**2 + SYS_PHI2**2))
    chi2 += np.sum((_TRACK["rv_med"].values[valid] - rv_mod[valid])**2
                   / (_TRACK["rv_err"].values[valid]**2 + SYS_RV**2))
    return -0.5 * chi2
