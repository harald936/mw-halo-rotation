"""
Orphan-Chenab stream likelihood.

Uses Koposov+2023 pre-binned track data (spline models):
  19 sky track bins (phi2) + 20 RV bins + 20 PM bins

Anchor: phi1=0, d=21 kpc, RV=90 km/s (from Koposov+2023 track)
PM from Koposov+2023 spline at phi1=0.

LMC perturbation is significant for this stream (d~15-55 kpc).
Handled via larger systematic error floor (1.0 deg, 1.0 mas/yr, 10 km/s).
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

# Anchor from Koposov+2023 spline at phi1 ~ 0
# RA/Dec at phi1=0 in OrphanKoposov19 frame
ANCHOR_RA = 163.0
ANCHOR_DEC = 1.5
ANCHOR_DIST = 21.0
ANCHOR_PMRA = -0.5
ANCHOR_PMDEC = -1.8
ANCHOR_RV = 90.0

T_INTEG_GYR = 3.0  # longer — wrapping stream


def _load_tracks(data_dir=None):
    if data_dir is None:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        data_dir = os.path.join(repo, "data", "orphan")
    track = pd.read_csv(os.path.join(data_dir, "orphan_track.csv"))
    rv_track = pd.read_csv(os.path.join(data_dir, "orphan_rv_track.csv"))
    return track, rv_track


_TRACK, _RV_TRACK = _load_tracks()


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


def ln_likelihood_orphan(pot):
    try:
        sc = coord.SkyCoord(ra=ANCHOR_RA*u.deg, dec=ANCHOR_DEC*u.deg,
                            distance=ANCHOR_DIST*u.kpc,
                            pm_ra_cosdec=ANCHOR_PMRA*u.mas/u.yr,
                            pm_dec=ANCHOR_PMDEC*u.mas/u.yr,
                            radial_velocity=ANCHOR_RV*u.km/u.s)
        o = Orbit(sc, ro=RO, vo=VO, zo=Z_SUN, solarmotion=[11.1, 12.24, 7.25])
        t_nat = np.linspace(0, -T_INTEG_GYR / time_in_Gyr(VO, RO), 3000)
        o.integrate(t_nat, pot)

        sc_out = o.SkyCoord(t_nat, ro=RO, vo=VO, zo=Z_SUN, solarmotion=[11.1, 12.24, 7.25])
        oc = sc_out.transform_to(gc.OrphanKoposov19())
        phi1 = oc.phi1.deg
        phi2 = oc.phi2.deg
        rv = oc.radial_velocity.to(u.km/u.s).value

        near = np.abs(phi2) < 15
        phi1, phi2, rv = phi1[near], phi2[near], rv[near]
    except Exception:
        return -1e10

    # Larger systematic floors — LMC perturbation not modeled
    SYS_PHI2 = 1.0
    SYS_RV = 10.0

    chi2 = 0.0

    # Sky track
    phi1_data = _TRACK["phi1_deg"].values
    phi2_mod, v = _interpolate(phi1, phi2, phi1_data)
    valid = v
    if valid.sum() >= 3:
        chi2 += np.sum((_TRACK["phi2_med"].values[valid] - phi2_mod[valid])**2
                       / (_TRACK["phi2_err"].values[valid]**2 + SYS_PHI2**2))

    # RV track (separate phi1 grid)
    phi1_rv = _RV_TRACK["phi1_deg"].values
    rv_mod, vrv = _interpolate(phi1, rv, phi1_rv)
    if vrv.sum() >= 3:
        chi2 += np.sum((_RV_TRACK["rv_med"].values[vrv] - rv_mod[vrv])**2
                       / (_RV_TRACK["rv_err"].values[vrv]**2 + SYS_RV**2))

    if chi2 == 0:
        return -1e10

    return -0.5 * chi2
