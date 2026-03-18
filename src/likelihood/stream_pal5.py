"""
Pal 5 stream likelihood.

Same method as GD-1 but for the Pal 5 stream:
  1. Anchor at the Pal 5 globular cluster (progenitor — known 6D position)
  2. Integrate orbit backward in the rotating potential
  3. Project to Pal 5 stream frame
  4. Chi-squared on (phi2, pm1, pm2, RV) residuals

Pal 5 anchor point (the globular cluster itself):
  RA = 229.022 deg, Dec = -0.111 deg
  distance = 21.9 kpc (Baumgardt)
  pmra = -2.730 mas/yr, pmdec = -2.654 mas/yr (Vasiliev & Baumgardt 2021)
  RV = -58.4 km/s (SIMBAD)

Pal 5 complements GD-1 by probing a different region of the halo:
  GD-1: [R, z] ~ [12, 7] kpc
  Pal 5: [R, z] ~ [8, 17] kpc (high above the plane)
"""

import numpy as np
import pandas as pd
import os

import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc

from galpy.orbit import Orbit
from galpy.util.conversion import time_in_Gyr

# Solar parameters
RO = 8.122
VO = 229.0
Z_SUN = 0.0208

# Pal 5 cluster anchor (progenitor)
ANCHOR_RA = 229.022     # deg
ANCHOR_DEC = -0.111     # deg
ANCHOR_DIST = 21.9      # kpc (Baumgardt)
ANCHOR_PMRA = -2.730    # mas/yr (Vasiliev & Baumgardt 2021, Gaia EDR3)
ANCHOR_PMDEC = -2.654   # mas/yr
ANCHOR_RV = -58.4       # km/s

T_INTEG_GYR = 3.0       # longer integration for Pal 5 (longer tails)


def _load_pal5_track(data_dir=None):
    if data_dir is None:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        data_dir = os.path.join(repo, "data", "pal5")
    return pd.read_csv(os.path.join(data_dir, "pal5_track.csv"))


_TRACK = _load_pal5_track()


def integrate_orbit_pal5(pot, t_gyr=T_INTEG_GYR, n_steps=3000):
    """Integrate Pal 5 cluster orbit backward."""
    sc_icrs = coord.SkyCoord(
        ra=ANCHOR_RA * u.deg,
        dec=ANCHOR_DEC * u.deg,
        distance=ANCHOR_DIST * u.kpc,
        pm_ra_cosdec=ANCHOR_PMRA * u.mas / u.yr,
        pm_dec=ANCHOR_PMDEC * u.mas / u.yr,
        radial_velocity=ANCHOR_RV * u.km / u.s,
    )

    o = Orbit(sc_icrs, ro=RO, vo=VO, zo=Z_SUN,
              solarmotion=[11.1, 12.24, 7.25])

    t_nat_max = t_gyr / time_in_Gyr(VO, RO)
    ts = np.linspace(0, -t_nat_max, n_steps)
    o.integrate(ts, pot)
    return o, ts


def orbit_to_pal5(orbit, ts):
    """Project orbit to Pal 5 stream coordinates."""
    sc = orbit.SkyCoord(ts, ro=RO, vo=VO, zo=Z_SUN,
                        solarmotion=[11.1, 12.24, 7.25])
    pal5 = sc.transform_to(gc.Pal5PriceWhelan18())

    phi1 = pal5.phi1.deg
    phi2 = pal5.phi2.deg
    pm_phi1_cosphi2 = pal5.pm_phi1_cosphi2.value
    pm_phi2 = pal5.pm_phi2.value
    rv = pal5.radial_velocity.to(u.km / u.s).value

    cos_phi2 = np.cos(np.radians(phi2))
    pm1 = pm_phi1_cosphi2 / cos_phi2

    return phi1, phi2, pm1, pm_phi2, rv


def _select_stream_segment(phi1, phi2, pm1, pm2, rv):
    """Select orbit segment near the Pal 5 stream (|phi2| < 10 deg)."""
    near = np.abs(phi2) < 10.0
    return phi1[near], phi2[near], pm1[near], pm2[near], rv[near]


def _interpolate_track(phi1_orbit, obs_orbit, phi1_data):
    """Interpolate orbit observable at data phi1 positions."""
    if len(phi1_orbit) < 5:
        return np.full_like(phi1_data, np.nan, dtype=float), \
               np.zeros(len(phi1_data), dtype=bool)

    sort_idx = np.argsort(phi1_orbit)
    phi1_sorted = phi1_orbit[sort_idx]
    obs_sorted = obs_orbit[sort_idx]

    unique_mask = np.diff(phi1_sorted, prepend=-999) > 0.001
    phi1_sorted = phi1_sorted[unique_mask]
    obs_sorted = obs_sorted[unique_mask]

    if len(phi1_sorted) < 3:
        return np.full_like(phi1_data, np.nan, dtype=float), \
               np.zeros(len(phi1_data), dtype=bool)

    valid = (phi1_data >= phi1_sorted.min()) & (phi1_data <= phi1_sorted.max())
    result = np.full_like(phi1_data, np.nan, dtype=float)
    if np.sum(valid) > 0:
        result[valid] = np.interp(phi1_data[valid], phi1_sorted, obs_sorted)
    return result, valid


def ln_likelihood_pal5(pot):
    """
    Log-likelihood of Pal 5 track given a potential.

    Uses phi2, pm1, pm2, and RV (4 channels).
    """
    try:
        orbit, ts = integrate_orbit_pal5(pot)
        phi1_orb, phi2_orb, pm1_orb, pm2_orb, rv_orb = orbit_to_pal5(orbit, ts)
    except Exception:
        return -1e10

    phi1s, phi2s, pm1s, pm2s, rvs = _select_stream_segment(
        phi1_orb, phi2_orb, pm1_orb, pm2_orb, rv_orb
    )

    phi1_data = _TRACK["phi1_deg"].values

    phi2_mod, v2 = _interpolate_track(phi1s, phi2s, phi1_data)
    pm1_mod, vp1 = _interpolate_track(phi1s, pm1s, phi1_data)
    pm2_mod, vp2 = _interpolate_track(phi1s, pm2s, phi1_data)
    rv_mod, vrv = _interpolate_track(phi1s, rvs, phi1_data)

    valid = v2 & vp1 & vp2 & vrv
    if np.sum(valid) < 3:
        return -1e10

    # Systematic error floors (same rationale as GD-1)
    SYS_PHI2 = 0.3    # deg
    SYS_PM = 0.3       # mas/yr
    SYS_RV = 5.0       # km/s

    chi2 = 0.0
    chi2 += np.sum((_TRACK["phi2_med"].values[valid] - phi2_mod[valid]) ** 2
                   / (_TRACK["phi2_err"].values[valid] ** 2 + SYS_PHI2 ** 2))
    chi2 += np.sum((_TRACK["pm1_med"].values[valid] - pm1_mod[valid]) ** 2
                   / (_TRACK["pm1_err"].values[valid] ** 2 + SYS_PM ** 2))
    chi2 += np.sum((_TRACK["pm2_med"].values[valid] - pm2_mod[valid]) ** 2
                   / (_TRACK["pm2_err"].values[valid] ** 2 + SYS_PM ** 2))
    chi2 += np.sum((_TRACK["rv_med"].values[valid] - rv_mod[valid]) ** 2
                   / (_TRACK["rv_err"].values[valid] ** 2 + SYS_RV ** 2))

    return -0.5 * chi2
