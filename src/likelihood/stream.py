"""
GD-1 stream likelihood.

Compares a model orbit (integrated backward in the rotating potential)
to the observed GD-1 binned track in stream coordinates.

Method:
  1. Set anchor point at phi1 ~ -15 deg (well-measured, high n_eff)
  2. Transform anchor from GD-1 frame to Galactocentric
  3. Integrate orbit backward ~1.2 Gyr in the (rotating) potential
  4. Project orbit back to GD-1 frame
  5. Interpolate model track at observed phi1 values
  6. Chi-squared on (phi2, pm_phi1, pm_phi2) residuals

The anchor's distance and RV come from the literature:
  distance ~ 10 kpc (Koposov+2010)
  RV ~ -133 km/s at the anchor point is the stream's systemic RV;
  however, the actual RV varies along the stream. We use the
  measured RV at the anchor position from our binned track.

pm_phi1 in the data is pm_phi1 (NOT pm_phi1_cosphi2).
gala's GD1 frame uses pm_phi1_cosphi2, so we convert.
"""

import numpy as np
import pandas as pd
import os

import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc

from galpy.orbit import Orbit
from galpy.util.conversion import time_in_Gyr

# Solar parameters (must match everywhere)
RO = 8.122   # kpc
VO = 229.0   # km/s
Z_SUN = 0.0208  # kpc

# Solar motion: Schoenrich+2010 peculiar + V0=229
# U, V, W = 11.1, 12.24, 7.25 km/s (peculiar)
# V_total = V0 + V_peculiar = 229 + 12.24 = 241.24 km/s
SOLAR_MOTION = coord.CartesianDifferential(
    [11.1, 241.24, 7.25] * u.km / u.s
)

GALCEN_FRAME = coord.Galactocentric(
    galcen_distance=RO * u.kpc,
    z_sun=Z_SUN * u.kpc,
    galcen_v_sun=SOLAR_MOTION,
)

# -----------------------------------------------------------------------
# Anchor point
# -----------------------------------------------------------------------
# Position along stream: phi1 = -15.5 deg (high n_eff bin)
# Observables from binned track
ANCHOR_PHI1 = -15.5   # deg
ANCHOR_PHI2 = -0.089  # deg (from track)
ANCHOR_DIST = 10.0    # kpc (Koposov+2010, literature value)
ANCHOR_PM1 = -10.883  # mas/yr (pm_phi1, NOT pm_phi1_cosphi2)
ANCHOR_PM2 = -2.531   # mas/yr (pm_phi2)
ANCHOR_RV = -183.3    # km/s (from RV track at phi1 ~ -14.5)

# Integration time
T_INTEG_GYR = 1.2     # Gyr backward

# -----------------------------------------------------------------------
# Load track data
# -----------------------------------------------------------------------
def _load_track_data(data_dir=None):
    """Load binned GD-1 track."""
    if data_dir is None:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        data_dir = os.path.join(repo, "data", "gd1")

    track = pd.read_csv(os.path.join(data_dir, "gd1_track.csv"))
    return track


_TRACK = _load_track_data()


# -----------------------------------------------------------------------
# Core functions
# -----------------------------------------------------------------------
def _anchor_to_galcen():
    """
    Transform the GD-1 anchor point to Galactocentric coordinates.

    Returns an astropy SkyCoord in the Galactocentric frame.
    """
    # Convert pm_phi1 to pm_phi1_cosphi2 (gala convention)
    cos_phi2 = np.cos(np.radians(ANCHOR_PHI2))
    pm_phi1_cosphi2 = ANCHOR_PM1 * cos_phi2

    gd1 = gc.GD1Koposov10(
        phi1=ANCHOR_PHI1 * u.deg,
        phi2=ANCHOR_PHI2 * u.deg,
        distance=ANCHOR_DIST * u.kpc,
        pm_phi1_cosphi2=pm_phi1_cosphi2 * u.mas / u.yr,
        pm_phi2=ANCHOR_PM2 * u.mas / u.yr,
        radial_velocity=ANCHOR_RV * u.km / u.s,
    )

    galcen = gd1.transform_to(GALCEN_FRAME)
    return galcen


def integrate_orbit(pot, t_gyr=T_INTEG_GYR, n_steps=2000):
    """
    Integrate the GD-1 anchor orbit backward in the given potential.

    Parameters
    ----------
    pot : list of galpy Potential objects
    t_gyr : float
        Integration time in Gyr (positive = backward).
    n_steps : int
        Number of integration steps.

    Returns
    -------
    orbit : galpy Orbit
        Integrated orbit object.
    ts : ndarray
        Time array in galpy natural units.
    """
    # Build anchor SkyCoord in GD-1 frame, transform to ICRS,
    # then let galpy handle the Galactocentric transform internally.
    # This avoids solar-motion double-counting between astropy and galpy.
    cos_phi2 = np.cos(np.radians(ANCHOR_PHI2))
    pm_phi1_cosphi2 = ANCHOR_PM1 * cos_phi2

    gd1 = gc.GD1Koposov10(
        phi1=ANCHOR_PHI1 * u.deg,
        phi2=ANCHOR_PHI2 * u.deg,
        distance=ANCHOR_DIST * u.kpc,
        pm_phi1_cosphi2=pm_phi1_cosphi2 * u.mas / u.yr,
        pm_phi2=ANCHOR_PM2 * u.mas / u.yr,
        radial_velocity=ANCHOR_RV * u.km / u.s,
    )
    icrs = gd1.transform_to(coord.ICRS())
    sc_icrs = coord.SkyCoord(icrs)

    o = Orbit(sc_icrs, ro=RO, vo=VO, zo=Z_SUN,
              solarmotion=[11.1, 12.24, 7.25])

    # Time array: 0 to -t_gyr (backward), in galpy natural units
    t_nat_max = t_gyr / time_in_Gyr(VO, RO)
    ts = np.linspace(0, -t_nat_max, n_steps)

    o.integrate(ts, pot)
    return o, ts


def orbit_to_gd1(orbit, ts):
    """
    Project an integrated orbit into GD-1 stream coordinates.

    Parameters
    ----------
    orbit : galpy Orbit
    ts : ndarray
        Time array.

    Returns
    -------
    phi1, phi2, pm1, pm2 : ndarrays
        Stream coordinates and proper motions along the orbit.
        pm1 is pm_phi1 (NOT pm_phi1_cosphi2).
    """
    # Get SkyCoord from orbit (in ICRS)
    # Must pass ro, vo, zo, and solarmotion for correct frame transform
    sc = orbit.SkyCoord(ts, ro=RO, vo=VO, zo=Z_SUN,
                        solarmotion=[11.1, 12.24, 7.25])

    # Transform to GD-1 frame
    gd1 = sc.transform_to(gc.GD1Koposov10())

    phi1 = gd1.phi1.deg
    phi2 = gd1.phi2.deg
    pm_phi1_cosphi2 = gd1.pm_phi1_cosphi2.value  # mas/yr
    pm_phi2 = gd1.pm_phi2.value  # mas/yr

    # Convert pm_phi1_cosphi2 to pm_phi1
    cos_phi2 = np.cos(np.radians(phi2))
    pm1 = pm_phi1_cosphi2 / cos_phi2

    return phi1, phi2, pm1, pm_phi2


def _select_stream_segment(phi1, phi2, pm1, pm2):
    """
    Select the orbit segment that corresponds to GD-1's stream track.

    The full orbit wraps around the sky, but we only want the segment
    near phi2 ~ 0 (close to the stream plane) and within the data's
    phi1 range. We select points with |phi2| < 15 deg.
    """
    near_stream = np.abs(phi2) < 15.0
    return phi1[near_stream], phi2[near_stream], pm1[near_stream], pm2[near_stream]


def _interpolate_track(phi1_orbit, obs_orbit, phi1_data):
    """
    Interpolate orbit observable at the data's phi1 positions.

    The orbit segment should already be filtered to the stream region.
    """
    if len(phi1_orbit) < 5:
        return np.full_like(phi1_data, np.nan, dtype=float), np.zeros(len(phi1_data), dtype=bool)

    # Sort by phi1
    sort_idx = np.argsort(phi1_orbit)
    phi1_sorted = phi1_orbit[sort_idx]
    obs_sorted = obs_orbit[sort_idx]

    # Remove duplicate phi1 values
    unique_mask = np.diff(phi1_sorted, prepend=-999) > 0.001
    phi1_sorted = phi1_sorted[unique_mask]
    obs_sorted = obs_sorted[unique_mask]

    if len(phi1_sorted) < 3:
        return np.full_like(phi1_data, np.nan, dtype=float), np.zeros(len(phi1_data), dtype=bool)

    # Interpolate (only within the orbit's phi1 range)
    valid = (phi1_data >= phi1_sorted.min()) & (phi1_data <= phi1_sorted.max())
    result = np.full_like(phi1_data, np.nan, dtype=float)
    if np.sum(valid) > 0:
        result[valid] = np.interp(phi1_data[valid], phi1_sorted, obs_sorted)

    return result, valid


def ln_likelihood_stream(pot):
    """
    Log-likelihood of the GD-1 stream track given a potential.

    Integrates the anchor orbit backward, projects to GD-1 frame,
    and computes chi-squared against the binned track data.

    ln L = -0.5 * sum_over_bins sum_over_channels
           [(obs - model)^2 / sigma^2]

    Channels: phi2, pm1, pm2 (3 observables per bin).

    Parameters
    ----------
    pot : list of galpy Potential objects

    Returns
    -------
    ln_L : float
        Log-likelihood. Returns -1e10 if orbit integration fails
        or the orbit doesn't cover the data's phi1 range.
    """
    try:
        orbit, ts = integrate_orbit(pot)
        phi1_orb, phi2_orb, pm1_orb, pm2_orb = orbit_to_gd1(orbit, ts)
    except Exception:
        return -1e10

    # Select only the orbit segment near the stream (|phi2| < 15 deg)
    phi1_seg, phi2_seg, pm1_seg, pm2_seg = _select_stream_segment(
        phi1_orb, phi2_orb, pm1_orb, pm2_orb
    )

    phi1_data = _TRACK["phi1_deg"].values

    # Interpolate model at data positions
    phi2_mod, valid2 = _interpolate_track(phi1_seg, phi2_seg, phi1_data)
    pm1_mod, valid_pm1 = _interpolate_track(phi1_seg, pm1_seg, phi1_data)
    pm2_mod, valid_pm2 = _interpolate_track(phi1_seg, pm2_seg, phi1_data)

    # Require all channels valid at each bin
    valid = valid2 & valid_pm1 & valid_pm2
    if np.sum(valid) < 10:
        return -1e10

    # Chi-squared
    chi2 = 0.0

    resid_phi2 = _TRACK["phi2_med"].values[valid] - phi2_mod[valid]
    chi2 += np.sum((resid_phi2 / _TRACK["phi2_err"].values[valid]) ** 2)

    resid_pm1 = _TRACK["pm1_med"].values[valid] - pm1_mod[valid]
    chi2 += np.sum((resid_pm1 / _TRACK["pm1_err"].values[valid]) ** 2)

    resid_pm2 = _TRACK["pm2_med"].values[valid] - pm2_mod[valid]
    chi2 += np.sum((resid_pm2 / _TRACK["pm2_err"].values[valid]) ** 2)

    return -0.5 * chi2
