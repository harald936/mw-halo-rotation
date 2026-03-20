"""
Mock-stream-based joint stream likelihood.

Releases 100 test particles per stream from the progenitor via the
spray method and integrates each independently. The interpolated
track of the particle cloud is compared to the data.

This captures realistic stream width, velocity dispersion, and
the effect of particles stripped at different halo orientations
(critical for Omega_p sensitivity).

Streams: GD-1, Pal 5, Jhelum, Orphan-Chenab

This module is the single source of truth for mock-stream
likelihood evaluation. 09_run_final.py imports from here.
"""

import logging
import numpy as np
import os
import pandas as pd

import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc

from galpy.orbit import Orbit
from galpy.util.conversion import time_in_Gyr

from ..stream.mockstream import generate_mock_stream

log = logging.getLogger(__name__)

RO = 8.122
VO = 229.0
Z_SUN = 0.0208

N_PARTICLES = 100
N_STEPS = 300

SYS_PM = 0.5   # mas/yr — systematic floor for proper motions
SYS_RV = 5.0   # km/s — systematic floor for radial velocities
SYS_DIST = 2.0 # kpc — systematic floor for heliocentric distance

# -----------------------------------------------------------------------
# Stream configurations
# -----------------------------------------------------------------------
STREAMS = {
    'gd1': {
        'anchor_ra': 174.3149, 'anchor_dec': 53.0698,
        'anchor_dist': 10.0, 'anchor_pmra': -7.5848, 'anchor_pmdec': -8.2047,
        'anchor_rv': -183.3,
        't_strip': 1.2, 'v_kick': 2.0,
        'frame': gc.GD1Koposov10,
        'track_file': 'data/gd1/gd1_track.csv',
        'rv_file': 'data/gd1/gd1_track_rv_desi.csv',
        'dist_file': 'data/gd1/gd1_dist_track.csv',
        'phi2_max': 15.0,
    },
    'pal5': {
        'anchor_ra': 229.022, 'anchor_dec': -0.111,
        'anchor_dist': 21.9, 'anchor_pmra': -2.730, 'anchor_pmdec': -2.654,
        'anchor_rv': -58.4,
        't_strip': 3.0, 'v_kick': 2.0,
        'frame': gc.Pal5PriceWhelan18,
        'track_file': 'data/pal5/pal5_track.csv',
        'rv_file': None,
        'dist_file': 'data/pal5/pal5_dist_track.csv',
        'phi2_max': 10.0,
    },
    'jhelum': {
        'anchor_ra': 343.2, 'anchor_dec': -50.8,
        'anchor_dist': 12.0, 'anchor_pmra': 6.0, 'anchor_pmdec': -5.0,
        'anchor_rv': 20.0,
        't_strip': 2.0, 'v_kick': 5.0,
        'frame': gc.JhelumBonaca19,
        'track_file': 'data/jhelum/jhelum_track.csv',
        'rv_file': None,
        'phi2_max': 10.0,
    },
    'orphan': {
        'anchor_ra': 163.0, 'anchor_dec': 1.5,
        'anchor_dist': 21.0, 'anchor_pmra': -0.5, 'anchor_pmdec': -1.8,
        'anchor_rv': 90.0,
        't_strip': 3.0, 'v_kick': 5.0,
        'frame': gc.OrphanKoposov19,
        'track_file': 'data/orphan/orphan_track.csv',
        'rv_file': 'data/orphan/orphan_rv_track.csv',
        'dist_file': 'data/orphan/orphan_dist_track.csv',
        'phi2_max': 15.0,
    },
}


def _load_all_tracks():
    """Load all track data at import time."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    tracks = {}
    for name, cfg in STREAMS.items():
        t = pd.read_csv(os.path.join(repo, cfg['track_file']))
        rv_t = None
        if cfg.get('rv_file'):
            rv_path = os.path.join(repo, cfg['rv_file'])
            if os.path.exists(rv_path):
                rv_t = pd.read_csv(rv_path)
        dist_t = None
        if cfg.get('dist_file'):
            dist_path = os.path.join(repo, cfg['dist_file'])
            if os.path.exists(dist_path):
                dist_t = pd.read_csv(dist_path)
        tracks[name] = {'track': t, 'rv_track': rv_t, 'dist_track': dist_t}
    return tracks


_ALL_TRACKS = _load_all_tracks()


def _particle_to_stream_coords(orbit, frame_cls):
    """Extract stream coordinates for a single particle at t=0.

    Returns (phi1, phi2, pm1, pm2, rv, dist_kpc).
    """
    sc = orbit.SkyCoord(0., ro=RO, vo=VO, zo=Z_SUN,
                        solarmotion=[11.1, 12.24, 7.25])
    stream = sc.transform_to(frame_cls())
    phi1 = stream.phi1.deg
    phi2 = stream.phi2.deg
    pm_phi1_cosphi2 = stream.pm_phi1_cosphi2.value
    pm_phi2 = stream.pm_phi2.value
    rv = stream.radial_velocity.to(u.km / u.s).value
    dist_kpc = stream.distance.to(u.kpc).value
    cos_phi2 = np.cos(np.radians(float(phi2)))
    pm1 = pm_phi1_cosphi2 / cos_phi2
    return float(phi1), float(phi2), float(pm1), float(pm_phi2), float(rv), float(dist_kpc)


def _interp_track(x_orb, y_orb, x_data):
    """Interpolate mock stream observable at data phi1 positions.

    Sorts by phi1, removes duplicates, and linearly interpolates.
    Returns (values, valid_mask).
    """
    if len(x_orb) < 5:
        return np.full_like(x_data, np.nan, dtype=float), np.zeros(len(x_data), dtype=bool)
    idx = np.argsort(x_orb)
    xs, ys = x_orb[idx], y_orb[idx]
    um = np.diff(xs, prepend=-999) > 0.001
    xs, ys = xs[um], ys[um]
    if len(xs) < 3:
        return np.full_like(x_data, np.nan, dtype=float), np.zeros(len(x_data), dtype=bool)
    v = (x_data >= xs.min()) & (x_data <= xs.max())
    r = np.full_like(x_data, np.nan, dtype=float)
    if v.sum() > 0:
        r[v] = np.interp(x_data[v], xs, ys)
    return r, v


def _extract_mock_particles(pot, name, n_particles=N_PARTICLES):
    """Generate mock stream and extract particle coordinates.

    Returns (phi1s, phi2s, pm1s, pm2s, rvs, dists) arrays, or None on failure.
    """
    cfg = STREAMS[name]

    sc = coord.SkyCoord(
        ra=cfg['anchor_ra'] * u.deg, dec=cfg['anchor_dec'] * u.deg,
        distance=cfg['anchor_dist'] * u.kpc,
        pm_ra_cosdec=cfg['anchor_pmra'] * u.mas / u.yr,
        pm_dec=cfg['anchor_pmdec'] * u.mas / u.yr,
        radial_velocity=cfg['anchor_rv'] * u.km / u.s,
    )
    prog = Orbit(sc, ro=RO, vo=VO, zo=Z_SUN,
                 solarmotion=[11.1, 12.24, 7.25])
    t_nat = np.linspace(0, -cfg['t_strip'] / time_in_Gyr(VO, RO), 1000)
    prog.integrate(t_nat, pot)

    orbits, _ = generate_mock_stream(
        pot, prog, t_strip_gyr=cfg['t_strip'],
        n_particles=n_particles, v_kick_kms=cfg['v_kick'],
        n_steps_per_particle=N_STEPS,
    )

    if len(orbits) < 20:
        log.debug("%s: only %d orbits survived integration", name, len(orbits))
        return None

    phi1s, phi2s, pm1s, pm2s, rvs, dists = [], [], [], [], [], []
    n_failed = 0
    for orb in orbits:
        try:
            p1, p2, m1, m2, rv, d = _particle_to_stream_coords(orb, cfg['frame'])
            phi1s.append(p1)
            phi2s.append(p2)
            pm1s.append(m1)
            pm2s.append(m2)
            rvs.append(rv)
            dists.append(d)
        except (ValueError, AttributeError):
            n_failed += 1
            continue

    if n_failed > 0:
        log.debug("%s: %d/%d particles failed coord transform", name, n_failed, len(orbits))

    if len(phi1s) < 20:
        return None

    phi1s = np.array(phi1s)
    phi2s = np.array(phi2s)
    pm1s = np.array(pm1s)
    pm2s = np.array(pm2s)
    rvs = np.array(rvs)
    dists = np.array(dists)

    near = np.abs(phi2s) < cfg.get('phi2_max', 15.0)
    phi1s, phi2s, pm1s, pm2s, rvs, dists = (
        phi1s[near], phi2s[near], pm1s[near], pm2s[near], rvs[near], dists[near]
    )

    if len(phi1s) < 10:
        return None

    return phi1s, phi2s, pm1s, pm2s, rvs, dists


def mock_stream_likelihood_single(pot, name, sigma_sys):
    """Compute mock-stream likelihood for one stream.

    Parameters
    ----------
    pot : list of galpy Potential objects
    name : str
        Stream name ('gd1', 'pal5', 'jhelum', 'orphan').
    sigma_sys : float
        Fitted model systematic uncertainty in degrees,
        applied to the phi2 sky track channel only.
        RV uses a fixed 5 km/s floor; PM uses 0.5 mas/yr.

    Returns
    -------
    lnL : float
        Log-likelihood, or -1e10 on failure.
    """
    data = _ALL_TRACKS[name]
    track = data['track']
    rv_track = data['rv_track']
    dist_track = data.get('dist_track')

    try:
        result = _extract_mock_particles(pot, name)
    except (RuntimeError, ValueError) as e:
        log.debug("%s: mock stream generation failed: %s", name, e)
        return -1e10

    if result is None:
        return -1e10

    phi1s, phi2s, pm1s, pm2s, rvs, dists = result

    chi2 = 0.0
    n_terms = 0

    # phi2 with fitted sigma_sys + log-normalization penalty
    phi1_data = track['phi1_deg'].values
    phi2_mod, v = _interp_track(phi1s, phi2s, phi1_data)
    if v.sum() >= 3:
        sigma2 = track['phi2_err'].values[v] ** 2 + sigma_sys ** 2
        chi2 += np.sum((track['phi2_med'].values[v] - phi2_mod[v]) ** 2 / sigma2)
        chi2 += np.sum(np.log(sigma2))
        n_terms += v.sum()

    # PM channels (where track has pm1_med, pm2_med)
    if 'pm1_med' in track.columns:
        pm1_mod, vpm1 = _interp_track(phi1s, pm1s, phi1_data)
        valid_pm1 = v & vpm1
        if valid_pm1.sum() >= 2:
            sigma_pm1 = np.sqrt(track['pm1_err'].values[valid_pm1] ** 2 + SYS_PM ** 2)
            chi2 += np.sum((track['pm1_med'].values[valid_pm1] - pm1_mod[valid_pm1]) ** 2 / sigma_pm1 ** 2)
            n_terms += valid_pm1.sum()

    if 'pm2_med' in track.columns:
        pm2_mod, vpm2 = _interp_track(phi1s, pm2s, phi1_data)
        valid_pm2 = v & vpm2
        if valid_pm2.sum() >= 2:
            sigma_pm2 = np.sqrt(track['pm2_err'].values[valid_pm2] ** 2 + SYS_PM ** 2)
            chi2 += np.sum((track['pm2_med'].values[valid_pm2] - pm2_mod[valid_pm2]) ** 2 / sigma_pm2 ** 2)
            n_terms += valid_pm2.sum()

    # RV from main track (Pal5, Jhelum have RV in main track)
    if 'rv_med' in track.columns:
        rv_mod, vrv = _interp_track(phi1s, rvs, phi1_data)
        valid_rv = v & vrv
        if valid_rv.sum() >= 2:
            sigma2_rv = track['rv_err'].values[valid_rv] ** 2 + SYS_RV ** 2
            chi2 += np.sum((track['rv_med'].values[valid_rv] - rv_mod[valid_rv]) ** 2 / sigma2_rv)
            n_terms += valid_rv.sum()

    # RV from separate track (GD-1 DESI, Orphan-Chenab)
    if rv_track is not None:
        phi1_rv = rv_track['phi1_deg'].values
        rv_mod2, vrv2 = _interp_track(phi1s, rvs, phi1_rv)
        if vrv2.sum() >= 2:
            sigma2_rv2 = rv_track['rv_err'].values[vrv2] ** 2 + SYS_RV ** 2
            chi2 += np.sum((rv_track['rv_med'].values[vrv2] - rv_mod2[vrv2]) ** 2 / sigma2_rv2)
            n_terms += vrv2.sum()

    # Distance from separate track (Orphan-Chenab RR Lyrae, Koposov+2023)
    if dist_track is not None:
        phi1_dist = dist_track['phi1_deg'].values
        dist_mod, vdist = _interp_track(phi1s, dists, phi1_dist)
        if vdist.sum() >= 2:
            sigma2_dist = dist_track['dist_err'].values[vdist] ** 2 + SYS_DIST ** 2
            chi2 += np.sum((dist_track['dist_med'].values[vdist] - dist_mod[vdist]) ** 2 / sigma2_dist)
            n_terms += vdist.sum()

    if n_terms == 0:
        return -1e10

    return -0.5 * chi2


def ln_likelihood_mock_streams(pot_no_lmc, sigma_sys, pot_with_lmc=None):
    """Joint mock-stream likelihood for all 4 streams.

    GD-1, Pal 5, and Jhelum are evaluated with pot_no_lmc.
    Orphan-Chenab is evaluated with pot_with_lmc (which should
    include the LMC rebuilt for the current halo parameters).
    If pot_with_lmc is None, pot_no_lmc is used for all streams.

    Parameters
    ----------
    pot_no_lmc : list of galpy Potential objects
        MW potential without LMC.
    sigma_sys : float
        Fitted model systematic uncertainty in degrees.
    pot_with_lmc : list of galpy Potential objects, optional
        MW potential with LMC for Orphan-Chenab. If None, uses pot_no_lmc.
    """
    total = 0.0
    for name in STREAMS:
        pot = pot_with_lmc if (name == 'orphan' and pot_with_lmc is not None) else pot_no_lmc
        lnL = mock_stream_likelihood_single(pot, name, sigma_sys)
        if lnL <= -1e9:
            return -1e10
        total += lnL
    return total
