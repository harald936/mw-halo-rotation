"""
Mock-stream-based joint stream likelihood.

Instead of integrating a single orbit, releases ~100 test particles
per stream from the progenitor's Lagrange points and integrates
each independently. The median track of the particle cloud is
compared to the data.

This captures realistic stream width, velocity dispersion, and
the effect of particles stripped at different halo orientations
(critical for Omega_p sensitivity).

Streams: GD-1, Pal 5, Jhelum, Orphan-Chenab
"""

import numpy as np
import os
import pandas as pd

import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc

from galpy.orbit import Orbit
from galpy.util.conversion import time_in_Gyr

from ..stream.mockstream import generate_mock_stream

RO = 8.122
VO = 229.0
Z_SUN = 0.0208

N_PARTICLES = 100
N_STEPS = 300

# -----------------------------------------------------------------------
# Stream configurations
# -----------------------------------------------------------------------
STREAMS = {
    'gd1': {
        'anchor_ra': 174.3149, 'anchor_dec': 53.0698,  # ICRS from GD-1 anchor
        'anchor_dist': 10.0, 'anchor_pmra': -7.5848, 'anchor_pmdec': -8.2047,
        'anchor_rv': -183.3,
        't_strip': 1.2, 'v_kick': 2.0,
        'frame': gc.GD1Koposov10,
        'track_file': 'data/gd1/gd1_track.csv',
        'rv_file': 'data/gd1/gd1_track_rv_desi.csv',
        'sys_phi2': 2.0, 'sys_rv': 20.0,
        'channels': ['phi2', 'rv'],
    },
    'pal5': {
        'anchor_ra': 229.022, 'anchor_dec': -0.111,
        'anchor_dist': 21.9, 'anchor_pmra': -2.730, 'anchor_pmdec': -2.654,
        'anchor_rv': -58.4,
        't_strip': 3.0, 'v_kick': 2.0,
        'frame': gc.Pal5PriceWhelan18,
        'track_file': 'data/pal5/pal5_track.csv',
        'rv_file': None,
        'sys_phi2': 2.0, 'sys_rv': 20.0,
        'channels': ['phi2', 'rv'],
    },
    'jhelum': {
        'anchor_ra': 343.2, 'anchor_dec': -50.8,
        'anchor_dist': 12.0, 'anchor_pmra': 6.0, 'anchor_pmdec': -5.0,
        'anchor_rv': 20.0,
        't_strip': 2.0, 'v_kick': 5.0,
        'frame': gc.JhelumBonaca19,
        'track_file': 'data/jhelum/jhelum_track.csv',
        'rv_file': None,
        'sys_phi2': 2.0, 'sys_rv': 20.0,
        'channels': ['phi2', 'rv'],
    },
    'orphan': {
        'anchor_ra': 163.0, 'anchor_dec': 1.5,
        'anchor_dist': 21.0, 'anchor_pmra': -0.5, 'anchor_pmdec': -1.8,
        'anchor_rv': 90.0,
        't_strip': 3.0, 'v_kick': 5.0,
        'frame': gc.OrphanKoposov19,
        'track_file': 'data/orphan/orphan_track.csv',
        'rv_file': 'data/orphan/orphan_rv_track.csv',
        'sys_phi2': 3.0, 'sys_rv': 20.0,
        'channels': ['phi2', 'rv'],
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
        tracks[name] = {'track': t, 'rv_track': rv_t}
    return tracks


_ALL_TRACKS = _load_all_tracks()


def _particle_to_stream_coords(orbit, frame_cls):
    """Extract stream coordinates for a single particle at t=0."""
    sc = orbit.SkyCoord(0., ro=RO, vo=VO, zo=Z_SUN,
                        solarmotion=[11.1, 12.24, 7.25])
    stream = sc.transform_to(frame_cls())
    phi1 = stream.phi1.deg
    phi2 = stream.phi2.deg
    rv = stream.radial_velocity.to(u.km / u.s).value
    return float(phi1), float(phi2), float(rv)


def _mock_stream_likelihood_single(pot, name):
    """Compute likelihood for one stream using mock stream generation."""
    cfg = STREAMS[name]
    data = _ALL_TRACKS[name]
    track = data['track']
    rv_track = data['rv_track']

    try:
        # Create progenitor orbit
        sc = coord.SkyCoord(
            ra=cfg['anchor_ra'] * u.deg, dec=cfg['anchor_dec'] * u.deg,
            distance=cfg['anchor_dist'] * u.kpc,
            pm_ra_cosdec=cfg['anchor_pmra'] * u.mas / u.yr,
            pm_dec=cfg['anchor_pmdec'] * u.mas / u.yr,
            radial_velocity=cfg['anchor_rv'] * u.km / u.s,
        )
        prog = Orbit(sc, ro=RO, vo=VO, zo=Z_SUN,
                     solarmotion=[11.1, 12.24, 7.25])

        # Integrate progenitor backward
        t_nat = np.linspace(0, -cfg['t_strip'] / time_in_Gyr(VO, RO), 1000)
        prog.integrate(t_nat, pot)

        # Generate mock stream
        orbits, _ = generate_mock_stream(
            pot, prog, t_strip_gyr=cfg['t_strip'],
            n_particles=N_PARTICLES, v_kick_kms=cfg['v_kick'],
            n_steps_per_particle=N_STEPS,
        )

        if len(orbits) < 20:
            return -1e10

        # Extract stream coords for each particle
        phi1s, phi2s, rvs = [], [], []
        for orb in orbits:
            try:
                p1, p2, rv = _particle_to_stream_coords(orb, cfg['frame'])
                phi1s.append(p1)
                phi2s.append(p2)
                rvs.append(rv)
            except Exception:
                continue

        if len(phi1s) < 20:
            return -1e10

        phi1s = np.array(phi1s)
        phi2s = np.array(phi2s)
        rvs = np.array(rvs)

        # Select near-stream particles
        near = np.abs(phi2s) < 15
        phi1s, phi2s, rvs = phi1s[near], phi2s[near], rvs[near]

        if len(phi1s) < 10:
            return -1e10

    except Exception:
        return -1e10

    # Interpolate at data positions
    def interp(x_orb, y_orb, x_data):
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

    chi2 = 0.0

    # phi2
    phi1_data = track['phi1_deg'].values
    phi2_mod, v = interp(phi1s, phi2s, phi1_data)
    if v.sum() >= 3:
        sigma = np.sqrt(track['phi2_err'].values[v] ** 2 + cfg['sys_phi2'] ** 2)
        chi2 += np.sum((track['phi2_med'].values[v] - phi2_mod[v]) ** 2 / sigma ** 2)

    # RV from track (Pal5, Jhelum have RV in main track)
    if 'rv_med' in track.columns and 'rv' in cfg['channels']:
        rv_mod, vrv = interp(phi1s, rvs, phi1_data)
        valid_rv = v & vrv
        if valid_rv.sum() >= 2:
            sigma_rv = np.sqrt(track['rv_err'].values[valid_rv] ** 2 + cfg['sys_rv'] ** 2)
            chi2 += np.sum((track['rv_med'].values[valid_rv] - rv_mod[valid_rv]) ** 2 / sigma_rv ** 2)

    # RV from separate track (GD-1 DESI, Orphan-Chenab)
    if rv_track is not None:
        phi1_rv = rv_track['phi1_deg'].values
        rv_mod2, vrv2 = interp(phi1s, rvs, phi1_rv)
        if vrv2.sum() >= 2:
            sigma_rv2 = np.sqrt(rv_track['rv_err'].values[vrv2] ** 2 + cfg['sys_rv'] ** 2)
            chi2 += np.sum((rv_track['rv_med'].values[vrv2] - rv_mod2[vrv2]) ** 2 / sigma_rv2 ** 2)

    if chi2 == 0:
        return -1e10

    return -0.5 * chi2


def ln_likelihood_mock_streams(pot):
    """Joint mock-stream likelihood for all 4 streams."""
    total = 0.0
    for name in STREAMS:
        lnL = _mock_stream_likelihood_single(pot, name)
        if lnL <= -1e9:
            return -1e10
        total += lnL
    return total
