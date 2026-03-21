"""
Mock stream generator using test particle stripping.

Releases N particles from the progenitor's position over
the disruption time, each with a small velocity kick in the
tangential direction (spray method). Each particle then orbits
independently in the (rotating) potential.

This is NOT N-body (no self-gravity between particles).
It IS physically realistic for streams with low-mass progenitors
(GD-1, Pal 5, Jhelum) where self-gravity is negligible.

The resulting mock stream has realistic width, velocity dispersion,
and density structure that a single orbit cannot capture.
"""

import numpy as np
from galpy.orbit import Orbit
from galpy.util.conversion import time_in_Gyr

# Solar parameters
RO = 8.122
VO = 229.0


def generate_mock_stream(pot, progenitor_orbit, t_strip_gyr,
                         n_particles=100, v_kick_kms=2.0,
                         n_steps_per_particle=500):
    """
    Generate a mock stellar stream by stripping particles.

    Parameters
    ----------
    pot : list of galpy Potential objects
    progenitor_orbit : galpy Orbit
        Pre-integrated progenitor orbit (integrated backward).
    t_strip_gyr : float
        Total stripping duration in Gyr.
    n_particles : int
        Number of particles to release (half leading, half trailing).
    v_kick_kms : float
        Velocity kick at stripping in km/s (sets stream width).
        For GD-1/Pal5: ~2 km/s. For Jhelum/OC: ~5 km/s.
    n_steps_per_particle : int
        Integration steps per particle.

    Returns
    -------
    orbits : list of galpy Orbit objects
        Each particle's orbit, integrated to t=0.
    strip_times : ndarray
        Time each particle was stripped (in galpy natural units).
    """
    t_conv = time_in_Gyr(VO, RO)
    t_strip_nat = t_strip_gyr / t_conv
    v_kick_nat = v_kick_kms / VO

    # Stripping times: evenly spaced from -t_strip to 0
    strip_times = np.linspace(-t_strip_nat, 0, n_particles)

    orbits = []
    surviving_strip_times = []
    for i, t_strip in enumerate(strip_times):
        # Get progenitor position at stripping time in NATURAL UNITS.
        # The progenitor orbit is created from SkyCoord with ro/vo,
        # so .R() etc. return physical units by default. We need
        # natural units because the particle Orbit is created without
        # ro/vo to stay in the galpy natural-unit convention.
        R = progenitor_orbit.R(t_strip, use_physical=False)
        vR = progenitor_orbit.vR(t_strip, use_physical=False)
        vT = progenitor_orbit.vT(t_strip, use_physical=False)
        z = progenitor_orbit.z(t_strip, use_physical=False)
        vz = progenitor_orbit.vz(t_strip, use_physical=False)
        phi = progenitor_orbit.phi(t_strip)

        # Add small velocity kick (alternating leading/trailing)
        # Use deterministic seed based on particle index for reproducibility
        # Same parameters → same stream → same likelihood (required by samplers)
        rng = np.random.RandomState(seed=42 + i)
        sign = 1 if i % 2 == 0 else -1
        # Kick along the orbit (tangential) + small random perpendicular
        dvT = sign * v_kick_nat * (0.8 + 0.4 * rng.rand())
        dvR = v_kick_nat * 0.3 * rng.randn()
        dvz = v_kick_nat * 0.3 * rng.randn()

        # Create stripped particle
        particle = Orbit([R, vR + dvR, vT + dvT, z, vz + dvz, phi])

        # Integrate from stripping time to t=0
        ts = np.linspace(t_strip, 0, n_steps_per_particle)
        try:
            particle.integrate(ts, pot)
            orbits.append(particle)
            surviving_strip_times.append(t_strip)
        except (RuntimeError, ValueError):
            continue  # skip failed integrations

    return orbits, np.array(surviving_strip_times)


def mock_stream_to_track(orbits, transform_func, phi1_bins):
    """
    Convert mock stream particles to a binned track.

    Parameters
    ----------
    orbits : list of galpy Orbit objects
        Each at t=0 (present day).
    transform_func : callable
        Function that takes an orbit and returns
        (phi1, phi2, pm1, pm2, rv) at t=0 for that particle.
    phi1_bins : ndarray
        Bin edges in phi1 for the track.

    Returns
    -------
    track : dict
        Keys: phi1, phi2, pm1, pm2, rv (median per bin).
    """
    # Extract present-day positions for all particles
    phi1_all, phi2_all, pm1_all, pm2_all, rv_all = [], [], [], [], []

    for orb in orbits:
        try:
            p1, p2, m1, m2, rv = transform_func(orb)
            phi1_all.append(p1)
            phi2_all.append(p2)
            pm1_all.append(m1)
            pm2_all.append(m2)
            rv_all.append(rv)
        except (ValueError, AttributeError):
            continue

    if len(phi1_all) < 10:
        return None

    phi1_all = np.array(phi1_all)
    phi2_all = np.array(phi2_all)
    pm1_all = np.array(pm1_all)
    pm2_all = np.array(pm2_all)
    rv_all = np.array(rv_all)

    # Bin
    centers, phi2_med, pm1_med, pm2_med, rv_med = [], [], [], [], []
    for i in range(len(phi1_bins) - 1):
        mask = (phi1_all >= phi1_bins[i]) & (phi1_all < phi1_bins[i + 1])
        if mask.sum() >= 3:
            centers.append(np.median(phi1_all[mask]))
            phi2_med.append(np.median(phi2_all[mask]))
            pm1_med.append(np.median(pm1_all[mask]))
            pm2_med.append(np.median(pm2_all[mask]))
            rv_med.append(np.median(rv_all[mask]))

    if len(centers) < 3:
        return None

    return {
        'phi1': np.array(centers),
        'phi2': np.array(phi2_med),
        'pm1': np.array(pm1_med),
        'pm2': np.array(pm2_med),
        'rv': np.array(rv_med),
    }
