"""
FINAL RUN: Dynesty + Mock Streams + LMC (Orphan only) + sigma_sys
==================================================================
5 free parameters: v_h, r_h, q_z, Omega_p, sigma_sys
4 streams with 200 mock particles each
LMC perturbation on Orphan-Chenab only
sigma_sys fitted (not guessed) — model uncertainty determined by data
12 cores, estimated ~3-4 hours
"""
import multiprocessing as mp
mp.set_start_method('fork', force=True)

import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.potential.composite import build_potential
from src.potential.lmc import build_lmc_potential
from src.likelihood.rotation_curve import ln_likelihood_rc
from src.stream.mockstream import generate_mock_stream
from src.likelihood.stream_mock import _particle_to_stream_coords, STREAMS, _ALL_TRACKS

from galpy.orbit import Orbit
from galpy.util.conversion import time_in_Gyr
import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc

import dynesty
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

RO, VO, Z_SUN = 8.122, 229.0, 0.0208
PLOTS = os.path.join(REPO, "results", "plots")

N_PARTICLES = 200
N_STEPS = 300

# Cache LMC potential (computed once)
_LMC_POT_CACHE = {}

# -----------------------------------------------------------------------
# Parameter setup: 5D
# -----------------------------------------------------------------------
PARAM_NAMES = ["v_h", "r_h", "q_z", "Omega_p", "sigma_sys"]
BOUNDS = [
    (100.0, 300.0),   # v_h
    (5.0, 40.0),      # r_h
    (0.5, 2.0),       # q_z
    (0.0, 0.5),       # Omega_p
    (0.01, 3.0),      # sigma_sys (degrees)
]

def prior_transform(u):
    theta = np.empty(5)
    for i, (lo, hi) in enumerate(BOUNDS):
        theta[i] = lo + u[i] * (hi - lo)
    return theta


def _mock_stream_single(pot, name, sigma_sys):
    """Mock stream likelihood for one stream with fitted sigma_sys."""
    cfg = STREAMS[name]
    data = _ALL_TRACKS[name]
    track = data['track']
    rv_track = data['rv_track']

    try:
        sc = coord.SkyCoord(
            ra=cfg['anchor_ra']*u.deg, dec=cfg['anchor_dec']*u.deg,
            distance=cfg['anchor_dist']*u.kpc,
            pm_ra_cosdec=cfg['anchor_pmra']*u.mas/u.yr,
            pm_dec=cfg['anchor_pmdec']*u.mas/u.yr,
            radial_velocity=cfg['anchor_rv']*u.km/u.s)
        prog = Orbit(sc, ro=RO, vo=VO, zo=Z_SUN, solarmotion=[11.1,12.24,7.25])
        t_nat = np.linspace(0, -cfg['t_strip']/time_in_Gyr(VO,RO), 1000)
        prog.integrate(t_nat, pot)

        orbits, _ = generate_mock_stream(pot, prog, cfg['t_strip'],
                                          n_particles=N_PARTICLES,
                                          v_kick_kms=cfg['v_kick'],
                                          n_steps_per_particle=N_STEPS)
        if len(orbits) < 20:
            return -1e10

        phi1s, phi2s, rvs = [], [], []
        for o in orbits:
            try:
                p1, p2, rv = _particle_to_stream_coords(o, cfg['frame'])
                phi1s.append(p1); phi2s.append(p2); rvs.append(rv)
            except:
                pass

        if len(phi1s) < 20:
            return -1e10

        phi1s, phi2s, rvs = np.array(phi1s), np.array(phi2s), np.array(rvs)
        near = np.abs(phi2s) < 15
        phi1s, phi2s, rvs = phi1s[near], phi2s[near], rvs[near]
        if len(phi1s) < 10:
            return -1e10
    except Exception:
        return -1e10

    def interp(x_o, y_o, x_d):
        if len(x_o) < 5:
            return np.full_like(x_d, np.nan, dtype=float), np.zeros(len(x_d), dtype=bool)
        idx = np.argsort(x_o)
        xs, ys = x_o[idx], y_o[idx]
        um = np.diff(xs, prepend=-999) > 0.001
        xs, ys = xs[um], ys[um]
        if len(xs) < 3:
            return np.full_like(x_d, np.nan, dtype=float), np.zeros(len(x_d), dtype=bool)
        v = (x_d >= xs.min()) & (x_d <= xs.max())
        r = np.full_like(x_d, np.nan, dtype=float)
        if v.sum() > 0:
            r[v] = np.interp(x_d[v], xs, ys)
        return r, v

    chi2 = 0.0
    n_terms = 0

    # phi2 with fitted sigma_sys
    phi1_data = track['phi1_deg'].values
    phi2_mod, v = interp(phi1s, phi2s, phi1_data)
    if v.sum() >= 3:
        sigma2 = track['phi2_err'].values[v]**2 + sigma_sys**2
        chi2 += np.sum((track['phi2_med'].values[v] - phi2_mod[v])**2 / sigma2)
        chi2 += np.sum(np.log(sigma2))  # penalty for large sigma_sys
        n_terms += v.sum()

    # RV (from main track if available)
    if 'rv_med' in track.columns:
        rv_mod, vrv = interp(phi1s, rvs, phi1_data)
        valid_rv = v & vrv
        if valid_rv.sum() >= 2:
            sigma_rv = 5.0  # km/s floor for RV (different unit than phi2)
            sigma2_rv = track['rv_err'].values[valid_rv]**2 + sigma_rv**2
            chi2 += np.sum((track['rv_med'].values[valid_rv] - rv_mod[valid_rv])**2 / sigma2_rv)
            n_terms += valid_rv.sum()

    # RV from separate track
    if rv_track is not None:
        phi1_rv = rv_track['phi1_deg'].values
        rv_mod2, vrv2 = interp(phi1s, rvs, phi1_rv)
        if vrv2.sum() >= 2:
            sigma2_rv2 = rv_track['rv_err'].values[vrv2]**2 + 5.0**2
            chi2 += np.sum((rv_track['rv_med'].values[vrv2] - rv_mod2[vrv2])**2 / sigma2_rv2)
            n_terms += vrv2.sum()

    if n_terms == 0:
        return -1e10

    return -0.5 * chi2


def log_likelihood(theta):
    v_h, r_h, q_z, Omega_p, sigma_sys = theta
    try:
        pot = build_potential(v_h, r_h, q_z, Omega_p, include_lmc=False)

        # For Orphan-Chenab, add LMC
        if 'lmc' not in _LMC_POT_CACHE:
            lmc_pot, _ = build_lmc_potential(pot)
            _LMC_POT_CACHE['lmc'] = lmc_pot
        pot_with_lmc = pot + [_LMC_POT_CACHE['lmc']]

        lnL = ln_likelihood_rc(pot)

        for name in ['gd1', 'pal5', 'jhelum']:
            lnL += _mock_stream_single(pot, name, sigma_sys)

        # Orphan-Chenab with LMC
        lnL += _mock_stream_single(pot_with_lmc, 'orphan', sigma_sys)

        return lnL if np.isfinite(lnL) else -1e10
    except Exception:
        return -1e10


# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------
if __name__ == '__main__':
    N_CORES = 12

    print(f"FINAL RUN: 5 params, {N_CORES} cores, mock streams + LMC")
    print(f"Params: {PARAM_NAMES}")
    print(f"Bounds: {BOUNDS}")

    with mp.Pool(N_CORES) as pool:
        sampler = dynesty.NestedSampler(
            log_likelihood, prior_transform, ndim=5,
            nlive=250, pool=pool, queue_size=N_CORES,
        )
        sampler.run_nested(print_progress=True)

    results = sampler.results
    from dynesty import utils as dyfunc

    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    resampled = dyfunc.resample_equal(samples, weights)

    print(f"\n{'='*55}")
    print(f"FINAL RESULTS")
    print(f"{'='*55}")
    print(f"Log-evidence: {results.logz[-1]:.1f} +/- {results.logzerr[-1]:.1f}")
    print(f"Evaluations: {sum(results.ncall)}")

    for i, name in enumerate(PARAM_NAMES):
        med = np.median(resampled[:, i])
        lo, hi = np.percentile(resampled[:, i], [16, 84])
        p95 = np.percentile(resampled[:, i], 95)
        print(f"  {name:10s}: {med:.4f}  68%=[{lo:.4f}, {hi:.4f}]  95%<{p95:.4f}")

    omega = resampled[:, 3]
    sigma = resampled[:, 4]
    print(f"\n*** OMEGA_P: {np.median(omega):.4f} km/s/kpc, 95% UL < {np.percentile(omega,95):.4f} ***")
    print(f"*** SIGMA_SYS: {np.median(sigma):.3f} deg (fitted, not guessed) ***")

    # Corner plot
    fig = corner.corner(resampled, labels=PARAM_NAMES,
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4f")
    fig.savefig(os.path.join(PLOTS, "final_corner.png"), dpi=150)

    # Omega_p posterior
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.hist(omega, bins=50, density=True, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(np.median(omega), color="red", lw=2, label=f"Median: {np.median(omega):.3f}")
    ax.axvline(np.percentile(omega,95), color="orange", ls="--", lw=2,
               label=f"95% UL: {np.percentile(omega,95):.3f}")
    ax.axvspan(0.08, 0.12, alpha=0.2, color="green", label="Bailin+04")
    ax.set_xlabel("Omega_p (km/s/kpc)")
    ax.set_ylabel("Posterior density")
    ax.set_title("Figure Rotation Rate — Final (Mock Streams + LMC + sigma_sys)")
    ax.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(PLOTS, "final_omega_p.png"), dpi=200)

    print("\nDone!")
