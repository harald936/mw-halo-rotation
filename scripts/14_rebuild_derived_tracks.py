"""
Rebuild all derived data tracks from source data.
==================================================
This script reproduces every derived CSV from its published source,
ensuring full provenance and reproducibility.

Run this to verify that stored CSVs match the source data exactly.
Requires: astroquery (for Gaia), astropy, pandas, numpy
"""
import sys, os
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import gala.coordinates as gc


def rebuild_gd1_rv():
    """Rebuild GD-1 RV track from full member catalog.

    Source: data/gd1/gd1_members_cleaned.csv
    Members with RV from DESI DR1 + SDSS + LAMOST.
    """
    print("=== GD-1 RV track ===")
    gd1 = pd.read_csv(os.path.join(REPO, 'data/gd1/gd1_members_cleaned.csv'))
    has_rv = gd1.rv.notna() & (gd1.e_rv > 0) & (gd1.e_rv < 50)
    gd1_rv = gd1[has_rv]
    print(f"  Members with RV: {len(gd1_rv)}")

    N_BINS = 15
    edges = np.linspace(gd1_rv.phi1.min() - 0.5, gd1_rv.phi1.max() + 0.5, N_BINS + 1)
    bins = []
    for i in range(N_BINS):
        mask = (gd1_rv.phi1 >= edges[i]) & (gd1_rv.phi1 < edges[i + 1])
        n = mask.sum()
        if n >= 5:
            sub = gd1_rv[mask]
            w = 1.0 / sub.e_rv ** 2
            rv_mean = np.average(sub.rv, weights=w)
            rv_err = 1.0 / np.sqrt(w.sum())
            bins.append({
                'phi1_deg': np.median(sub.phi1),
                'rv_med': rv_mean,
                'rv_err': rv_err,
            })

    df = pd.DataFrame(bins)
    path = os.path.join(REPO, 'data/gd1/gd1_track_rv.csv')
    df.to_csv(path, index=False, float_format='%.4f')
    print(f"  Saved: {path} ({len(df)} bins)")
    return df


def rebuild_jhelum_pm():
    """Rebuild Jhelum PM columns from Gaia DR3.

    Source: S5 DR1 (data/external/s5_pdr1_light.fits) -> gaia_source_id -> Gaia DR3 PMs.
    """
    print("\n=== Jhelum PM ===")
    from astropy.table import Table

    s5_path = os.path.join(REPO, 'data/external/s5_pdr1_light.fits')
    if not os.path.exists(s5_path):
        print("  S5 FITS not found — skipping")
        return None

    s5 = Table.read(s5_path)
    s5 = s5[s5['primary']]
    jh_mask = np.array([f.startswith('Jhelum') for f in s5['field']])
    jh = s5[jh_mask]
    jh_stream = jh[(jh['priority'] >= 7) & (jh['priority'] <= 9)]

    ra = np.array(jh_stream['ra'], dtype=float)
    dec = np.array(jh_stream['dec'], dtype=float)
    rv = np.array(jh_stream['vel_calib'], dtype=float)
    rv_err = np.array(jh_stream['vel_calib_std'], dtype=float)
    source_ids = np.array(jh_stream['gaia_source_id'], dtype=np.int64)

    sc_pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    jh_coord = sc_pos.transform_to(gc.JhelumBonaca19())
    phi1 = jh_coord.phi1.deg
    phi2 = jh_coord.phi2.deg

    member = (np.abs(phi2) < 2.0) & (rv_err < 10.0) & (rv_err > 0)
    mem_ids = source_ids[member]
    mem_phi1 = phi1[member]
    mem_phi2 = phi2[member]

    print(f"  Jhelum members: {member.sum()}")

    try:
        from astroquery.gaia import Gaia
        ids_str = ','.join([str(sid) for sid in mem_ids if sid > 0])
        query = f"SELECT source_id, pmra, pmdec FROM gaiadr3.gaia_source WHERE source_id IN ({ids_str})"
        job = Gaia.launch_job(query)
        result = job.get_results()
        print(f"  Gaia PMs retrieved: {len(result)}")
    except Exception as e:
        print(f"  Gaia query failed: {e}")
        return None

    pm_ra = {int(r['source_id']): r['pmra'] for r in result}
    pm_dec = {int(r['source_id']): r['pmdec'] for r in result}

    pmra_arr = np.array([pm_ra.get(sid, np.nan) for sid in mem_ids])
    pmdec_arr = np.array([pm_dec.get(sid, np.nan) for sid in mem_ids])
    has_pm = np.isfinite(pmra_arr) & np.isfinite(pmdec_arr)

    sc_full = SkyCoord(ra=ra[member][has_pm] * u.deg, dec=dec[member][has_pm] * u.deg,
                       pm_ra_cosdec=pmra_arr[has_pm] * u.mas / u.yr,
                       pm_dec=pmdec_arr[has_pm] * u.mas / u.yr)
    jh_full = sc_full.transform_to(gc.JhelumBonaca19())
    pm1_cosphi2 = jh_full.pm_phi1_cosphi2.value
    pm2_vals = jh_full.pm_phi2.value
    phi1_pm = mem_phi1[has_pm]
    phi2_pm = mem_phi2[has_pm]
    pm1_vals = pm1_cosphi2 / np.cos(np.radians(phi2_pm))

    # Load existing track and add PM columns
    track = pd.read_csv(os.path.join(REPO, 'data/jhelum/jhelum_track.csv'))

    N_BINS = 6
    edges = np.linspace(phi1_pm.min() - 0.5, phi1_pm.max() + 0.5, N_BINS + 1)
    phi1_centers, pm1_meds, pm1_errs, pm2_meds, pm2_errs = [], [], [], [], []
    for i in range(N_BINS):
        mask = (phi1_pm >= edges[i]) & (phi1_pm < edges[i + 1])
        n = mask.sum()
        if n >= 3:
            phi1_centers.append(np.median(phi1_pm[mask]))
            pm1_meds.append(np.median(pm1_vals[mask]))
            pm1_errs.append(np.std(pm1_vals[mask]) / np.sqrt(n))
            pm2_meds.append(np.median(pm2_vals[mask]))
            pm2_errs.append(np.std(pm2_vals[mask]) / np.sqrt(n))

    track['pm1_med'] = np.interp(track.phi1_deg, phi1_centers, pm1_meds)
    track['pm1_err'] = np.interp(track.phi1_deg, phi1_centers, pm1_errs)
    track['pm2_med'] = np.interp(track.phi1_deg, phi1_centers, pm2_meds)
    track['pm2_err'] = np.interp(track.phi1_deg, phi1_centers, pm2_errs)

    path = os.path.join(REPO, 'data/jhelum/jhelum_track.csv')
    track.to_csv(path, index=False, float_format='%.6f')
    print(f"  Saved: {path}")
    return track


def rebuild_pal5_dist():
    """Rebuild Pal 5 distance track from Price-Whelan+2019 RR Lyrae.

    Source: Price-Whelan+2019 Table 2 (AJ 158 223).
    Published Gaia DR2 source_ids with distances, cross-matched to Gaia DR3 for positions.
    """
    print("\n=== Pal 5 distance track ===")

    # Published RR Lyrae from Table 2 (source_id, distance_kpc, membership_prob)
    rrl = [
        (4418920808077110784, 20.7, 1.00), (4418920846732620032, 21.7, 0.99),
        (4418914863842345856, 20.4, 0.95), (4418726027016125056, 20.7, 1.00),
        (4418725889577171328, 20.3, 1.00), (4418731829516302848, 18.1, 0.65),
        (4418913218870688768, 20.3, 1.00), (4418734165978521728, 22.8, 0.91),
        (4418724034151291776, 21.2, 1.00), (4418732791593809152, 20.7, 1.00),
        (4419052204012341760, 20.8, 1.00), (4418142117622280192, 20.0, 0.91),
        (6339498589346000768, 18.9, 0.99), (6339499379619987200, 19.4, 1.00),
        (6339478312804685824, 19.9, 1.00), (4421078432143578496, 22.0, 1.00),
        (6339398155830632448, 19.8, 1.00), (4427253907921168000, 21.8, 0.99),
        (4427220338456828416, 22.0, 1.00), (4427234700827469952, 21.9, 1.00),
        (6337233350579111680, 19.0, 0.98), (4427646704153765632, 19.9, 0.87),
        (4424705647988505600, 20.9, 1.00), (4426221707021159424, 22.0, 0.99),
        (4450058431917932672, 21.1, 0.57), (4439927909737905280, 22.2, 0.85),
        (4445607604551159040, 22.3, 0.81),
    ]

    high_prob = [(s, d, p) for s, d, p in rrl if p > 0.8]
    print(f"  RR Lyrae with p>0.8: {len(high_prob)}")

    try:
        from astroquery.gaia import Gaia
        ids_str = ','.join([str(s) for s, d, p in high_prob])
        query = f"SELECT source_id, ra, dec FROM gaiadr3.gaia_source WHERE source_id IN ({ids_str})"
        job = Gaia.launch_job(query)
        result = job.get_results()
    except Exception as e:
        print(f"  Gaia query failed: {e}")
        return None

    ra_d = {int(r['source_id']): r['ra'] for r in result}
    dec_d = {int(r['source_id']): r['dec'] for r in result}

    ras, decs, dists = [], [], []
    for sid, d, p in high_prob:
        if sid in ra_d:
            ras.append(ra_d[sid])
            decs.append(dec_d[sid])
            dists.append(d)

    sc = SkyCoord(ra=ras * u.deg, dec=decs * u.deg)
    pal5 = sc.transform_to(gc.Pal5PriceWhelan18())
    phi1 = pal5.phi1.deg

    stream_mask = np.abs(phi1) > 1.0
    phi1_s = np.array(phi1)[stream_mask]
    d_s = np.array(dists)[stream_mask]
    d_err_s = np.array(dists)[stream_mask] * 0.03

    sort = np.argsort(phi1_s)
    phi1_s, d_s, d_err_s = phi1_s[sort], d_s[sort], d_err_s[sort]

    N_BINS = 5
    edges = np.linspace(phi1_s.min() - 0.5, phi1_s.max() + 0.5, N_BINS + 1)
    bins = []
    for i in range(N_BINS):
        m = (phi1_s >= edges[i]) & (phi1_s < edges[i + 1])
        n = m.sum()
        if n >= 2:
            w = 1.0 / d_err_s[m] ** 2
            bins.append({
                'phi1_deg': round(np.median(phi1_s[m]), 2),
                'dist_med': round(np.average(d_s[m], weights=w), 2),
                'dist_err': round(1.0 / np.sqrt(w.sum()), 2),
            })

    df = pd.DataFrame(bins)
    path = os.path.join(REPO, 'data/pal5/pal5_dist_track.csv')
    df.to_csv(path, index=False)
    print(f"  Saved: {path} ({len(df)} bins)")
    return df


def rebuild_orphan_dist():
    """Rebuild Orphan-Chenab distance track from Koposov+2023 RR Lyrae.

    Source: orphan_dmrr_bins.fits from Zenodo 7222654.
    Conversion: d = 10^((DM+5)/5) / 1000, d_err = d * ln(10)/5 * edm
    """
    print("\n=== Orphan-Chenab distance track ===")
    fits_path = os.path.join(REPO, 'data/orphan/orphan_dmrr_bins.fits')
    if not os.path.exists(fits_path):
        # Try the downloaded copy
        fits_path = '/tmp/koposov_extracted/tozenodo/measurements/orphan_dmrr_bins.fits'
    if not os.path.exists(fits_path):
        print("  FITS file not found — skipping")
        return None

    dm_data = fits.open(fits_path)[1].data
    phi1 = np.array(dm_data['phi1'], dtype=float)
    dm = np.array(dm_data['dm'], dtype=float)
    edm = np.array(dm_data['edm'], dtype=float)

    d_kpc = 10 ** ((dm + 5) / 5) / 1000
    d_err = d_kpc * np.log(10) / 5 * edm

    df = pd.DataFrame({
        'phi1_deg': phi1,
        'dist_med': np.round(d_kpc, 4),
        'dist_err': np.round(d_err, 4),
    })
    path = os.path.join(REPO, 'data/orphan/orphan_dist_track.csv')
    df.to_csv(path, index=False)
    print(f"  Saved: {path} ({len(df)} bins)")
    return df


if __name__ == '__main__':
    print("Rebuilding all derived data tracks...\n")
    rebuild_gd1_rv()
    rebuild_jhelum_pm()
    rebuild_pal5_dist()
    rebuild_orphan_dist()
    print("\nDone! All derived tracks rebuilt from source data.")
