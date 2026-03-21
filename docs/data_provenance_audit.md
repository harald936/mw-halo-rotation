# Data Provenance Audit

The older claim that "all 317 data points are fully verified" was too strong.

The current, scripted audit is:

```bash
conda run -n mwgd1 python scripts/17_audit_final_run_inputs.py
```

That audit writes a fresh report to:

- `docs/final_run_input_audit.md`

Current high-level status:

- Most final-run inputs are either exactly reproducible from local source products or strongly source-backed by public catalogs.
- The fast signed production script is `scripts/16_run_final_signed_lmcfast.py`.
- The main remaining provenance caveat is the Jhelum PM augmentation: the current scripted Gaia DR3 rebuild does not exactly reproduce the stored PM bins.
- Orphan distance bins are source-backed by the Koposov+2023 Zenodo release, but the raw `orphan_dmrr_bins.fits` file is not present locally in this clone, so the audit may classify that channel as source-backed rather than exact.

Use `docs/final_run_input_audit.md` as the live provenance summary, not this placeholder note.
