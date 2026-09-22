# C3 — PhotoBook

The PhotoBook dialogue corpus is not redistributed here.

`c3_manifest.csv` lists all 2,502 candidate games with `n_utt`, `n_turns` and
`drop_reason`, giving the complete record of the 2,171 games analysed and the
331 excluded.

`scripts/prep_c3.py` fetches the public corpus and `scripts/prep_c3b.py` rebuilds
the paired turn files used by `scripts/run_c3.py`.
