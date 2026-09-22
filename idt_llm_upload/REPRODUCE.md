# Reproduction map

Each manuscript display item and reported number, with the script that produces it
and the data it consumes.

| Display item / number | Script | Data | Status |
|---|---|---|---|
| Table 1 | — editorial | — | n/a |
| Figure 1 — loop schematic | `pending` | — | source unconfirmed |
| Figure 2a,b — Omega at W=1 | `pending` | `results/closure/omega_w1_*.csv` | **code missing** |
| Figure 3 — band occupancy by W | `pending` | `results/closure/sweep_c*.csv` | **code missing** |
| Figure 4a–d — contribution | `scripts/fig4_v5.py` | `results/contribution/c1_nulls.csv`, `c3_nulls.csv` | complete |
| Figure 5a–c — controls | `scripts/fig5.py` | `results/contribution/c1_controls.csv`, `c2_controls.csv`, `c1_cos.csv` | complete |
| C1 clears 25 / 36 / 24 | `scripts/run_c1.py` | `results/contribution/c1_nulls.csv` | complete |
| C2 four pooled cells clear | `scripts/run_c2.py`, `run_c2b.py` | `results/contribution/c2_nulls*.csv` | complete |
| C3 clears 620 / 784 / 143 | `scripts/run_c3.py` | `results/contribution/c3_nulls.csv` | complete |
| C3 eligibility 2,171 of 2,502 | `scripts/prep_c3b.py`, `src/c3_pairing_audit.py` | `data/c3/c3_manifest.csv` | complete |
| Control reductions 92.0 / 86.8 / 90.0% | `scripts/controls.py`, `controls_c2.py` | `results/contribution/c1_controls.csv`, `c2_controls.csv` | complete |
| Control cosine 0.53 / 0.48 / 0.37 | `scripts/cos_c1.py`, `controls_c2.py` | `results/contribution/c1_cos.csv` | complete |
| Calibration long, type-I 0.045 | `scripts/calib.py` | `results/calibration/calibration.csv` | complete |
| Calibration short, type-I 0.028 | `scripts/calib_short.py` | `results/calibration/calibration_short.csv` | complete |
| Length / prefix analysis | `scripts/prefix.py` | `results/contribution/prefix.csv` | complete |
| Omega counts, 57 of 58 and 1,025 vs 1,015 | `pending` | `results/closure/omega_counts.txt` | **code missing** |
| Window bands W=2 … W=50 | `pending` | `results/closure/sweep_summary.txt` | **code missing** |
| D1 bound verification | `src/verify_d1_proof.py`, `src/verify_band_scaling.py` | analytic | complete |
| Estimator verification | `src/verify_estimator.py` | analytic | complete |
| C2 generation pipeline | `src/generate.py`, `seam_log.py`, `seam_run.py`, `pilot_gate.py` | — | complete |

Rows marked `pending` are the closure driver and plotting scripts; the closure readings they
produced are included in `results/closure/` and are the values reported in the manuscript.
