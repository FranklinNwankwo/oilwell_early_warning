# Oil Well Early-Warning System (Petrobras 3W 2.0.0)

Warns a production operator **before** an undesirable event in an offshore oil well
becomes established. It is scored on lead time and false alarms per well-day, never
accuracy, and tested on the question an operator actually asks: *can simulated events
plus my own wells' normal history replace labeled real events?*

> Status: Phase 0 (data audit, core library) and Phase 1 (causal feature pipeline) complete. Results tables land in Phase 2-3.

## What is here

| Path | Purpose |
|---|---|
| `src/ew3w/schema.py` | 3W constants, sensor groups, physical plausibility ranges, unit fixes |
| `src/ew3w/data.py` | Instance discovery/loading, per-well timeline stitching with overlap de-duplication |
| `src/ew3w/quality.py` | Causal sensor health: OK / MISSING / FROZEN / INVALID |
| `src/ew3w/features.py` | Causal 10 s window features for the 5 shared sensors (Phase 1) |
| `src/ew3w/alerting.py` | Hysteresis alerter (one `step()` for offline eval and streaming) |
| `src/ew3w/metrics.py` | Event-level metrics: delay, lead time, censoring, false alarms per well-day (exact Poisson CI), well-cluster bootstrap |
| `src/ew3w/splits.py` | Well-grouped folds and protocols P1-P4 (sim-only to real+sim) |
| `scripts/audit_dataset.py` | Reproduces every number in the plan; writes `reports/data_audit.md` |
| `scripts/build_features.py` | Builds the feature cache under `data/features/<config key>/` |
| `tests/test_core.py` | Unit tests: data, health, alerter, metrics, folds |
| `tests/test_features.py` | Phase 1 gate: causality, streaming == batch, bad-sensor robustness |
| `tests/test_real_data.py` | Integration tests on the real dataset (set `EW3W_DATASET`) |

## Setup (Windows, PowerShell)

```powershell
git clone --depth 1 https://github.com/petrobras/3W.git C:\data\3W   # ~1.9 GB dataset
cd oilwell-early-warning
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
$env:EW3W_DATASET = "C:\data\3W"   # where YOUR 3W clone is: repo root or its dataset folder
python -m pytest -q                # 47 tests (6 of them read the real dataset)
python scripts/audit_dataset.py    # ~5-10 min on a laptop
python scripts/build_features.py   # ~10-20 min on a laptop, ~400 MB of disk
```

## Features (Phase 1)

One row every 10 s (epoch-aligned, emitted only when the block is complete), 79 features from the
5 sensors every source shares (P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP) plus two pressure
differences (P-PDG − P-TPT, P-TPT − P-MON-CKP):

- **Per signal, windows 1 / 5 / 30 min:** level vs a lagged 6 h robust baseline, change vs the previous window, variability. All in robust-scale units, never absolute values (absolute pressure identifies the well).
- **Scale floors** (0.5 bar, 0.5 °C): simulated steady states are noise-free, so without a floor a change would score ~80× higher in simulation than in a real well.
- **Per sensor:** health code, fraction usable in the last 5 min, staleness (seconds since last change, for historian-held sensors), plus the count of usable sensors.

Guarantees, enforced by `tests/test_features.py`: no feature uses data after its timestamp (prefix
invariance and future-perturbation tests), a live service keeping only `FeatureConfig.history_s`
(7.5 h) of data reproduces batch features, and state resets at data gaps.

## Key audit findings (from `reports/data_audit.md`)

- Simulated/drawn instances carry only 5 sensors; P-PDG is healthy in 18% of real normal samples but 100% of synthetic ones.
- A random forest that sees **only which sensors are broken** reaches ROC-AUC 0.986 under a random split and 0.740 grouped by well. All splits are grouped by well, and 0.740 is a baseline to beat.
- Real instances overlap their predecessor by exactly the 1-hour unlabeled lead-in: 116.0 de-duplicated normal well-days vs 140.7 naive.
- Many healthy P-TPT/T-TPT series are held for ~601 s between historian updates, so the frozen rule uses 1 hour (and runs on real data only).
- Historian fill values (P-PDG = 1.18e42 Pa, -999 °C) are flagged INVALID.
- **All 20 hand-drawn instances store pressure in bar**, although `dataset.ini` declares Pa; they are converted at load time.
- 43 of 57 real class-9 instances contain no event label and are excluded from scoring.
- Even after normalisation, real normal operation is far "busier" than simulated normal: median |P-TPT 5-min level| is 0.336 vs 0.014 robust-scale units. A threshold tuned on simulation alone will over-alarm on real wells; Phase 3's P1 vs P2 comparison measures exactly this.

## Data licence

3W Dataset by Petrobras, CC BY 4.0: https://github.com/petrobras/3W
