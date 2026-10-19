# Oil Well Early-Warning System (Petrobras 3W 2.0.0)

Warns a production operator **before** an undesirable event in an offshore oil well
becomes established. It is scored on lead time and false alarms per well-day, never
accuracy, and tested on the question an operator actually asks: *can simulated events
plus my own wells' normal history replace labeled real events?*

> Status: Phase 0 complete (data audit + core library + tests). Results tables land in Phase 2-3.

## What is here

| Path | Purpose |
|---|---|
| `src/ew3w/schema.py` | 3W constants, sensor groups, physical plausibility ranges |
| `src/ew3w/data.py` | Instance discovery/loading, per-well timeline stitching with overlap de-duplication |
| `src/ew3w/quality.py` | Causal sensor health: OK / MISSING / FROZEN / INVALID |
| `src/ew3w/alerting.py` | Hysteresis alerter (one `step()` for offline eval and streaming) |
| `src/ew3w/metrics.py` | Event-level metrics: delay, lead time, censoring, false alarms per well-day (exact Poisson CI), well-cluster bootstrap |
| `src/ew3w/splits.py` | Well-grouped folds and protocols P1-P4 (sim-only to real+sim) |
| `scripts/audit_dataset.py` | Reproduces every number in the plan; writes `reports/data_audit.md` |
| `tests/test_core.py` | 24 unit tests for the above |
| `tests/test_real_data.py` | 4 integration tests on the real dataset (set `EW3W_DATASET`) |

## Setup (Windows, PowerShell)

```powershell
git clone --depth 1 https://github.com/petrobras/3W.git C:\data\3W   # ~1.9 GB dataset
cd oilwell-early-warning
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q
$env:EW3W_DATASET = "C:\data\3W"   # where YOUR 3W clone is: repo root or its dataset folder
python -m pytest -q tests/test_real_data.py
python scripts/audit_dataset.py   # uses EW3W_DATASET; ~2-5 min
```

## Key audit findings (from `reports/data_audit.md`)

- Simulated/drawn instances carry only 5 sensors; P-PDG is healthy in 18% of real normal samples but 100% of synthetic ones.
- A random forest that sees **only which sensors are broken** reaches ROC-AUC 0.986 under a random split and 0.740 grouped by well. All splits are grouped by well, and 0.740 is a baseline to beat.
- Real instances overlap their predecessor by exactly the 1-hour unlabeled lead-in: 116.0 de-duplicated normal well-days vs 140.7 naive.
- Many healthy P-TPT/T-TPT series are held for ~601 s between historian updates, so the frozen rule uses 1 hour (and runs on real data only).
- Historian fill values (P-PDG = 1.18e42 Pa, -999 °C) are flagged INVALID.
- 43 of 57 real class-9 instances contain no event label and are excluded from scoring.

## Data licence

3W Dataset by Petrobras, CC BY 4.0: https://github.com/petrobras/3W
