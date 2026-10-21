# Design: early warning for offshore oil wells (Petrobras 3W 2.0.0)

This document records what the system is, why each design decision was taken, what it costs,
what the evidence says, and what would change before it ran next to a real control room.
Numbers come from the reports in `reports/` and can be regenerated with the scripts listed in the
README.

## 1. Problem

A production operator watches many wells. Some undesirable events (a spurious DHSV closure, a
hydrate forming in the production line, scaling in the choke) build up over minutes to days
before they are *established* and cost production. The system's job is to raise an alert
**before establishment** without crying wolf.

That defines the metrics. Accuracy is not one of them:

| Metric | Definition | Why |
|---|---|---|
| Warned before established | Share of events with an observable onset whose first alert comes before the steady-state label (10 min grace before onset) | The operator's question: "did I hear about it in time?" |
| Lead time | Establishment time minus first alert, minutes | How much time the operator got |
| False alarms per well-day | Alert episodes on normal-operation data per day of normal exposure, exact Poisson CI | Alarm fatigue is per well and per day, not per sample |
| Time in alarm | Share of normal time spent in alarm | A never-clearing alarm is one "episode" but is useless |
| Recall | Share of all scorable event instances detected at all | Includes events with no observable onset |

The operating point is fixed in advance, not tuned on test data: **at most 1 false alarm per
well-week (0.143 per well-day) and at most 1% time in alarm**, both set on training wells only.

The second question is the one an operator with no labeled history actually faces: *can
simulated events plus my own wells' normal history replace labeled real events?* That is
protocol **P2**, the headline protocol.

## 2. Data facts that drove the design

Measured in `reports/data_audit.md`; each one forced a decision below.

- **40 real wells, but labeled normal operation exists for only 9 of them** (116.0 de-duplicated
  well-days). Every false-alarm number rests on those 9 wells.
- **Real instances overlap.** Each begins with a 1 h unlabeled lead-in that duplicates the end of
  the previous instance: 140.7 naive normal well-days vs 116.0 de-duplicated. Instances are
  stitched per well and overlaps removed, or exposure is double-counted.
- **Simulated and hand-drawn data carry only 5 sensors** (P-PDG, P-TPT, T-TPT, P-MON-CKP,
  T-JUS-CKP). Any protocol that mixes sources can only use those 5.
- **Real sensors are often broken; simulated ones never are.** P-PDG is usable in 18% of real
  normal samples and 100% of simulated ones. A classifier that sees only *which sensors work*
  separates events from normal at ROC-AUC 0.986 on a random split and 0.740 grouped by well: a
  shortcut, and a leak if splits are not grouped by well.
- **Historian artifacts.** Healthy P-TPT/T-TPT are often held for ~601 s between updates (so a
  "frozen" rule must wait longer than that); fill values such as P-PDG = 1.18e42 Pa and −999 °C
  appear; all 20 hand-drawn instances store pressure in bar although `dataset.ini` says Pa.
- **Labels are incomplete.** 43 of 57 real class-9 instances contain no event label; classes 3
  and 4 have no transient, so their onset is unobservable and lead time is undefined.
- **Real normal operation is far busier than simulated normal operation**: median |5-min P-TPT
  level| is 0.336 vs 0.014 robust-scale units.

## 3. Architecture

```
historian, 1 Hz ──► sensor health ──► 10 s causal features ──► detector score ──► percentile vs ──► hysteresis ──► status
 (5 sensors)        OK / MISSING /     79 per row: level,        physics rules      this site's        alerter        NORMAL / PENDING /
                    FROZEN / INVALID   change, variability vs    (primary) and      normal history     60 s on,       ALERT / CLEARING /
                                       a lagged 6 h baseline     sim-trained GBDT   (training wells)   300 s off      DEGRADED (+ reason)
```

One code path serves evaluation and the live service. `features.build_features` (batch) and
`stream.FeatureStream` (one reading at a time, bounded state) are tested equal row by row;
`alerting.HysteresisAlerter.step` is the same function in both. The demo
(`app/demo.py`) runs the streaming form on held-out wells and reproduces the batch evaluation's
outcome for 12 of 12 detector-event pairs, to the minute.

## 4. Decisions and trade-offs

| Decision | Alternative rejected | Why | What it costs |
|---|---|---|---|
| Only the 5 shared sensors | All 27 real sensors | Every protocol, including sim-only, must see the same inputs | Events that act on other sensors are invisible: class 9 (service-line hydrate) acts on gas-lift sensors and no detector can see it |
| Features relative to the signal's own lagged 6 h baseline, never absolute values | Raw levels | Absolute pressure identifies the well, and the well identifies the label | Needs ≥30 min of usable history plus a 30 min lag before most features exist: the first hour after every data gap is half-blind |
| Scale floors (0.5 bar, 0.5 °C) | Pure robust scale | Simulated steady states are noise-free, so a change would score ~80× higher in simulation than in a real well | Before a baseline exists the scale *is* the floor; see §6, issue 1 |
| Epoch-aligned 10 s blocks, emitted only when complete | Rows aligned to segment start | A live service with a trailing buffer produces identical rows | 10 s of added latency, irrelevant against lead times of minutes to hours |
| Frozen = unchanged for 1 h, real data only | 10 min | Healthy sensors are historian-held for ~601 s | A genuinely stuck sensor looks healthy for up to an hour (the demo shows this live) |
| Well-grouped folds (5) | Random or instance split | The missingness leak (0.986 vs 0.740) and overlapping instances | Fewer, larger, noisier folds: CIs are wide |
| Thresholds from training-normal percentiles, two budgets, out-of-fold for anything that learns | Tuning on test wells; one budget | Test wells stay untouched; a never-clearing alarm gamed the episode budget in the first Phase 2 run; in-sample calibration let a learned scorer look quiet on training wells (0.1% in alarm) and alarm 50% of the time on test wells | The threshold may not transfer: test false alarms land above the training budget (§5) |
| **Physics rules as the primary detector**, sim-trained GBDT alongside | GBDT as primary | Measured: rules beat every learned variant under P2 (paired −0.29 [−0.50, −0.09] for the main GBDT) and degrade more slowly when sensors fail | Rules cover only the event types someone wrote a rule for; flow instability (class 4) is almost never caught |
| Missingness-matched sensor dropout for learned models | Train on simulated rows as they are | Removes "which sensors work" as a label shortcut: +0.19 [+0.05, +0.33] warned | Moves reliance rather than removing it: survives losing P-PDG, collapses without T-TPT (Phase 4) |
| Hysteresis: 60 s to raise, 300 s below a lower threshold to clear, dead-band one decade | Raw threshold | A raw threshold on a noisy score flickers; each flicker is a page | Long alarms: the dead-band can hold a false alarm for hours (§6, issue 2) |
| DEGRADED when fewer than 2 sensors are usable; display also says DEGRADED when the detector cannot score | Report NORMAL | "I can't see" must never look like "all clear". Phase 4 found the count rule alone leaves the rules silently blind 11-31% of the time with 2-3 sensors missing | Display rule only: alert behaviour is unchanged, so the evaluated numbers stand |
| Thresholds shipped as an exact tail of the calibration distribution | Full sorted array; a fitted parametric tail | Alert decisions only compare against percentiles ≥ 0.9, so keeping the top 15% exactly gives identical decisions at 2.9 MB for both detectors | Percentiles below the tail are approximate (display only) |

## 5. Results (P2: simulated events + the site's own normal history)

| Detector | Warned before established [95% CI] | Recall | FA / well-day, official [95% CI] | FA / well-day, operator-visible | Time in alarm |
|---|---|---|---|---|---|
| Physics rules (primary) | **0.58** [0.41, 0.76] | 0.22 | 0.164 [0.099, 0.256] | 0.198 [0.126, 0.298] | 0.8% |
| GBDT, simulation-trained | 0.42 [0.24, 0.61] | 0.23 | 0.216 [0.140, 0.318] | not recomputed | 0.3% |
| GBDT, main model | 0.29 [0.16, 0.44] | 0.08 | 0.009 [0.000, 0.048] | not recomputed | 0.0% |

48 events have a defined lead time; CIs are well-cluster bootstrap (2,000) and exact Poisson.
Median lead for the rules is 179 min. Supporting results:

- **Sim-only calibration fails on real wells** (P1): rules 1.88 and z-score 2.85 false alarms per
  well-day, 13-20× the budget. Site normal history is required.
- **Sensor failures make the system quieter, not noisier** (Phase 4): with 0-3 sensors missing the
  rules warn for 0.58 / 0.48 / 0.36 / 0.19 of events and false alarms fall from 0.164 to 0.030.
- **The demo is the evaluated system**: streaming equals batch to the last stored digit (6 of
  1.26 M feature values on a real well differ by one rounding unit); 603 µs per reading with both
  detectors, ~1,650× real time per well on one laptop core.

## 6. Known issues (found by the demo, measured in `reports/phase6_followups.md`)

**Issue 1: alarms in the first hour after a data gap.** Until a baseline exists the oscillation
rule divides a 30-min standard deviation by the 0.5 bar floor, so ordinary choke-pressure noise
scores high. 4 of the rules' 23 operator-visible false alarms start exactly 15.7 min after data
resumes (the moment the 30-min window first has enough data, plus the 60 s raise delay).

**Issue 2: the official false-alarm metric did not count them.** It counts an episode only if
it *starts* on normal-labeled data, and every instance opens with an unlabeled hour. Counting
what an operator would see, the rules raise **0.198** false alarms per well-day, not 0.164.
The ranking of detectors does not change; the headline number was optimistic by 21%.

**Fix, evaluated as a separate variant** (`baselines.PhysicsRulesWarmup`): the oscillation rule
scores only once that signal's baseline exists.

| | Warned | Recall | Visible FA / well-day | Time in alarm |
|---|---|---|---|---|
| Physics rules (as evaluated) | 0.58 | 0.22 | 0.198 | 0.76% |
| Physics rules, warm-up fix | 0.58 | 0.21 | **0.164** | 0.56% |

All 4 post-gap alarms disappear; no early warning is lost or gained (paired difference +0.00);
4 detections without lead credit are lost (classes 3, 4, 4, 5: mostly oscillation events whose
recording opens mid-event). Recommended for deployment.

**Issue 3: the dead-band holds alarms for hours.** The clearing threshold sits one decade below
the alarm threshold (for the rules on fold 1: 1 in 25 vs 1 in 251 normal readings). Holding the alarm
threshold fixed:

| Clearing threshold | Visible FA / well-day | Median alarm | 90th pct | Longest | Share of FA time in alarms > 1 h | Time in alarm | Warned |
|---|---|---|---|---|---|---|---|
| One decade below (as evaluated) | 0.198 | 35 min | 141 min | 358 min | 60% | 0.76% | 0.58 |
| Half a decade below | 0.207 | 24 min | 57 min | 161 min | 32% | 0.45% | 0.58 |
| No dead-band | 0.216 | 14 min | 52 min | 137 min | 36% | 0.32% | 0.58 |

Half a decade cuts the longest alarm by more than half and time in alarm by 40% for 5% more alarm
episodes, with no change in early warnings. Recommended, after re-running threshold calibration
with it (the table holds the alarm threshold fixed, so the training budgets were not re-checked).

## 7. Limitations and threats to validity

- **Small samples.** 48 events with a defined lead time, from few wells; the 95% CI on the
  headline spans 0.41-0.76. False alarms come from 9 wells, and all 23 come from 2 of them
  (WELL-00001: 17, WELL-00019: 6). This is a noisy-well problem more than a fleet problem, and
  9 wells cannot tell us how common noisy wells are.
- **Recall is low (0.22).** It is dominated by 343 flow-instability instances (class 4) that no
  detector catches: they are absent from simulation and have no observable onset. The system
  warns early on the events it was built for; it is not a general anomaly detector.
- **The threshold does not fully transfer.** Calibrated for ≤ 0.143 false alarms per well-day on
  training wells, the rules run at 0.164 (0.198 visible) on held-out wells. A real deployment
  needs per-well recalibration and monitoring (§8).
- **Labels are the dataset's.** Onset and establishment times are expert annotations; lead time
  is measured against them. 43 of 57 class-9 instances have none and are excluded.
- **One operator, one basin, historical data.** No claim transfers to other fields untested.

## 8. What changes in a real deployment

**Data and integration**
- Read from the plant historian or an OPC UA gateway on a one-way link from the OT network; never
  write back. Handle late and out-of-order data explicitly (the stream rejects time going
  backwards today; production needs a short reorder buffer) and treat clock jumps as gaps.
- Watch the feed itself: no data for N minutes is its own DEGRADED reason, distinct from bad
  sensors. Today a gap silently resets state.

**Per-well onboarding and calibration**
- A new well has no normal history, so no threshold: run it in *commissioning* (scores computed,
  shown as DEGRADED, no alerts) until it has enough normal data, then calibrate on that well's
  own history rather than a pooled population. All false alarms here came from 2 wells, which is
  exactly what per-well thresholds address.
- Apply the warm-up fix and the half-decade dead-band (§6), re-calibrated.

**Alarm management**
- Integrate with the site's alarm philosophy (ISA-18.2 / EEMUA 191): acknowledge, shelve, and a
  priority class for advisory alerts, so these never compete with safety alarms. Track standing
  (stale) alarms as a KPI; issue 3 is exactly that.
- Every alert carries its reason. The rules provide it natively (which rule fired and on which sensors,
  e.g. flow_drop: P-TPT and T-TPT both well below their baselines); this is a large practical advantage of rules over the learned model, and should be
  surfaced in the UI (the demo shows scores, not yet the firing rule).
- The system is **advisory**. It is not a safety instrumented function and must never be wired
  into one.

**Monitoring**
- Per well: alert rate against its Poisson expectation, time in alarm, share of time DEGRADED,
  share of time each detector cannot score, sensor-health rates.
- Drift: compare the live score distribution with the calibration distribution weekly; a shift
  in normal-time percentiles means the threshold no longer means what it meant.
- Operator feedback on each alert (useful / not useful / real event) becomes the labeled data
  this dataset lacks. Route it into the next calibration, not straight into the model.

**Rollout**
- Shadow mode first (scored and logged, nobody paged) for several weeks on the pilot wells;
  compare against what operators actually saw. Then a small pilot with named operators, then
  expand well by well.

**Engineering**
- Ship versioned artifacts with their environment. Pickles are tied to library versions (the
  demo pins them); production should export to a portable format (e.g. ONNX for the GBDT; the
  rules are just code and a threshold table) and record the training data hash, feature config
  key and calibration date with each threshold.
- Capacity is not a constraint: ~600 µs per reading per well means one core serves hundreds of
  wells at 1 Hz. Run one process per group of wells, with a heartbeat, and restart from a
  persisted ring buffer, or from 7.5 h of historian data, which reproduces the feature state
  exactly (tested); alerter state restarts clear.
- Keep the guarantees as CI tests: causality (prefix invariance), streaming equals batch,
  thresholds' decisions equal to the full calibration, and the demo episodes' outcomes.

## 9. Reproducibility

- All randomness is seeded; two independent full Phase 3 runs gave byte-identical reports, and
  the Phase 4 study reproduced exactly on a second machine.
- 96 tests: causality, streaming equals batch, no test-well data in training or calibration,
  injected failures equal a rebuild from raw data, compact thresholds equal the full calibration,
  the app replays headless and shows DEGRADED when blind.
- Nothing evaluated in Phases 2-5 was changed after the fact. Corrections and fixes in Phase 6 are
  additional metrics and separately named variants.
