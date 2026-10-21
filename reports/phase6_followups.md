# Phase 6: follow-ups to the Phase 5 findings

P2, all 5 folds, 116.0 normal well-days, 48 events with a defined lead. Same harness and budgets as Phases 2-4; nothing evaluated earlier is changed.

## 1. False alarms an operator would see

The official metric (`metrics.count_false_alarms`) counts an alarm only if it **starts** on a normal-labeled sample. Every real instance opens with an unlabeled hour, so an alarm raised there and still active when the normal labels begin was not counted. Recounting every alarm that overlaps normal-labeled data:

| Detector | Official FA / well-day [95% CI] | Visible FA / well-day [95% CI] | Visible alarms | Start < 1 h after data resumes | Wells with any (of wells with normal data) |
|---|---|---|---|---|---|
| physics_rules | 0.164 [0.099, 0.256] | 0.198 [0.126, 0.298] | 23 | 4 | 2 of 9 |
| physics_rules_warmup | 0.164 [0.099, 0.256] | 0.164 [0.099, 0.256] | 19 | 0 | 2 of 9 |

## 2. Warm-up fix: no oscillation scoring before a baseline exists

| Detector | Warned before established | Recall | Visible FA / well-day | Time in alarm |
|---|---|---|---|---|
| physics_rules | 0.58 (n=48) | 0.22 | 0.198 | 0.76% |
| physics_rules_warmup | 0.58 (n=48) | 0.21 | 0.164 | 0.56% |

Paired on the same events, warmup − original: **+0.00 [+0.00, +0.00]** (48 events; only warmup warned: 0, only original warned: 0).

Event instances whose outcome changes at all: **6** of 482.

Detections lost: **4** (classes 3, 4, 4, 5), gained: 0; early warnings lost: **0**. The remaining changes move a lead time by at most 10 s. So the post-gap scoring never earned an early warning; it did produce 4 detections without lead credit (mostly classes 3/4, oscillation events whose recording opens mid-event), and those are the price of the fix.

| instance_id                 |   event_class | well       | detected   | detected_warmup   | warned_before_established   | warned_before_established_warmup   |   lead_s |   lead_s_warmup |
|:----------------------------|--------------:|:-----------|:-----------|:------------------|:----------------------------|:-----------------------------------|---------:|----------------:|
| 3/WELL-00014_20171028070000 |             3 | WELL-00014 | True       | False             | False                       | False                              |      nan |             nan |
| 7/WELL-00001_20170226130146 |             7 | WELL-00001 | True       | True              | True                        | True                               |    15300 |           15290 |
| 4/WELL-00001_20170316110203 |             4 | WELL-00001 | True       | False             | False                       | False                              |      nan |             nan |
| 5/WELL-00016_20180426131710 |             5 | WELL-00016 | True       | False             | False                       | False                              |      nan |             nan |
| 6/WELL-00002_20140325160304 |             6 | WELL-00002 | True       | True              | False                       | False                              |     -550 |            -560 |
| 4/WELL-00005_20170625083000 |             4 | WELL-00005 | True       | False             | False                       | False                              |      nan |             nan |

## 3. Dead-band: how far below the alarm threshold must the score fall to clear?

Original physics rules; on-threshold held at each fold's calibrated value, only the off-threshold changes.

| Off-threshold | Visible FA / well-day | Median alarm, min | 90th pct, min | Longest, min | Share of FA time in alarms > 1 h | Time in alarm | Warned |
|---|---|---|---|---|---|---|---|
| one decade (as evaluated) | 0.198 | 35 | 141 | 358 | 60% | 0.76% | 0.58 |
| half a decade | 0.207 | 24 | 57 | 161 | 32% | 0.45% | 0.58 |
| none | 0.216 | 14 | 52 | 137 | 36% | 0.32% | 0.58 |

Per-alarm detail (both detectors): `reports/phase6_false_alarms.csv`.
