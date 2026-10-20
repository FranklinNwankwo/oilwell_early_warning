# Phase 3: main model

Gradient-boosted trees on the 63 value features, with missingness-matched sensor dropout (`src/ew3w/model.py`). Same harness as Phase 2: thresholds chosen on training wells only (<= 0.143 false alarms per well-day AND <= 1% time in alarm, out-of-fold scores), then frozen and applied to the held-out wells. 95% CIs: well-cluster bootstrap (2000) and exact Poisson.

| Model | Protocol | Warned [95% CI] | n | Recall | Lead, min [95% CI] | FA / well-day [95% CI] | Time in alarm | Class 4 recall |
|---|---|---|---|---|---|---|---|---|
| gbdt | P1_sim_only | 0.98 [0.92, 1.00] | 48 | 0.99 | 188 [96, 759] | 0.362 [0.261, 0.490] | 88.8% | 1.00 |
| gbdt | P2_sim_plus_site_normal | 0.29 [0.16, 0.44] | 48 | 0.08 | 130 [44, 1855] | 0.009 [0.000, 0.048] | 0.0% | 0.01 |
| gbdt | P3_real | 0.38 [0.22, 0.56] | 48 | 0.07 | 103 [42, 1361] | 0.121 [0.066, 0.203] | 0.2% | 0.01 |
| gbdt | P4_real_plus_sim | 0.40 [0.23, 0.57] | 48 | 0.07 | 161 [40, 1341] | 0.569 [0.440, 0.724] | 3.5% | 0.00 |
| gbdt_sim_trained | P2_sim_plus_site_normal | 0.42 [0.24, 0.61] | 48 | 0.23 | 175 [62, 1366] | 0.216 [0.140, 0.318] | 0.3% | 0.12 |
| gbdt_no_dropout | P2_sim_plus_site_normal | 0.10 [0.00, 0.25] | 48 | 0.16 | 53 [21, 532] | 0.129 [0.072, 0.213] | 0.8% | 0.09 |
| physics_rules | P2_sim_plus_site_normal | 0.58 [0.41, 0.76] | 48 | 0.22 | 179 [72, 1370] | 0.164 [0.099, 0.256] | 0.8% | 0.09 |

## Head-to-head on the same events (P2)

Difference in warned-before-established on the same event instances, well-cluster bootstrap CI.

| Comparison | Difference [95% CI] | Events | Only A warned | Only B warned |
|---|---|---|---|---|
| gbdt − physics_rules | -0.29 [-0.50, -0.09] | 48 | 3 | 17 |
| gbdt − gbdt_no_dropout | +0.19 [+0.05, +0.33] | 48 | 11 | 2 |
| gbdt_sim_trained − physics_rules | -0.17 [-0.35, +0.02] | 48 | 4 | 12 |
| gbdt_sim_trained − gbdt | +0.12 [-0.04, +0.31] | 48 | 10 | 4 |

## Sim-to-real gap

A classifier separating simulated from real normal rows, grouped by well/instance: ROC-AUC **0.990** on rows as they are, **0.823** after giving simulated rows real sensor-availability patterns. The second number is the gap in the sensor VALUES themselves.

## Per-class results, gbdt under P2

|       |   n_instances |   n_wells |   event_recall |   n_uncensored |   warned_before_established |   median_delay_min |   median_lead_min |   pre_onset_alerts_per_inst |
|:------|--------------:|----------:|---------------:|---------------:|----------------------------:|-------------------:|------------------:|----------------------------:|
| 1     |             4 |         3 |           0    |              4 |                        0    |             nan    |            nan    |                        0    |
| 2     |            22 |         7 |           0.32 |             11 |                        0.36 |               7.5  |             83.5  |                        0.05 |
| 3     |            32 |         2 |           0.53 |              0 |                      nan    |             nan    |            nan    |                        0    |
| 4     |           343 |         7 |           0.01 |              0 |                      nan    |             nan    |            nan    |                        0    |
| 5     |            11 |         3 |           0.55 |              7 |                        0.71 |              12.92 |             52.67 |                        0    |
| 6     |             6 |         2 |           0    |              6 |                        0    |             nan    |            nan    |                        0    |
| 7     |            36 |         6 |           0    |              5 |                        0    |             nan    |            nan    |                        0    |
| 8     |            14 |         9 |           0.36 |             12 |                        0.42 |               9.83 |           1926.5  |                        0    |
| 9     |            14 |         7 |           0    |              3 |                        0    |             nan    |            nan    |                        0    |
| all   |           482 |        33 |           0.08 |             48 |                        0.29 |               9.25 |            129.5  |                        0    |
| macro |           nan |       nan |           0.2  |            nan |                        0.21 |             nan    |            nan    |                      nan    |

## Operating curve on test folds, gbdt P2 (reference only, not used to choose thresholds)

|   on_percentile |   warned_before_established |   event_recall |   fa_per_well_day |
|----------------:|----------------------------:|---------------:|------------------:|
|          0.99   |                      0.4792 |         0.1639 |            0.2673 |
|          0.9968 |                      0.3333 |         0.0996 |            0.0604 |
|          0.999  |                      0.3125 |         0.083  |            0.0086 |
|          0.9997 |                      0.2708 |         0.0643 |            0      |
|          0.9999 |                      0.2708 |         0.0539 |            0      |
|          1      |                      0.2292 |         0.0456 |            0      |
|          1      |                      0.1875 |         0.0353 |            0      |
|          1      |                      0.1667 |         0.0311 |            0      |
|          1      |                      0.1458 |         0.027  |            0      |

## Does the training-side threshold carry over to unseen wells?

Per fold: false alarms and time in alarm on the training wells (where the threshold was set, out of fold) next to the same quantities on that fold's held-out wells.

| model            | protocol                |   fold |   on_percentile |   train_fa_per_day |   test_fa_per_day |   train_in_alarm_pct |   test_in_alarm_pct | meets_budget   |
|:-----------------|:------------------------|-------:|----------------:|-------------------:|------------------:|---------------------:|--------------------:|:---------------|
| gbdt             | P1_sim_only             |      0 |        0.996838 |              0.127 |             0.764 |                 0.43 |               96.68 | True           |
| gbdt             | P1_sim_only             |      1 |        0.996838 |              0.127 |             0.052 |                 0.43 |               99.97 | True           |
| gbdt             | P1_sim_only             |      2 |        0.996838 |              0.127 |             0.285 |                 0.43 |               99.1  | True           |
| gbdt             | P1_sim_only             |      3 |        0.996838 |              0.127 |             0     |                 0.43 |               12.17 | True           |
| gbdt             | P1_sim_only             |      4 |        0.996838 |              0.127 |             0.622 |                 0.43 |               99.32 | True           |
| gbdt             | P2_sim_plus_site_normal |      0 |        0.999206 |              0.141 |             0     |                 0.22 |                0    | True           |
| gbdt             | P2_sim_plus_site_normal |      1 |        0.999499 |              0.124 |             0     |                 0.13 |                0    | True           |
| gbdt             | P2_sim_plus_site_normal |      2 |        0.999499 |              0.108 |             0     |                 0.14 |                0    | True           |
| gbdt             | P2_sim_plus_site_normal |      3 |        0.999369 |              0.136 |             0.075 |                 0.2  |                0.15 | True           |
| gbdt             | P2_sim_plus_site_normal |      4 |        0.999602 |              0.102 |             0     |                 0.11 |                0    | True           |
| gbdt             | P3_real                 |      0 |        0.999602 |              0.13  |             0     |                 0.1  |                0    | True           |
| gbdt             | P3_real                 |      1 |        0.999206 |              0.124 |             0.726 |                 0.15 |                0.96 | True           |
| gbdt             | P3_real                 |      2 |        0.999    |              0.122 |             0     |                 0.26 |                0    | True           |
| gbdt             | P3_real                 |      3 |        0.999369 |              0.136 |             0     |                 0.17 |                0    | True           |
| gbdt             | P3_real                 |      4 |        0.999499 |              0.132 |             0     |                 0.15 |                0    | True           |
| gbdt             | P4_real_plus_sim        |      0 |        0.999    |              0.121 |             0     |                 0.31 |                0    | True           |
| gbdt             | P4_real_plus_sim        |      1 |        0.999602 |              0.101 |             0     |                 0.14 |                0    | True           |
| gbdt             | P4_real_plus_sim        |      2 |        0.998741 |              0.142 |             1.566 |                 0.42 |                9.66 | True           |
| gbdt             | P4_real_plus_sim        |      3 |        0.999684 |              0.127 |             0     |                 0.14 |                0    | True           |
| gbdt             | P4_real_plus_sim        |      4 |        0.999    |              0.139 |             0     |                 0.35 |                0    | True           |
| gbdt_sim_trained | P2_sim_plus_site_normal |      0 |        0.999499 |              0.119 |             0     |                 0.09 |                0    | True           |
| gbdt_sim_trained | P2_sim_plus_site_normal |      1 |        0.999499 |              0.114 |             1.193 |                 0.1  |                1.66 | True           |
| gbdt_sim_trained | P2_sim_plus_site_normal |      2 |        0.999499 |              0.135 |             0     |                 0.13 |                0    | True           |
| gbdt_sim_trained | P2_sim_plus_site_normal |      3 |        0.999206 |              0.127 |             0.075 |                 0.22 |                0.13 | True           |
| gbdt_sim_trained | P2_sim_plus_site_normal |      4 |        0.999206 |              0.142 |             0.057 |                 0.18 |                0.06 | True           |
| gbdt_no_dropout  | P2_sim_plus_site_normal |      0 |        0.997488 |              0.087 |             0.551 |                 0.92 |                3.28 | True           |
| gbdt_no_dropout  | P2_sim_plus_site_normal |      1 |        0.999    |              0.124 |             0     |                 0.35 |                0    | True           |
| gbdt_no_dropout  | P2_sim_plus_site_normal |      2 |        0.997488 |              0.122 |             0     |                 0.55 |                0    | True           |
| gbdt_no_dropout  | P2_sim_plus_site_normal |      3 |        0.996838 |              0.127 |             0.151 |                 0.54 |                0.99 | True           |
| gbdt_no_dropout  | P2_sim_plus_site_normal |      4 |        0.997488 |              0.102 |             0     |                 0.93 |                0    | True           |
| physics_rules    | P2_sim_plus_site_normal |      0 |        0.996838 |              0.108 |             0     |                 0.58 |                0    | True           |
| physics_rules    | P2_sim_plus_site_normal |      1 |        0.996019 |              0.114 |             0.778 |                 0.59 |                2.96 | True           |
| physics_rules    | P2_sim_plus_site_normal |      2 |        0.996838 |              0.068 |             0     |                 0.55 |                0    | True           |
| physics_rules    | P2_sim_plus_site_normal |      3 |        0.999206 |              0.117 |             0.301 |                 0.19 |                2.31 | True           |
| physics_rules    | P2_sim_plus_site_normal |      4 |        0.996838 |              0.122 |             0     |                 0.57 |                0    | True           |
