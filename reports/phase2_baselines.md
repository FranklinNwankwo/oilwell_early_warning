# Phase 2: baseline results

Out-of-fold over 5 well-grouped folds (every well tested once). Operating point chosen on TRAINING data only: lowest threshold with <= 0.143 false alarms per well-day (1 per well-week) AND <= 1% time in alarm on training normal data. Alerter: 60 s to raise, 300 s to clear, 10 s rows. Detection grace before onset: 10 min. 95% CIs: well-cluster bootstrap (2000 resamples) for event metrics, exact Poisson for false alarms.

**Warned** = share of events (onset observable, established phase recorded) whose first alert came before the event became established. **Recall** = share of all scorable events detected at all. **Lead** = median minutes from first alert to establishment.

| Baseline | Protocol | Warned [95% CI] | n | Recall | Lead, min [95% CI] | FA / well-day [95% CI] | Time in alarm | Class 4 recall |
|---|---|---|---|---|---|---|---|---|
| health_only | P1_sim_only | 0.00 [0.00, 0.00] | 48 | 0.00 | – [–, –] | 0.000 [0.000, 0.032] | 0.0% | 0.00 |
| health_only | P2_sim_plus_site_normal | 0.12 [0.00, 0.32] | 48 | 0.21 | 53 [-19, 532] | 0.147 [0.085, 0.235] | 1.3% | 0.13 |
| health_only | P3_real | 0.00 [0.00, 0.00] | 48 | 0.00 | – [–, –] | 0.026 [0.005, 0.076] | 1.0% | 0.00 |
| health_only | P4_real_plus_sim | 0.00 [0.00, 0.00] | 48 | 0.00 | – [–, –] | 0.009 [0.000, 0.048] | 0.0% | 0.01 |
| physics_rules | P1_sim_only | 0.90 [0.80, 0.98] | 48 | 0.77 | 175 [74, 1371] | 1.880 [1.639, 2.147] | 32.5% | 0.70 |
| physics_rules | P2_sim_plus_site_normal | 0.58 [0.41, 0.76] | 48 | 0.22 | 179 [72, 1370] | 0.164 [0.099, 0.256] | 0.8% | 0.09 |
| physics_rules | P3_real | 0.58 [0.41, 0.76] | 48 | 0.22 | 179 [72, 1370] | 0.164 [0.099, 0.256] | 0.8% | 0.09 |
| physics_rules | P4_real_plus_sim | 0.60 [0.44, 0.78] | 48 | 0.23 | 176 [66, 1314] | 0.276 [0.189, 0.390] | 1.1% | 0.11 |
| zscore | P1_sim_only | 0.96 [0.89, 1.00] | 48 | 0.79 | 183 [86, 1336] | 2.854 [2.555, 3.179] | 45.5% | 0.72 |
| zscore | P2_sim_plus_site_normal | 0.48 [0.26, 0.69] | 48 | 0.09 | 171 [53, 1775] | 0.129 [0.072, 0.213] | 0.4% | 0.00 |
| zscore | P3_real | 0.48 [0.26, 0.69] | 48 | 0.09 | 171 [53, 1775] | 0.129 [0.072, 0.213] | 0.4% | 0.00 |
| zscore | P4_real_plus_sim | 0.56 [0.37, 0.76] | 48 | 0.13 | 148 [62, 1789] | 0.181 [0.112, 0.277] | 0.5% | 0.01 |
| iforest | P1_sim_only | 0.73 [0.53, 0.90] | 48 | 0.55 | 297 [74, 1366] | 0.138 [0.079, 0.224] | 25.1% | 0.44 |
| iforest | P2_sim_plus_site_normal | 0.31 [0.14, 0.50] | 48 | 0.07 | 100 [44, 1422] | 0.138 [0.079, 0.224] | 0.5% | 0.01 |
| iforest | P3_real | 0.31 [0.14, 0.50] | 48 | 0.07 | 100 [44, 1422] | 0.138 [0.079, 0.224] | 0.5% | 0.01 |
| iforest | P4_real_plus_sim | 0.27 [0.10, 0.45] | 48 | 0.07 | 100 [32, 1205] | 0.103 [0.053, 0.181] | 0.7% | 0.00 |

Normal exposure: 116.0 well-days; scorable real events: 482.

## Per-class results, P2 (simulated events + site normal history)

### health_only

|       |   n_instances |   n_wells |   event_recall |   n_uncensored |   warned_before_established |   median_delay_min |   median_lead_min |   pre_onset_alerts_per_inst |
|:------|--------------:|----------:|---------------:|---------------:|----------------------------:|-------------------:|------------------:|----------------------------:|
| 1     |             4 |         3 |           0    |              4 |                        0    |                nan |            nan    |                        0    |
| 2     |            22 |         7 |           0    |             11 |                        0    |                nan |            nan    |                        0    |
| 3     |            32 |         2 |           1    |              0 |                      nan    |                nan |            nan    |                        0    |
| 4     |           343 |         7 |           0.13 |              0 |                      nan    |                nan |            nan    |                        0    |
| 5     |            11 |         3 |           0.45 |              7 |                        0.43 |                -10 |             48.42 |                        0    |
| 6     |             6 |         2 |           0    |              6 |                        0    |                nan |            nan    |                        0    |
| 7     |            36 |         6 |           0.19 |              5 |                        0    |               1627 |            nan    |                        0.06 |
| 8     |            14 |         9 |           0    |             12 |                        0    |                nan |            nan    |                        0    |
| 9     |            14 |         7 |           0.86 |              3 |                        1    |                -10 |            113.33 |                        0.29 |
| all   |           482 |        33 |           0.21 |             48 |                        0.12 |                -10 |             52.67 |                        0.01 |
| macro |           nan |       nan |           0.29 |            nan |                        0.2  |                nan |            nan    |                      nan    |

### physics_rules

|       |   n_instances |   n_wells |   event_recall |   n_uncensored |   warned_before_established |   median_delay_min |   median_lead_min |   pre_onset_alerts_per_inst |
|:------|--------------:|----------:|---------------:|---------------:|----------------------------:|-------------------:|------------------:|----------------------------:|
| 1     |             4 |         3 |           0.5  |              4 |                        0.25 |             329.67 |             77.92 |                        0    |
| 2     |            22 |         7 |           1    |             11 |                        1    |               1.92 |             83.33 |                        0.05 |
| 3     |            32 |         2 |           0.84 |              0 |                      nan    |             nan    |            nan    |                        0    |
| 4     |           343 |         7 |           0.09 |              0 |                      nan    |             nan    |            nan    |                        0    |
| 5     |            11 |         3 |           0.36 |              7 |                        0.29 |             -10    |             53    |                        0    |
| 6     |             6 |         2 |           0.33 |              6 |                        0.17 |              11.58 |              5.83 |                        0    |
| 7     |            36 |         6 |           0.06 |              5 |                        0.4  |             739.92 |            242.75 |                        0.03 |
| 8     |            14 |         9 |           0.71 |             12 |                        0.83 |               2.83 |           2347.08 |                        0.14 |
| 9     |            14 |         7 |           0.29 |              3 |                        0.33 |             162.42 |            253.67 |                        0.14 |
| all   |           482 |        33 |           0.22 |             48 |                        0.58 |               2.42 |            179.08 |                        0.01 |
| macro |           nan |       nan |           0.47 |            nan |                        0.47 |             nan    |            nan    |                      nan    |

### zscore

|       |   n_instances |   n_wells |   event_recall |   n_uncensored |   warned_before_established |   median_delay_min |   median_lead_min |   pre_onset_alerts_per_inst |
|:------|--------------:|----------:|---------------:|---------------:|----------------------------:|-------------------:|------------------:|----------------------------:|
| 1     |             4 |         3 |           0.25 |              4 |                        0.25 |              70    |             93.67 |                        0    |
| 2     |            22 |         7 |           0.91 |             11 |                        0.82 |               6    |             74.5  |                        0.18 |
| 3     |            32 |         2 |           0.22 |              0 |                      nan    |             nan    |            nan    |                        0    |
| 4     |           343 |         7 |           0    |              0 |                      nan    |             nan    |            nan    |                        0    |
| 5     |            11 |         3 |           0.45 |              7 |                        0.57 |             -10    |             48.42 |                        0.18 |
| 6     |             6 |         2 |           0    |              6 |                        0    |             nan    |            nan    |                        0    |
| 7     |            36 |         6 |           0.03 |              5 |                        0    |            6113    |            nan    |                        0    |
| 8     |            14 |         9 |           0.64 |             12 |                        0.75 |               6.33 |           1930    |                        0.29 |
| 9     |            14 |         7 |           0    |              3 |                        0    |             nan    |            nan    |                        0    |
| all   |           482 |        33 |           0.09 |             48 |                        0.48 |               6.17 |            170.67 |                        0.02 |
| macro |           nan |       nan |           0.28 |            nan |                        0.34 |             nan    |            nan    |                      nan    |

### iforest

|       |   n_instances |   n_wells |   event_recall |   n_uncensored |   warned_before_established |   median_delay_min |   median_lead_min |   pre_onset_alerts_per_inst |
|:------|--------------:|----------:|---------------:|---------------:|----------------------------:|-------------------:|------------------:|----------------------------:|
| 1     |             4 |         3 |           0    |              4 |                        0    |             nan    |            nan    |                        0    |
| 2     |            22 |         7 |           0.32 |             11 |                        0.55 |               2.17 |             72.33 |                        0.05 |
| 3     |            32 |         2 |           0.19 |              0 |                      nan    |             nan    |            nan    |                        0    |
| 4     |           343 |         7 |           0.01 |              0 |                      nan    |             nan    |            nan    |                        0    |
| 5     |            11 |         3 |           0.45 |              7 |                        0.43 |             -10    |             52.67 |                        0.09 |
| 6     |             6 |         2 |           0    |              6 |                        0    |             nan    |            nan    |                        0    |
| 7     |            36 |         6 |           0    |              5 |                        0    |             nan    |            nan    |                        0    |
| 8     |            14 |         9 |           0.43 |             12 |                        0.42 |              -0.75 |           1684    |                        1.29 |
| 9     |            14 |         7 |           0.29 |              3 |                        0.33 |             170.75 |            250.5  |                        0.21 |
| all   |           482 |        33 |           0.07 |             48 |                        0.31 |              -4.33 |            100    |                        0.05 |
| macro |           nan |       nan |           0.19 |            nan |                        0.25 |             nan    |            nan    |                      nan    |

## Operating curves on test folds, P2 (reference only, not used to choose thresholds)

### health_only

|   on_percentile |   warned_before_established |   event_recall |   fa_per_well_day |
|----------------:|----------------------------:|---------------:|------------------:|
|          0.99   |                       0.125 |         0.2095 |            0.1466 |
|          0.9968 |                       0.125 |         0.2095 |            0.1466 |
|          0.999  |                       0.125 |         0.2095 |            0.1466 |
|          0.9997 |                       0.125 |         0.2095 |            0.1466 |
|          0.9999 |                       0.125 |         0.2095 |            0.1466 |
|          1      |                       0.125 |         0.2095 |            0.1466 |
|          1      |                       0.125 |         0.2095 |            0.1466 |
|          1      |                       0.125 |         0.2095 |            0.1466 |
|          1      |                       0.125 |         0.2095 |            0.1466 |

### physics_rules

|   on_percentile |   warned_before_established |   event_recall |   fa_per_well_day |
|----------------:|----------------------------:|---------------:|------------------:|
|          0.99   |                      0.7083 |         0.3527 |            0.526  |
|          0.9968 |                      0.5625 |         0.222  |            0.1035 |
|          0.999  |                      0.4792 |         0.0975 |            0.0345 |
|          0.9997 |                      0.4375 |         0.0788 |            0.0345 |
|          0.9999 |                      0.4167 |         0.0664 |            0.0345 |
|          1      |                      0.4167 |         0.0643 |            0.0345 |
|          1      |                      0.4167 |         0.0643 |            0.0345 |
|          1      |                      0.4167 |         0.0643 |            0.0345 |
|          1      |                      0.4167 |         0.0643 |            0.0345 |

### zscore

|   on_percentile |   warned_before_established |   event_recall |   fa_per_well_day |
|----------------:|----------------------------:|---------------:|------------------:|
|          0.99   |                      0.6875 |         0.1701 |            0.6468 |
|          0.9968 |                      0.5417 |         0.0954 |            0.1552 |
|          0.999  |                      0.1667 |         0.0228 |            0.1035 |
|          0.9997 |                      0      |         0      |            0      |
|          0.9999 |                      0      |         0      |            0      |
|          1      |                      0      |         0      |            0      |
|          1      |                      0      |         0      |            0      |
|          1      |                      0      |         0      |            0      |
|          1      |                      0      |         0      |            0      |

### iforest

|   on_percentile |   warned_before_established |   event_recall |   fa_per_well_day |
|----------------:|----------------------------:|---------------:|------------------:|
|          0.99   |                      0.4792 |         0.2095 |            0.1035 |
|          0.9968 |                      0.3542 |         0.1017 |            0.0776 |
|          0.999  |                      0.25   |         0.0456 |            0.138  |
|          0.9997 |                      0.25   |         0.0394 |            0.1466 |
|          0.9999 |                      0.25   |         0.0332 |            0.1294 |
|          1      |                      0.2292 |         0.027  |            0.1121 |
|          1      |                      0.2292 |         0.027  |            0.1121 |
|          1      |                      0.2083 |         0.0249 |            0.1035 |
|          1      |                      0.2083 |         0.0249 |            0.1121 |

## Training-side calibration per fold

| scorer        | protocol                |   fold |   on_percentile |   train_fa_per_day |   train_time_in_alarm_pct |   train_normal_days | meets_budget   |
|:--------------|:------------------------|-------:|----------------:|-------------------:|--------------------------:|--------------------:|:---------------|
| health_only   | P1_sim_only             |      0 |        0.99     |              0     |                      0    |                31.5 | True           |
| health_only   | P1_sim_only             |      1 |        0.99     |              0     |                      0    |                31.5 | True           |
| health_only   | P1_sim_only             |      2 |        0.99     |              0     |                      0    |                31.5 | True           |
| health_only   | P1_sim_only             |      3 |        0.99     |              0     |                      0    |                31.5 | True           |
| health_only   | P1_sim_only             |      4 |        0.99     |              0     |                      0    |                31.5 | True           |
| health_only   | P2_sim_plus_site_normal |      0 |        0.999842 |              0     |                      0    |                92.4 | True           |
| health_only   | P2_sim_plus_site_normal |      1 |        0.996019 |              0     |                      0    |                96.7 | True           |
| health_only   | P2_sim_plus_site_normal |      2 |        0.99     |              0.027 |                      0.22 |                73.8 | True           |
| health_only   | P2_sim_plus_site_normal |      3 |        0.994988 |              0     |                      0    |               102.7 | True           |
| health_only   | P2_sim_plus_site_normal |      4 |        0.994988 |              0     |                      0    |                98.3 | True           |
| health_only   | P3_real                 |      0 |        0.992057 |              0     |                      0    |                92.4 | True           |
| health_only   | P3_real                 |      1 |        0.99369  |              0.041 |                      0.08 |                96.7 | True           |
| health_only   | P3_real                 |      2 |        0.99369  |              0     |                      0    |                73.8 | True           |
| health_only   | P3_real                 |      3 |        0.994988 |              0     |                      0    |               102.7 | True           |
| health_only   | P3_real                 |      4 |        0.992057 |              0.02  |                      0.3  |                98.3 | True           |
| health_only   | P4_real_plus_sim        |      0 |        0.994988 |              0.129 |                      0.57 |               123.9 | True           |
| health_only   | P4_real_plus_sim        |      1 |        0.99     |              0     |                      0    |               128.2 | True           |
| health_only   | P4_real_plus_sim        |      2 |        0.994988 |              0.009 |                      0.01 |               105.4 | True           |
| health_only   | P4_real_plus_sim        |      3 |        0.99369  |              0     |                      0    |               134.2 | True           |
| health_only   | P4_real_plus_sim        |      4 |        0.99     |              0     |                      0    |               129.8 | True           |
| physics_rules | P1_sim_only             |      0 |        0.999    |              0.095 |                      0.12 |                31.5 | True           |
| physics_rules | P1_sim_only             |      1 |        0.999    |              0.095 |                      0.12 |                31.5 | True           |
| physics_rules | P1_sim_only             |      2 |        0.999    |              0.095 |                      0.12 |                31.5 | True           |
| physics_rules | P1_sim_only             |      3 |        0.999    |              0.095 |                      0.12 |                31.5 | True           |
| physics_rules | P1_sim_only             |      4 |        0.999    |              0.095 |                      0.12 |                31.5 | True           |
| physics_rules | P2_sim_plus_site_normal |      0 |        0.996838 |              0.108 |                      0.58 |                92.4 | True           |
| physics_rules | P2_sim_plus_site_normal |      1 |        0.996019 |              0.114 |                      0.59 |                96.7 | True           |
| physics_rules | P2_sim_plus_site_normal |      2 |        0.996838 |              0.068 |                      0.55 |                73.8 | True           |
| physics_rules | P2_sim_plus_site_normal |      3 |        0.999206 |              0.117 |                      0.19 |               102.7 | True           |
| physics_rules | P2_sim_plus_site_normal |      4 |        0.996838 |              0.122 |                      0.57 |                98.3 | True           |
| physics_rules | P3_real                 |      0 |        0.996838 |              0.108 |                      0.58 |                92.4 | True           |
| physics_rules | P3_real                 |      1 |        0.996019 |              0.114 |                      0.59 |                96.7 | True           |
| physics_rules | P3_real                 |      2 |        0.996838 |              0.068 |                      0.55 |                73.8 | True           |
| physics_rules | P3_real                 |      3 |        0.999206 |              0.117 |                      0.19 |               102.7 | True           |
| physics_rules | P3_real                 |      4 |        0.996838 |              0.122 |                      0.57 |                98.3 | True           |
| physics_rules | P4_real_plus_sim        |      0 |        0.997488 |              0.081 |                      0.43 |               123.9 | True           |
| physics_rules | P4_real_plus_sim        |      1 |        0.996019 |              0.133 |                      0.63 |               128.2 | True           |
| physics_rules | P4_real_plus_sim        |      2 |        0.996838 |              0.133 |                      0.56 |               105.4 | True           |
| physics_rules | P4_real_plus_sim        |      3 |        0.998741 |              0.142 |                      0.27 |               134.2 | True           |
| physics_rules | P4_real_plus_sim        |      4 |        0.997488 |              0.092 |                      0.44 |               129.8 | True           |
| zscore        | P1_sim_only             |      0 |        0.994988 |              0.063 |                      0.26 |                31.5 | True           |
| zscore        | P1_sim_only             |      1 |        0.994988 |              0.063 |                      0.26 |                31.5 | True           |
| zscore        | P1_sim_only             |      2 |        0.994988 |              0.063 |                      0.26 |                31.5 | True           |
| zscore        | P1_sim_only             |      3 |        0.994988 |              0.063 |                      0.26 |                31.5 | True           |
| zscore        | P1_sim_only             |      4 |        0.994988 |              0.063 |                      0.26 |                31.5 | True           |
| zscore        | P2_sim_plus_site_normal |      0 |        0.996019 |              0.13  |                      0.59 |                92.4 | True           |
| zscore        | P2_sim_plus_site_normal |      1 |        0.996838 |              0.134 |                      0.43 |                96.7 | True           |
| zscore        | P2_sim_plus_site_normal |      2 |        0.998415 |              0     |                      0    |                73.8 | True           |
| zscore        | P2_sim_plus_site_normal |      3 |        0.998415 |              0.127 |                      0.26 |               102.7 | True           |
| zscore        | P2_sim_plus_site_normal |      4 |        0.996838 |              0.132 |                      0.43 |                98.3 | True           |
| zscore        | P3_real                 |      0 |        0.996019 |              0.13  |                      0.59 |                92.4 | True           |
| zscore        | P3_real                 |      1 |        0.996838 |              0.134 |                      0.43 |                96.7 | True           |
| zscore        | P3_real                 |      2 |        0.998415 |              0     |                      0    |                73.8 | True           |
| zscore        | P3_real                 |      3 |        0.998415 |              0.127 |                      0.26 |               102.7 | True           |
| zscore        | P3_real                 |      4 |        0.996838 |              0.132 |                      0.43 |                98.3 | True           |
| zscore        | P4_real_plus_sim        |      0 |        0.994988 |              0.121 |                      0.55 |               123.9 | True           |
| zscore        | P4_real_plus_sim        |      1 |        0.996838 |              0.109 |                      0.37 |               128.2 | True           |
| zscore        | P4_real_plus_sim        |      2 |        0.996019 |              0.133 |                      0.55 |               105.4 | True           |
| zscore        | P4_real_plus_sim        |      3 |        0.997488 |              0.134 |                      0.31 |               134.2 | True           |
| zscore        | P4_real_plus_sim        |      4 |        0.996838 |              0.108 |                      0.37 |               129.8 | True           |
| iforest       | P1_sim_only             |      0 |        0.996838 |              0.127 |                      0.64 |                31.5 | True           |
| iforest       | P1_sim_only             |      1 |        0.996838 |              0.127 |                      0.64 |                31.5 | True           |
| iforest       | P1_sim_only             |      2 |        0.996838 |              0.127 |                      0.64 |                31.5 | True           |
| iforest       | P1_sim_only             |      3 |        0.996838 |              0.127 |                      0.64 |                31.5 | True           |
| iforest       | P1_sim_only             |      4 |        0.996838 |              0.127 |                      0.64 |                31.5 | True           |
| iforest       | P2_sim_plus_site_normal |      0 |        0.998005 |              0.032 |                      0.88 |                92.4 | True           |
| iforest       | P2_sim_plus_site_normal |      1 |        0.997488 |              0.041 |                      0.97 |                96.7 | True           |
| iforest       | P2_sim_plus_site_normal |      2 |        0.997488 |              0.081 |                      0.85 |                73.8 | True           |
| iforest       | P2_sim_plus_site_normal |      3 |        0.999    |              0.136 |                      0.33 |               102.7 | True           |
| iforest       | P2_sim_plus_site_normal |      4 |        0.996838 |              0.112 |                      0.79 |                98.3 | True           |
| iforest       | P3_real                 |      0 |        0.998005 |              0.032 |                      0.88 |                92.4 | True           |
| iforest       | P3_real                 |      1 |        0.997488 |              0.041 |                      0.97 |                96.7 | True           |
| iforest       | P3_real                 |      2 |        0.997488 |              0.081 |                      0.85 |                73.8 | True           |
| iforest       | P3_real                 |      3 |        0.999    |              0.136 |                      0.33 |               102.7 | True           |
| iforest       | P3_real                 |      4 |        0.996838 |              0.112 |                      0.79 |                98.3 | True           |
| iforest       | P4_real_plus_sim        |      0 |        0.998005 |              0.024 |                      0.86 |               123.9 | True           |
| iforest       | P4_real_plus_sim        |      1 |        0.996838 |              0.016 |                      0.94 |               128.2 | True           |
| iforest       | P4_real_plus_sim        |      2 |        0.998005 |              0.028 |                      0.91 |               105.4 | True           |
| iforest       | P4_real_plus_sim        |      3 |        0.999    |              0.142 |                      0.31 |               134.2 | True           |
| iforest       | P4_real_plus_sim        |      4 |        0.998415 |              0.131 |                      0.32 |               129.8 | True           |
