# 3W 2.0.0 data audit

Instances: 2228 (real 1119, simulated 1089, drawn 20); real wells: 40.

## Counts by class and source
|                  |   instances |   wells |   hours |   unscorable |   median_transient_min |   reaches_steady |
|:-----------------|------------:|--------:|--------:|-------------:|-----------------------:|-----------------:|
| (0, 'real')      |         594 |       9 | 3377.27 |            0 |                 nan    |             0    |
| (1, 'drawn')     |          10 |       0 |  152    |            0 |                 453.74 |             0.6  |
| (1, 'real')      |           4 |       3 |   65.78 |            0 |                 163.62 |             1    |
| (1, 'simulated') |         114 |       0 | 2311.98 |            0 |                 840    |             1    |
| (2, 'real')      |          22 |       7 |   76.94 |            0 |                  39.17 |             0.5  |
| (2, 'simulated') |          16 |       0 |  128    |            0 |                  60    |             1    |
| (3, 'real')      |          32 |       2 |  190.1  |            0 |                 nan    |             1    |
| (3, 'simulated') |          74 |       0 | 1184.7  |            0 |                 nan    |             1    |
| (4, 'real')      |         343 |       7 | 1024.91 |            0 |                 nan    |             1    |
| (5, 'real')      |          11 |       3 |  122.06 |            0 |                 298.52 |             0.64 |
| (5, 'simulated') |         439 |       0 | 3572.85 |            0 |                  80    |             1    |
| (6, 'real')      |           6 |       2 |   21.52 |            0 |                  12.26 |             1    |
| (6, 'simulated') |         215 |       0 | 1612.44 |            0 |                 120    |             1    |
| (7, 'drawn')     |          10 |       0 |  672    |            0 |                4206.33 |             0.4  |
| (7, 'real')      |          36 |       6 | 2184.71 |            0 |                2410.37 |             0.14 |
| (8, 'real')      |          14 |       9 | 1335.84 |            0 |                3780.9  |             0.86 |
| (8, 'simulated') |          81 |       0 |  607.48 |            0 |                 288.33 |             0.75 |
| (9, 'real')      |          57 |      15 |  732.05 |           43 |                 205.54 |             0.05 |
| (9, 'simulated') |         150 |       0 | 1901.63 |            0 |                 306    |             1    |

## Normal exposure
Naive (sum of real normal instance lengths): 140.7 well-days. De-duplicated labeled normal: 116.0 well-days across 9 wells.

|            |   labeled_normal_days |
|:-----------|----------------------:|
| WELL-00001 |                 19.28 |
| WELL-00002 |                 42.14 |
| WELL-00003 |                  5.37 |
| WELL-00004 |                  0.99 |
| WELL-00005 |                 11.34 |
| WELL-00006 |                 23.41 |
| WELL-00007 |                  0.17 |
| WELL-00008 |                 11.67 |
| WELL-00019 |                  1.62 |

## Shared-sensor health (fraction of samples OK = present, in physical range, not frozen)
|                  |   P-PDG |   P-TPT |   T-TPT |   P-MON-CKP |   T-JUS-CKP |
|:-----------------|--------:|--------:|--------:|------------:|------------:|
| (0, 'real')      |    0.18 |    0.86 |    0.88 |        0.55 |        0.5  |
| (1, 'drawn')     |    1    |    1    |    1    |        1    |        1    |
| (1, 'real')      |    0.07 |    0.74 |    0.99 |        0.75 |        0.75 |
| (1, 'simulated') |    1    |    1    |    1    |        1    |        1    |
| (2, 'real')      |    0.77 |    0.94 |    1    |        0.32 |        0.05 |
| (2, 'simulated') |    1    |    1    |    1    |        1    |        1    |
| (3, 'real')      |    0.97 |    1    |    1    |        1    |        1    |
| (3, 'simulated') |    1    |    1    |    1    |        1    |        1    |
| (4, 'real')      |    0.29 |    0.75 |    0.98 |        0.67 |        0.43 |
| (5, 'real')      |    0.49 |    0.91 |    1    |        1    |        1    |
| (5, 'simulated') |    1    |    1    |    1    |        1    |        1    |
| (6, 'real')      |    0.39 |    0.99 |    1    |        0.5  |        0.5  |
| (6, 'simulated') |    1    |    1    |    0    |        1    |        1    |
| (7, 'drawn')     |    1    |    1    |    1    |        1    |        1    |
| (7, 'real')      |    0.03 |    0.85 |    0.89 |        1    |        0.89 |
| (8, 'real')      |    0.65 |    0.57 |    0.49 |        0.97 |        0    |
| (8, 'simulated') |    1    |    1    |    1    |        1    |        1    |
| (9, 'real')      |    0.51 |    0.46 |    0.53 |        1    |        0.53 |
| (9, 'simulated') |    1    |    1    |    1    |        1    |        1    |

## Missingness leak test
Event-vs-normal classifiers that see ONLY per-instance sensor-health fractions (which sensors are present/frozen/invalid, no values), n=1076 scorable real instances, out-of-fold ROC-AUC:

| model | split | ROC-AUC |
|---|---|---|
| logistic regression | grouped by well | 0.523 |
| random forest | grouped by well | 0.740 |
| random forest | random by instance (wrong) | 0.986 |

Missingness identifies the WELL, and the well identifies the label: with a random split the shortcut is near-perfect. Grouped by well it still partly transfers, so the grouped random-forest score is the health-only baseline every model must beat.

## Pressure units by source
Median raw magnitude of non-zero P-TPT / P-MON-CKP / P-PDG per instance. dataset.ini declares Pa; hand-drawn instances are bar-scale and are converted x1e5 at load time.

| source    |   < 1e3 (bar) |   >= 1e5 (Pa) |
|:----------|--------------:|--------------:|
| drawn     |            20 |             0 |
| real      |             0 |          1062 |
| simulated |             0 |          1089 |

## Invalid (physically implausible) values, fraction of real samples
|           |   invalid_fraction |
|:----------|-------------------:|
| ABER-CKP  |            0.00089 |
| P-ANULAR  |            0.00081 |
| P-JUS-CKP |            0.00357 |
| P-PDG     |            0.02715 |
| P-TPT     |            0.00894 |
| T-MON-CKP |            0.00089 |
| T-PDG     |            0.03111 |

## Well folds (5-fold, grouped by well, balanced per class)
|   fold |   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 |   9 |   normal_days |   wells |
|-------:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|--------------:|--------:|
|      0 | 115 |   2 |   2 |  31 |  31 |   0 |   0 |   2 |   3 |  11 |          23.6 |       9 |
|      1 |  93 |   1 |  13 |   1 |  36 |   4 |   0 |   3 |   2 |  11 |          19.3 |       8 |
|      2 | 209 |   1 |   2 |   0 | 112 |   6 |   3 |   4 |   2 |  11 |          42.1 |       8 |
|      3 |  58 |   0 |   2 |   0 |  83 |   1 |   0 |  19 |   4 |  12 |          13.3 |       8 |
|      4 | 119 |   0 |   3 |   0 |  81 |   0 |   3 |   8 |   3 |  12 |          17.7 |       7 |
