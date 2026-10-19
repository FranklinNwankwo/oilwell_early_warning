"""Evaluation protocols. All real-data splits are grouped by WELL.

The operator's real question is not "does a model trained on real events detect real
events?" (they have almost none) but "can SIMULATED events plus my own wells' NORMAL
history replace labeled real events?". So the protocols are an ablation on the same
well-disjoint test folds:

  P1  sim-only      train: simulated + drawn instances only.            (no real data at all)
  P2  sim + normal  train: P1 + real NORMAL data of the training wells.  (realistic deployment)
  P3  real          train: all real data of the training wells.          (upper reference)
  P4  real + sim    train: P3 + simulated + drawn.                       (does sim help real?)

Test, for every protocol and fold: all real instances of the fold's held-out wells.
P2 - P1 isolates the value of site normal data; P3 - P2 isolates the value of labeled
real events. Real class 4 (Flow Instability) has no simulated instances, so under
P1/P2 it is an UNSEEN event type: we report whether the detector flags it anyway.

Excluded from scoring (but kept for training normal context where applicable):
  * real class-9 instances with zero event-labeled samples (43 of 57 in 3W 2.0.0).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

PROTOCOLS = ("P1_sim_only", "P2_sim_plus_site_normal", "P3_real", "P4_real_plus_sim")


@dataclass(frozen=True)
class Fold:
    k: int
    test_wells: tuple[str, ...]
    train_wells: tuple[str, ...]


def assign_well_folds(real_catalog: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> dict[str, int]:
    """Deterministic greedy assignment of wells to folds that balances, per fold, the
    number of event instances of every class and the normal well-days.

    `real_catalog` needs columns: well, event_class, and 'normal_days' (labeled class-0
    days contributed by that instance; 0 for event instances is fine).
    """
    rng = np.random.default_rng(seed)
    cat = real_catalog
    per_well = pd.crosstab(cat["well"], cat["event_class"])
    per_well = per_well.drop(columns=0, errors="ignore")
    normal_days = cat.groupby("well")["normal_days"].sum()
    per_well = per_well.reindex(normal_days.index, fill_value=0)
    per_well["_normal"] = normal_days
    # scale every column to "fraction of total" so rare classes weigh as much as common ones
    totals = per_well.sum().replace(0, 1)
    frac = per_well / totals
    # biggest wells first; random tie-break so the seed matters only for ties
    # Some wells dominate a class (WELL-00014 holds 31/32 real class-3 instances,
    # WELL-00011 13/22 of class 2): perfect balance is impossible, so minimise the
    # squared load (spreads every class) plus a term that equalises well counts.
    frac["_wells"] = 1.0 / len(frac)
    order = sorted(frac.index, key=lambda w: (-frac.loc[w].max(), rng.random()))
    load = np.zeros((n_folds, frac.shape[1]))
    out: dict[str, int] = {}
    for w in order:
        v = frac.loc[w].to_numpy()
        cost = [np.sum((load[f] + v) ** 2) - np.sum(load[f] ** 2) for f in range(n_folds)]
        f = int(np.argmin(cost))
        load[f] += v
        out[w] = f
    return out


def make_folds(well_fold: dict[str, int]) -> list[Fold]:
    by = defaultdict(list)
    for w, f in well_fold.items():
        by[f].append(w)
    wells = sorted(well_fold)
    return [
        Fold(k=f, test_wells=tuple(sorted(by[f])), train_wells=tuple(w for w in wells if w not in by[f]))
        for f in sorted(by)
    ]


def fold_balance_table(real_catalog: pd.DataFrame, well_fold: dict[str, int]) -> pd.DataFrame:
    c = real_catalog.assign(fold=real_catalog.well.map(well_fold))
    tab = pd.crosstab(c["fold"], c["event_class"])
    tab["normal_days"] = c.groupby("fold")["normal_days"].sum().round(1)
    tab["wells"] = c.groupby("fold")["well"].nunique()
    return tab


def training_selection(protocol: str, catalog: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    """Rows of `catalog` (instance-level) usable for TRAINING under a protocol.

    For P2 the real rows are normal instances only; event instances of training
    wells are withheld entirely (not just their event samples), so no labeled real
    event signal can leak in through an instance's normal prefix either.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(protocol)
    synth = catalog.source.isin(["simulated", "drawn"])
    real_train = (catalog.source == "real") & catalog.well.isin(fold.train_wells)
    if protocol == "P1_sim_only":
        sel = synth
    elif protocol == "P2_sim_plus_site_normal":
        sel = synth | (real_train & (catalog.event_class == 0))
    elif protocol == "P3_real":
        sel = real_train
    else:
        sel = synth | real_train
    return catalog[sel]


def test_selection(catalog: pd.DataFrame, fold: Fold, scorable: pd.Series) -> pd.DataFrame:
    """Real instances of the held-out wells that can be scored (see module docstring)."""
    return catalog[(catalog.source == "real") & catalog.well.isin(fold.test_wells) & scorable]
