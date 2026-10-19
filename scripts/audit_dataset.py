"""Reproducible audit of 3W 2.0.0. Writes data/catalog.parquet and reports/data_audit.md.

Usage (PowerShell, from the repo root):
    python scripts/audit_dataset.py --dataset C:/path/to/3W
or, with the EW3W_DATASET environment variable set:
    python scripts/audit_dataset.py

Every number quoted in docs/ comes from this script's output. Re-run it after any
3W dataset upgrade and diff the report.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ew3w.data import catalog, discover, load_instance, stitch_well  # noqa: E402
from ew3w.quality import FrozenRule, Health, health_matrix  # noqa: E402
from ew3w.schema import ALL_SENSORS, SHARED_SENSORS, TRANSIENT_OFFSET  # noqa: E402
from ew3w.splits import assign_well_folds, fold_balance_table  # noqa: E402


def instance_stats(meta) -> dict:
    df = load_instance(meta)
    # Frozen detection only on real data: synthetic instances are clean by construction.
    h = health_matrix(df, list(ALL_SENSORS), FrozenRule(), segment_col=None, detect_frozen=meta.source == "real")
    cl = df["class"].to_numpy()
    c = meta.event_class
    lab = ~np.isnan(cl)
    i_tr = np.flatnonzero(cl == TRANSIENT_OFFSET + c) if c else np.array([])
    i_ss = np.flatnonzero(cl == c) if c else np.array([])
    first_lab = np.flatnonzero(lab)[0] if lab.any() else None
    onset = i_tr[0] if i_tr.size else (i_ss[0] if i_ss.size else None)
    row = dict(
        n=len(df),
        unlabeled_s=int((~lab).sum()),
        normal_s=int((cl == 0).sum()),
        transient_s=int(i_tr.size),
        steady_s=int(i_ss.size),
        event_rows=int(i_tr.size + i_ss.size),
        normal_before_onset_s=int(onset - first_lab) if (onset is not None and first_lab is not None) else 0,
        states=",".join(map(str, sorted(set(df["state"].dropna().astype(int))))),
    )
    # Raw (unconverted) pressure magnitude, to audit units per source.
    raw = pd.read_parquet(meta.path, columns=["P-TPT", "P-MON-CKP", "P-PDG"]).to_numpy(dtype=np.float64)
    raw = np.abs(raw[np.isfinite(raw) & (raw != 0)])
    row["raw_pressure_median"] = float(np.median(raw)) if raw.size else np.nan
    for s in ALL_SENSORS:
        row[f"ok_{s}"] = float((h[s] == Health.OK).mean())
        row[f"frozen_{s}"] = float((h[s] == Health.FROZEN).mean())
        row[f"invalid_{s}"] = float((h[s] == Health.INVALID).mean())
    return row


def missingness_leak_test(cat: pd.DataFrame) -> dict:
    """Can sensor HEALTH alone (no sensor values) tell event instances from normal ones?
    If yes, any model that sees missingness can cheat. Reported grouped by well (the
    honest number) and with a random split (to show what leaks if you don't group)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    r = cat[(cat.source == "real") & cat.scorable].reset_index(drop=True)
    X = r[[f"ok_{s}" for s in ALL_SENSORS]].to_numpy()
    y = (r.event_class > 0).to_numpy().astype(int)
    oof = np.zeros(len(r))
    for tr, te in GroupKFold(n_splits=5).split(X, y, r.well):
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    # Same model, WRONG split (random by instance): shows what leaks if you don't group.
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    def oof_auc(splitter, groups=None, model=None):
        p = np.zeros(len(r))
        for tr, te in splitter.split(X, y, groups):
            base = LogisticRegression(max_iter=2000, class_weight="balanced") if model is None else model
            m = clone(base).fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        return roc_auc_score(y, p)

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1)
    return dict(
        auc_grouped_lr=roc_auc_score(y, oof),
        auc_grouped_rf=oof_auc(GroupKFold(n_splits=5), r.well, rf),
        auc_random_rf=oof_auc(StratifiedKFold(5, shuffle=True, random_state=0), None, rf),
        n=len(r),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default=os.environ.get("EW3W_DATASET"),
        help="3W repo root or its 'dataset' folder (default: the EW3W_DATASET environment variable)",
    )
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    if not a.dataset:
        ap.error("no dataset path: pass --dataset or set the EW3W_DATASET environment variable")
    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    metas = discover(a.dataset)
    cat = catalog(metas)
    stats = []
    for i, m in enumerate(metas):
        stats.append(instance_stats(m))
        if (i + 1) % 200 == 0:
            print(f"  profiled {i + 1}/{len(metas)}", flush=True)
    cat = pd.concat([cat, pd.DataFrame(stats)], axis=1)
    # Real event instances with no event-labeled sample cannot be scored as events.
    cat["scorable"] = ~((cat.event_class > 0) & (cat.event_rows == 0))

    # De-duplicated normal exposure via stitched timelines (only class/state needed).
    expo = {}
    for w, idx in cat[(cat.source == "real") & (cat.event_class == 0)].groupby("well").groups.items():
        tl = stitch_well([metas[i] for i in idx], sensors=[])
        expo[w] = float((tl["class"] == 0).sum()) / 86_400
    naive_days = cat[(cat.source == "real") & (cat.event_class == 0)].n.sum() / 86_400
    cat["normal_days"] = np.where((cat.source == "real") & (cat.event_class == 0), cat.normal_s / 86_400, 0.0)
    cat.to_parquet(out / "data" / "catalog.parquet")

    leak = missingness_leak_test(cat)
    real = cat[cat.source == "real"]
    folds = assign_well_folds(real, n_folds=5)
    pd.Series(folds, name="fold").to_frame().to_csv(out / "data" / "well_folds.csv")

    g = cat.groupby(["event_class", "source"])
    counts = g.agg(
        instances=("instance_id", "size"),
        wells=("well", "nunique"),
        hours=("n", lambda s: s.sum() / 3600),
        unscorable=("scorable", lambda s: int((~s).sum())),
        median_transient_min=("transient_s", lambda s: s[s > 0].median() / 60 if (s > 0).any() else np.nan),
        reaches_steady=("steady_s", lambda s: (s > 0).mean()),
    ).round(2)
    avail = cat.groupby(["event_class", "source"])[[f"ok_{s}" for s in SHARED_SENSORS]].mean().round(2)
    avail.columns = [c[3:] for c in avail.columns]

    md = [
        "# 3W 2.0.0 data audit",
        "",
        f"Instances: {len(cat)} (real {int((cat.source == 'real').sum())}, simulated {int((cat.source == 'simulated').sum())}, drawn {int((cat.source == 'drawn').sum())}); real wells: {real.well.nunique()}.",
        "",
        "## Counts by class and source",
        counts.to_markdown(),
        "",
        "## Normal exposure",
        f"Naive (sum of real normal instance lengths): {naive_days:.1f} well-days. "
        f"De-duplicated labeled normal: {sum(expo.values()):.1f} well-days across {len(expo)} wells.",
        "",
        pd.Series(expo, name="labeled_normal_days").round(2).to_frame().to_markdown(),
        "",
        "## Shared-sensor health (fraction of samples OK = present, in physical range, not frozen)",
        avail.to_markdown(),
        "",
        "## Missingness leak test",
        f"Event-vs-normal classifiers that see ONLY per-instance sensor-health fractions (which "
        f"sensors are present/frozen/invalid, no values), n={leak['n']} scorable real instances, out-of-fold ROC-AUC:",
        "",
        "| model | split | ROC-AUC |",
        "|---|---|---|",
        f"| logistic regression | grouped by well | {leak['auc_grouped_lr']:.3f} |",
        f"| random forest | grouped by well | {leak['auc_grouped_rf']:.3f} |",
        f"| random forest | random by instance (wrong) | {leak['auc_random_rf']:.3f} |",
        "",
        "Missingness identifies the WELL, and the well identifies the label: with a random split "
        "the shortcut is near-perfect. Grouped by well it still partly transfers, so the grouped "
        "random-forest score is the health-only baseline every model must beat.",
        "",
        "## Pressure units by source",
        "Median raw magnitude of non-zero P-TPT / P-MON-CKP / P-PDG per instance. dataset.ini "
        "declares Pa; hand-drawn instances are bar-scale and are converted x1e5 at load time.",
        "",
        cat.assign(band=pd.cut(cat.raw_pressure_median, [0, 1e3, 1e5, np.inf], labels=["< 1e3 (bar)", "1e3-1e5", ">= 1e5 (Pa)"]))
        .pipe(lambda d: pd.crosstab(d.source, d.band)).to_markdown(),
        "",
        "## Invalid (physically implausible) values, fraction of real samples",
        cat[cat.source == "real"][[f"invalid_{s}" for s in ALL_SENSORS]].mean().pipe(lambda s: s[s > 0])
        .rename(lambda c: c[8:]).round(5).to_frame("invalid_fraction").to_markdown(),
        "",
        "## Well folds (5-fold, grouped by well, balanced per class)",
        fold_balance_table(real, folds).to_markdown(),
        "",
    ]
    (out / "reports" / "data_audit.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
