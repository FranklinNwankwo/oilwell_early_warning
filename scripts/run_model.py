"""Phase 3: main model under every protocol, the dropout ablation, and the head-to-head
against the best baseline. Writes reports/phase3_model.md.

Usage (PowerShell, from the repo root, after scripts/build_features.py and
scripts/audit_dataset.py):
    python scripts/run_model.py              # ~75 min here (2 CPUs), ~4.1 GB peak RAM
    python scripts/run_model.py --boot 200   # faster, rougher CIs

Runs (all through evaluation.run_protocol: training-wells-only, out-of-fold
calibration, 1 FA/well-week + 1% time-in-alarm budgets):
    gbdt              x P1, P2, P3, P4      the main model, missingness-matched dropout
    gbdt_sim_trained  x P2                  trained on simulation only; site normal data used
                                            only for masks and the threshold
    gbdt_no_dropout   x P2                  ablation: does the dropout matter?
    physics_rules     x P2                  best Phase 2 baseline, re-run for the paired test

Reproducibility: two independent full runs produced byte-identical reports and
outcome files (all randomness is seeded).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ew3w.alerting import HysteresisConfig  # noqa: E402
from ew3w.baselines import VALUE_COLS, PhysicsRules  # noqa: E402
from ew3w.evaluation import TIME_IN_ALARM_BUDGET, headline, load_cache, operating_curve, per_class, run_protocol  # noqa: E402
from ew3w.features import FeatureConfig  # noqa: E402
from ew3w.metrics import paired_difference  # noqa: E402
from ew3w.model import GBDTEarlyWarning, GBDTNoDropout, GBDTSimTrained, apply_mask, availability  # noqa: E402
from ew3w.splits import PROTOCOLS, make_folds  # noqa: E402

FA_BUDGET = 1 / 7
P2 = "P2_sim_plus_site_normal"


def _fmt(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def domain_gap(real: pd.DataFrame, synth: pd.DataFrame, seed: int = 0) -> dict:
    """How well can a classifier tell simulated from real NORMAL rows? Grouped by
    well/instance. 'raw' uses rows as they are; 'matched' first gives the simulated rows
    real availability patterns, so what remains is the gap in the VALUES themselves."""
    rng = np.random.default_rng(seed)
    rn = real[(real["event_class"] == 0) & (real["class"] == 0)]
    sn = synth[synth["class"] == 0]
    rn = rn.iloc[rng.choice(len(rn), min(len(rn), 150_000), replace=False)]
    sn = sn.iloc[rng.choice(len(sn), min(len(sn), 150_000), replace=False)]
    out = {}
    for mode in ("raw", "matched"):
        Xs = sn[VALUE_COLS].to_numpy(np.float32, copy=True)
        if mode == "matched":
            Xs = apply_mask(Xs, availability(rn)[rng.integers(0, len(rn), len(sn))])
        X = np.vstack([rn[VALUE_COLS].to_numpy(np.float32), Xs])
        y = np.r_[np.zeros(len(rn)), np.ones(len(sn))]
        g = np.r_[rn["well"].astype(str).to_numpy(), sn["well"].astype(str).to_numpy()]
        p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            m = HistGradientBoostingClassifier(max_iter=100, random_state=seed).fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        out[mode] = roc_auc_score(y, p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--folds", default="data/well_folds.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    cache = Path(a.cache) if a.cache else Path("data/features") / FeatureConfig().key
    if not cache.exists():
        sys.exit(f"Feature cache {cache} not found: run scripts/build_features.py first.")
    if not Path(a.folds).exists():
        sys.exit(f"{a.folds} not found: run scripts/audit_dataset.py first.")
    folds = make_folds(pd.read_csv(a.folds, index_col=0)["fold"].to_dict())
    alert = HysteresisConfig(period_s=FeatureConfig().block_s)
    t0 = time.perf_counter()

    cols = sorted(set(GBDTEarlyWarning.columns) | set(PhysicsRules.columns))
    real, synth = load_cache(cache, cols, synth_event_cap=GBDTEarlyWarning().per_group)
    print(f"Loaded real {len(real):,} / synthetic {len(synth):,} rows ({time.perf_counter() - t0:.0f} s)", flush=True)

    runs = [(GBDTEarlyWarning, p) for p in PROTOCOLS] + [(GBDTSimTrained, P2), (GBDTNoDropout, P2), (PhysicsRules, P2)]
    results = {}
    for S, p in runs:
        print(f"{S.name} / {p}", flush=True)
        r = run_protocol(S, p, real, synth, folds, alert, FA_BUDGET)
        h = headline(r, n_boot=a.boot)
        results[(S.name, p)] = (r, h)
        print(f"  warned {_fmt(h['warned_before_established'])}  recall {_fmt(h['event_recall'])}  "
              f"FA {_fmt(h['fa_per_well_day'], 3)}/well-day  in alarm {100 * h['time_in_alarm']:.1f}%  ({r.seconds:.0f} s)", flush=True)

    gb = results[("gbdt", P2)][0]
    curve = operating_curve(gb, real, folds, alert)
    vs_rules = paired_difference(gb.outcomes, results[("physics_rules", P2)][0].outcomes, n_boot=a.boot)
    st = results[("gbdt_sim_trained", P2)][0]
    st_vs_rules = paired_difference(st.outcomes, results[("physics_rules", P2)][0].outcomes, n_boot=a.boot)
    st_vs_gb = paired_difference(st.outcomes, gb.outcomes, n_boot=a.boot)
    vs_nodrop = paired_difference(gb.outcomes, results[("gbdt_no_dropout", P2)][0].outcomes, n_boot=a.boot)
    print("Domain gap (sim vs real normal) ...", flush=True)
    gap = domain_gap(real, synth)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([r.outcomes.assign(scorer=r.scorer, protocol=r.protocol) for r, _ in results.values()]).to_csv(
        out / "phase3_outcomes.csv", index=False)

    md = [
        "# Phase 3: main model",
        "",
        f"Gradient-boosted trees on the 63 value features, with missingness-matched sensor dropout "
        f"(`src/ew3w/model.py`). Same harness as Phase 2: thresholds chosen on training wells only "
        f"(<= {FA_BUDGET:.3f} false alarms per well-day AND <= {100 * TIME_IN_ALARM_BUDGET:.0f}% time in alarm, "
        f"out-of-fold scores), then frozen and applied to the held-out wells. 95% CIs: well-cluster bootstrap "
        f"({a.boot}) and exact Poisson.",
        "",
        "| Model | Protocol | Warned [95% CI] | n | Recall | Lead, min [95% CI] | FA / well-day [95% CI] | Time in alarm | Class 4 recall |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (name, p), (r, h) in results.items():
        md.append(
            f"| {name} | {p} | {_fmt(h['warned_before_established'])} [{_fmt(h['warned_ci'][0])}, {_fmt(h['warned_ci'][1])}] | "
            f"{h['n_lead_defined']} | {_fmt(h['event_recall'])} | {_fmt(h['median_lead_min'], 0)} "
            f"[{_fmt(h['lead_ci'][0], 0)}, {_fmt(h['lead_ci'][1], 0)}] | {_fmt(h['fa_per_well_day'], 3)} "
            f"[{_fmt(h['fa_ci'][0], 3)}, {_fmt(h['fa_ci'][1], 3)}] | {100 * h['time_in_alarm']:.1f}% | {_fmt(h['class4_recall'])} |")
    md += [
        "",
        "## Head-to-head on the same events (P2)",
        "",
        "Difference in warned-before-established on the same event instances, well-cluster bootstrap CI.",
        "",
        "| Comparison | Difference [95% CI] | Events | Only A warned | Only B warned |",
        "|---|---|---|---|---|",
        f"| gbdt − physics_rules | {vs_rules['diff']:+.2f} [{vs_rules['lo']:+.2f}, {vs_rules['hi']:+.2f}] | {vs_rules['n']} | {vs_rules['a_only']} | {vs_rules['b_only']} |",
        f"| gbdt − gbdt_no_dropout | {vs_nodrop['diff']:+.2f} [{vs_nodrop['lo']:+.2f}, {vs_nodrop['hi']:+.2f}] | {vs_nodrop['n']} | {vs_nodrop['a_only']} | {vs_nodrop['b_only']} |",
        f"| gbdt_sim_trained − physics_rules | {st_vs_rules['diff']:+.2f} [{st_vs_rules['lo']:+.2f}, {st_vs_rules['hi']:+.2f}] | {st_vs_rules['n']} | {st_vs_rules['a_only']} | {st_vs_rules['b_only']} |",
        f"| gbdt_sim_trained − gbdt | {st_vs_gb['diff']:+.2f} [{st_vs_gb['lo']:+.2f}, {st_vs_gb['hi']:+.2f}] | {st_vs_gb['n']} | {st_vs_gb['a_only']} | {st_vs_gb['b_only']} |",
        "",
        "## Sim-to-real gap",
        "",
        f"A classifier separating simulated from real normal rows, grouped by well/instance: ROC-AUC "
        f"**{gap['raw']:.3f}** on rows as they are, **{gap['matched']:.3f}** after giving simulated rows real "
        f"sensor-availability patterns. The second number is the gap in the sensor VALUES themselves.",
        "",
        "## Per-class results, gbdt under P2",
        "",
        per_class(gb)[["n_instances", "n_wells", "event_recall", "n_uncensored", "warned_before_established",
                       "median_delay_min", "median_lead_min", "pre_onset_alerts_per_inst"]].round(2).to_markdown(),
        "",
        "## Operating curve on test folds, gbdt P2 (reference only, not used to choose thresholds)",
        "",
        curve.round(4).to_markdown(index=False),
        "",
        "## Does the training-side threshold carry over to unseen wells?",
        "",
        "Per fold: false alarms and time in alarm on the training wells (where the threshold was set, "
        "out of fold) next to the same quantities on that fold's held-out wells.",
        "",
    ]
    fold_of = {w: f.k for f in folds for w in f.test_wells}
    rows = []
    for r, _ in results.values():
        for k, c in sorted(r.calibrations.items()):
            fa = [x for x in r.fa if fold_of.get(x.well) == k]
            expo = sum(x.exposure_s for x in fa)
            rows.append(dict(model=r.scorer, protocol=r.protocol, fold=k, on_percentile=c.on_percentile,
                             train_fa_per_day=round(c.train_fa_per_day, 3),
                             test_fa_per_day=round(sum(x.episodes for x in fa) / (expo / 86_400), 3) if expo else np.nan,
                             train_in_alarm_pct=round(100 * c.train_time_in_alarm, 2),
                             test_in_alarm_pct=round(100 * sum(x.time_in_alarm_s for x in fa) / expo, 2) if expo else np.nan,
                             meets_budget=c.meets_budget))
    md += [pd.DataFrame(rows).to_markdown(index=False), ""]
    (out / "phase3_model.md").write_text("\n".join(md), encoding="utf-8")
    print(f"vs physics_rules: {vs_rules}\nvs no-dropout: {vs_nodrop}\ndomain gap: {gap}")
    print(f"Done in {time.perf_counter() - t0:.0f} s -> {out / 'phase3_model.md'}")


if __name__ == "__main__":
    main()
