"""Phase 2: run every baseline under every protocol and write the results report.

Usage (PowerShell, from the repo root, after scripts/build_features.py):
    python scripts/run_baselines.py                       # all 4 baselines x 4 protocols
    python scripts/run_baselines.py --scorers zscore      # one baseline
    python scripts/run_baselines.py --boot 200            # faster, rougher CIs

Writes:
    reports/phase2_baselines.md     headline table, per-class tables, operating curves
    reports/phase2_outcomes.csv     one row per (scorer, protocol, event instance)
    reports/phase2_false_alarms.csv one row per (scorer, protocol, well)
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ew3w.alerting import HysteresisConfig  # noqa: E402
from ew3w.baselines import SCORERS  # noqa: E402
from ew3w.evaluation import TIME_IN_ALARM_BUDGET, headline, load_cache, operating_curve, per_class, run_protocol  # noqa: E402
from ew3w.features import FeatureConfig  # noqa: E402
from ew3w.splits import PROTOCOLS, make_folds  # noqa: E402

FA_BUDGET = 1 / 7  # 1 false alarm per well-week


def _fmt(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _ci(t, nd=2):
    return f"[{_fmt(t[0], nd)}, {_fmt(t[1], nd)}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="feature cache folder (default: data/features/<current config key>)")
    ap.add_argument("--scorers", nargs="+", default=list(SCORERS), choices=list(SCORERS))
    ap.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    ap.add_argument("--folds", default="data/well_folds.csv")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples for CIs")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    cache = Path(a.cache) if a.cache else Path("data/features") / FeatureConfig().key
    if not cache.exists():
        sys.exit(f"Feature cache {cache} not found: run scripts/build_features.py first.")
    if not Path(a.folds).exists():
        sys.exit(f"{a.folds} not found: run scripts/audit_dataset.py first.")
    folds = make_folds(pd.read_csv(a.folds, index_col=0)["fold"].to_dict())
    t0 = time.perf_counter()
    alert = HysteresisConfig(period_s=FeatureConfig().block_s)

    results, heads, curves = [], [], {}
    for s in a.scorers:
        # Load only this scorer's columns (and only normal synthetic rows if it never
        # trains on events): keeps peak memory under ~3 GB on an 8 GB laptop.
        S = SCORERS[s]
        t1 = time.perf_counter()
        real, synth = load_cache(cache, S.columns, synth_normal_only=not S.needs_events)
        print(f"{s}: loaded {len(S.columns)} columns, real {len(real):,} / synthetic {len(synth):,} rows "
              f"({time.perf_counter() - t1:.0f} s)", flush=True)
        for p in a.protocols:
            print(f"{s} / {p}", flush=True)
            r = run_protocol(SCORERS[s], p, real, synth, folds, alert, FA_BUDGET)
            results.append(r)
            heads.append(headline(r, n_boot=a.boot))
            if p == "P2_sim_plus_site_normal":
                curves[s] = operating_curve(r, real, folds, alert)
            h = heads[-1]
            print(f"  warned {_fmt(h['warned_before_established'])}  recall {_fmt(h['event_recall'])}  "
                  f"FA {_fmt(h['fa_per_well_day'], 3)}/well-day  ({r.seconds:.0f} s)", flush=True)
            r.test_percentiles.clear()  # curves done; free memory
        del real, synth
        gc.collect()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([r.outcomes.assign(scorer=r.scorer, protocol=r.protocol) for r in results]).to_csv(
        out / "phase2_outcomes.csv", index=False)
    pd.DataFrame([dict(asdict(f), scorer=r.scorer, protocol=r.protocol) for r in results for f in r.fa]).to_csv(
        out / "phase2_false_alarms.csv", index=False)

    md = [
        "# Phase 2: baseline results",
        "",
        f"Out-of-fold over 5 well-grouped folds (every well tested once). Operating point chosen on "
        f"TRAINING data only: lowest threshold with <= {FA_BUDGET:.3f} false alarms per well-day "
        f"(1 per well-week) AND <= {100 * TIME_IN_ALARM_BUDGET:.0f}% time in alarm on training normal data. "
        f"Alerter: {alert.on_delay_s} s to raise, {alert.off_delay_s} s to clear, "
        f"{alert.period_s} s rows. Detection grace before onset: 10 min. 95% CIs: well-cluster bootstrap "
        f"({a.boot} resamples) for event metrics, exact Poisson for false alarms.",
        "",
        "**Warned** = share of events (onset observable, established phase recorded) whose first alert "
        "came before the event became established. **Recall** = share of all scorable events detected "
        "at all. **Lead** = median minutes from first alert to establishment.",
        "",
        "| Baseline | Protocol | Warned [95% CI] | n | Recall | Lead, min [95% CI] | FA / well-day [95% CI] | Time in alarm | Class 4 recall |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for h in heads:
        md.append(
            f"| {h['scorer']} | {h['protocol']} | {_fmt(h['warned_before_established'])} {_ci(h['warned_ci'])} | "
            f"{h['n_lead_defined']} | {_fmt(h['event_recall'])} | {_fmt(h['median_lead_min'], 0)} {_ci(h['lead_ci'], 0)} | "
            f"{_fmt(h['fa_per_well_day'], 3)} {_ci(h['fa_ci'], 3)} | {100 * h['time_in_alarm']:.1f}% | {_fmt(h['class4_recall'])} |"
        )
    md += ["", f"Normal exposure: {heads[0]['normal_well_days']:.1f} well-days; scorable real events: {heads[0]['n_events']}.", ""]
    md += ["## Per-class results, P2 (simulated events + site normal history)", ""]
    for r in results:
        if r.protocol == "P2_sim_plus_site_normal":
            t = per_class(r)[["n_instances", "n_wells", "event_recall", "n_uncensored", "warned_before_established",
                              "median_delay_min", "median_lead_min", "pre_onset_alerts_per_inst"]]
            md += [f"### {r.scorer}", "", t.round(2).to_markdown(), ""]
    md += ["## Operating curves on test folds, P2 (reference only, not used to choose thresholds)", ""]
    for s, c in curves.items():
        md += [f"### {s}", "", c.round(4).to_markdown(index=False), ""]
    md += ["## Training-side calibration per fold", ""]
    rows = []
    for r in results:
        for k, c in sorted(r.calibrations.items()):
            rows.append(dict(scorer=r.scorer, protocol=r.protocol, fold=k, on_percentile=c.on_percentile,
                             train_fa_per_day=round(c.train_fa_per_day, 3),
                             train_time_in_alarm_pct=round(100 * c.train_time_in_alarm, 2),
                             train_normal_days=round(c.train_normal_days, 1), meets_budget=c.meets_budget))
    md += [pd.DataFrame(rows).to_markdown(index=False), ""]
    (out / "phase2_baselines.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Done in {time.perf_counter() - t0:.0f} s -> {out / 'phase2_baselines.md'}")


if __name__ == "__main__":
    main()
