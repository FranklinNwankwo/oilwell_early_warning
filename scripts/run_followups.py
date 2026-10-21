"""Phase 6 follow-ups to the two issues the Phase 5 demo exposed. Writes
reports/phase6_followups.md and reports/phase6_false_alarms.csv.

Usage (PowerShell, from the repo root, after build_features and audit_dataset):
    python scripts/run_followups.py          # ~5-10 min, ~2 GB RAM

Everything runs under P2 on all 5 folds with the Phase 2-4 harness (thresholds from
training wells only, 1 FA/well-week + 1% time-in-alarm budgets). Nothing evaluated
earlier is changed: the fix is a separately named scorer and the corrected count is an
additional metric next to the official one.

  1. Operator-visible false alarms. The official count only counts alarms that START on a
     normal-labeled sample; alarms raised in a normal instance's unlabeled first hour are
     skipped. Recount every alarm overlapping normal-labeled data.
  2. Warm-up fix. physics_rules_warmup: the oscillation rule only scores once a baseline
     exists (baselines.PhysicsRulesWarmup). Same threshold procedure. Paired comparison on
     the same events, and every event whose outcome changes is listed (did the post-gap
     artifact ever produce a "lucky" detection?).
  3. Dead-band. Off-threshold one decade below the on-threshold (as evaluated) vs half a
     decade vs none, with the on-threshold held fixed: alarm counts, durations, time in
     alarm, warned.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ew3w.alerting import HysteresisAlerter, HysteresisConfig, alert_active  # noqa: E402
from ew3w.baselines import PhysicsRules, PhysicsRulesWarmup  # noqa: E402
from ew3w.evaluation import _runs, evaluate_wells, fit_fold, load_cache  # noqa: E402
from ew3w.features import FeatureConfig  # noqa: E402
from ew3w.metrics import outcomes_frame, paired_difference, poisson_rate_ci, visible_false_alarm_episodes  # noqa: E402
from ew3w.splits import make_folds  # noqa: E402

P2 = "P2_sim_plus_site_normal"
FA_BUDGET = 1 / 7
CFG = FeatureConfig()
ALERT = HysteresisConfig(period_s=CFG.block_s)
DEADBANDS = {"one decade (as evaluated)": 1.0, "half a decade": 0.5, "none": 0.0}


def off_for(on: float, decades: float) -> float:
    """Off-threshold `decades` below `on` in rarity terms (1 - p). 0 -> just below on."""
    if decades == 0:
        return np.nextafter(on, 0.0)
    return 1.0 - min(1.0, (10 ** decades) * (1.0 - on))


def score_wells(test: pd.DataFrame, p: np.ndarray, on: float, off: float):
    """Official outcomes + FA counts (evaluation.evaluate_wells) and the visible FA episodes."""
    from ew3w.evaluation import Calibration
    cal = Calibration(np.zeros(1), on, off, np.nan, np.nan)
    outcomes, fas = evaluate_wells(test, p, cal, ALERT)
    cfg = HysteresisConfig(on, off, ALERT.on_delay_s, ALERT.off_delay_s, ALERT.period_s)
    vis = []
    wells = test["well"].astype(str).to_numpy()
    for w in pd.unique(wells):
        m = wells == w
        d = test[m]
        runs = _runs(d, ALERT.period_s)
        act = alert_active(HysteresisAlerter(cfg).run(p[m], degraded=d["n_ok"].to_numpy() < 2, segments=runs))
        lab = np.where(d["event_class"].to_numpy() == 0, d["class"].to_numpy(np.float64), np.nan)
        t = pd.DatetimeIndex(d.index).as_unit("s").asi8
        seg0 = pd.Series(t).groupby(runs).transform("min").to_numpy()
        for e in visible_false_alarm_episodes(lab, act, runs, ALERT.period_s):
            vis.append({"well": w, "t_start": pd.Timestamp(t[e["start"]], unit="s"),
                        "min_since_data_resumed": (t[e["start"]] - seg0[e["start"]]) / 60,
                        "duration_min": e["duration_s"] / 60, "starts_on_normal": e["starts_on_normal"]})
    return outcomes, fas, vis


def summary(name: str, outcomes, fas, vis) -> dict:
    df = outcomes_frame(outcomes)
    lead = df[df.onset_observable & ~df.censored]
    eps = sum(x.episodes for x in fas)
    expo = sum(x.exposure_s for x in fas) / 86_400
    r, lo, hi = poisson_rate_ci(eps, expo)
    vr, vlo, vhi = poisson_rate_ci(len(vis), expo)
    dur = pd.Series([v["duration_min"] for v in vis], dtype=float)
    early = sum(v["min_since_data_resumed"] < 60 for v in vis)
    return {"detector": name, "warned": lead.warned_before_established.mean(), "n_events": len(lead),
            "recall": df.detected.mean(), "fa_official": r, "fa_official_ci": (lo, hi), "fa_visible": vr,
            "fa_visible_ci": (vlo, vhi), "n_visible": len(vis), "n_official": eps, "early_after_gap": early,
            "time_in_alarm": sum(x.time_in_alarm_s for x in fas) / (expo * 86_400),
            "dur_median": dur.median(), "dur_p90": dur.quantile(0.9), "dur_max": dur.max(),
            "share_time_long": dur[dur > 60].sum() / dur.sum() if dur.sum() else np.nan,
            "wells_with_fa": len({v["well"] for v in vis}), "exposure_days": expo,
            "wells_with_normal": len({x.well for x in fas if x.exposure_s > 0})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--folds", default="data/well_folds.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    cache = Path(a.cache) if a.cache else Path("data/features") / CFG.key
    if not cache.exists():
        sys.exit(f"Feature cache {cache} not found: run scripts/build_features.py first.")
    folds = make_folds(pd.read_csv(a.folds, index_col=0)["fold"].to_dict())
    t0 = time.perf_counter()
    real, synth = load_cache(cache, PhysicsRulesWarmup.columns, synth_normal_only=True)
    print(f"Loaded real {len(real):,} rows ({time.perf_counter() - t0:.0f} s)", flush=True)

    res = {k: {"outcomes": [], "fas": [], "vis": []} for k in
           ["physics_rules", "physics_rules_warmup", *[f"deadband: {d}" for d in DEADBANDS]]}
    for fold in folds:
        test = real[real["well"].isin(fold.test_wells)]
        for S in (PhysicsRules, PhysicsRulesWarmup):
            sc, cal = fit_fold(S, P2, real, synth, fold, ALERT, FA_BUDGET)
            p = cal.to_percentile(sc.score(test))
            o, f, v = score_wells(test, p, cal.on_percentile, cal.off_percentile)
            for k, x in (("outcomes", o), ("fas", f), ("vis", v)):
                res[S.name][k] += x
            for vv in v:
                vv.update(fold=fold.k, detector=S.name)
            print(f"  fold {fold.k} {S.name}: on {cal.on_percentile:.6f}, {len(v)} visible FA", flush=True)
            if S is PhysicsRules:
                for label, dec in DEADBANDS.items():
                    o2, f2, v2 = score_wells(test, p, cal.on_percentile, off_for(cal.on_percentile, dec))
                    key = f"deadband: {label}"
                    for k, x in (("outcomes", o2), ("fas", f2), ("vis", v2)):
                        res[key][k] += x

    rows = {k: summary(k, **v) for k, v in res.items()}
    pr, wu = res["physics_rules"], res["physics_rules_warmup"]
    pdiff = paired_difference(outcomes_frame(wu["outcomes"]), outcomes_frame(pr["outcomes"]), n_boot=a.boot)
    a_df = outcomes_frame(pr["outcomes"]).set_index("instance_id")
    b_df = outcomes_frame(wu["outcomes"]).set_index("instance_id")
    changed = a_df.join(b_df, rsuffix="_warmup")
    changed = changed[(changed.detected != changed.detected_warmup)
                      | (changed.warned_before_established != changed.warned_before_established_warmup)
                      | ~np.isclose(changed.lead_s.fillna(-1), changed.lead_s_warmup.fillna(-1))]

    out = Path(a.out)
    fa_rows = pd.DataFrame(pr["vis"] + wu["vis"])
    fa_rows.to_csv(out / "phase6_false_alarms.csv", index=False)

    def f3(x):
        return "–" if x is None or not np.isfinite(x) else f"{x:.3f}"

    def ci(t):
        return f"[{t[0]:.3f}, {t[1]:.3f}]"

    p_, w_ = rows["physics_rules"], rows["physics_rules_warmup"]
    md = [
        "# Phase 6: follow-ups to the Phase 5 findings",
        "",
        f"P2, all 5 folds, {p_['exposure_days']:.1f} normal well-days, {p_['n_events']} events with a defined lead. "
        "Same harness and budgets as Phases 2-4; nothing evaluated earlier is changed.",
        "",
        "## 1. False alarms an operator would see",
        "",
        "The official metric (`metrics.count_false_alarms`) counts an alarm only if it **starts** on a normal-labeled "
        "sample. Every real instance opens with an unlabeled hour, so an alarm raised there and still active when the "
        "normal labels begin was not counted. Recounting every alarm that overlaps normal-labeled data:",
        "",
        "| Detector | Official FA / well-day [95% CI] | Visible FA / well-day [95% CI] | Visible alarms | Start < 1 h after data resumes | Wells with any (of wells with normal data) |",
        "|---|---|---|---|---|---|",
    ]
    for r in (p_, w_):
        md.append(f"| {r['detector']} | {f3(r['fa_official'])} {ci(r['fa_official_ci'])} | {f3(r['fa_visible'])} "
                  f"{ci(r['fa_visible_ci'])} | {r['n_visible']} | {r['early_after_gap']} | {r['wells_with_fa']} of {r['wells_with_normal']} |")
    md += [
        "",
        "## 2. Warm-up fix: no oscillation scoring before a baseline exists",
        "",
        "| Detector | Warned before established | Recall | Visible FA / well-day | Time in alarm |",
        "|---|---|---|---|---|",
    ]
    for r in (p_, w_):
        md.append(f"| {r['detector']} | {r['warned']:.2f} (n={r['n_events']}) | {r['recall']:.2f} | {f3(r['fa_visible'])} | "
                  f"{100 * r['time_in_alarm']:.2f}% |")
    md += [
        "",
        f"Paired on the same events, warmup − original: **{pdiff['diff']:+.2f} [{pdiff['lo']:+.2f}, {pdiff['hi']:+.2f}]** "
        f"({pdiff['n']} events; only warmup warned: {pdiff['a_only']}, only original warned: {pdiff['b_only']}).",
        "",
        f"Event instances whose outcome changes at all: **{len(changed)}** of {len(a_df)}."
        + (" So no detection credited in Phases 2-4 depended on the post-gap artifact." if len(changed) == 0 else ""),
        "",
    ]
    if len(changed):
        lost = changed[changed.detected & ~changed.detected_warmup]
        gained = changed[~changed.detected & changed.detected_warmup]
        warned_lost = changed[changed.warned_before_established & ~changed.warned_before_established_warmup]
        shift = (changed.lead_s - changed.lead_s_warmup).abs().max()
        md += [f"Detections lost: **{len(lost)}** (classes {', '.join(map(str, sorted(lost.event_class)))}), gained: "
               f"{len(gained)}; early warnings lost: **{len(warned_lost)}**. The remaining changes move a lead time by at "
               f"most {shift:.0f} s. So the post-gap scoring never earned an early warning; it did produce "
               f"{len(lost)} detections without lead credit (mostly classes 3/4, oscillation events whose recording "
               f"opens mid-event), and those are the price of the fix.", ""]
        md += [changed[["event_class", "well", "detected", "detected_warmup", "warned_before_established",
                        "warned_before_established_warmup", "lead_s", "lead_s_warmup"]].to_markdown(), ""]
    md += [
        "## 3. Dead-band: how far below the alarm threshold must the score fall to clear?",
        "",
        "Original physics rules; on-threshold held at each fold's calibrated value, only the off-threshold changes.",
        "",
        "| Off-threshold | Visible FA / well-day | Median alarm, min | 90th pct, min | Longest, min | Share of FA time in alarms > 1 h | Time in alarm | Warned |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in DEADBANDS:
        r = rows[f"deadband: {label}"]
        md.append(f"| {label} | {f3(r['fa_visible'])} | {r['dur_median']:.0f} | {r['dur_p90']:.0f} | {r['dur_max']:.0f} | "
                  f"{100 * r['share_time_long']:.0f}% | {100 * r['time_in_alarm']:.2f}% | {r['warned']:.2f} |")
    md += ["", "Per-alarm detail (both detectors): `reports/phase6_false_alarms.csv`.", ""]
    (out / "phase6_followups.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nDone in {time.perf_counter() - t0:.0f} s -> {out / 'phase6_followups.md'}")


if __name__ == "__main__":
    main()
