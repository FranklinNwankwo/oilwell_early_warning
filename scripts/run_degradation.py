"""Phase 4: sensor-degradation study. Writes reports/phase4_degradation.md + .png + .csv.

Usage (PowerShell, from the repo root, with EW3W_DATASET set; after build_features,
audit_dataset):
    python scripts/run_degradation.py               # ~2-3 h on a laptop
    python scripts/run_degradation.py --resume      # continue after an interruption (checkpoint per fold)
    python scripts/run_degradation.py --skip-frozen # missing-sensor conditions only (no raw data needed)

Design (decided after Phase 3: physics rules are the primary detector):
  detectors  physics_rules (primary), gbdt_sim_trained (comparison), and
             gbdt_sim_trained_no_dropout (ablation: what dropout buys under failure),
             all under P2 (simulated events + the site's own normal history).
  training   each detector is fit and its threshold set ONCE per fold on CLEAN training
             wells (evaluation.fit_fold). Nothing is refit or re-thresholded per condition:
             a real deployment cannot re-tune when a sensor dies.
  conditions on the held-out wells only:
             clean; every subset of 1-4 sensors MISSING from the start (30 subsets);
             each sensor FROZEN once per instance (within the hour before each event's
             onset, anywhere inside normal instances), injected into raw data and rebuilt.
  metrics    warned before established, recall, false alarms per well-day, time in alarm,
             plus two blindness measures: share of rows where the detector has NO score
             while the system is NOT degraded (silent blindness), and events whose onset
             falls while the system is DEGRADED (fewer than 2 usable sensors).
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ew3w.alerting import HysteresisConfig  # noqa: E402
from ew3w.baselines import VALUE_COLS, PhysicsRules  # noqa: E402
from ew3w.data import discover, stitch_well  # noqa: E402
from ew3w.degradation import drop_sensors, freeze_raw, missing_conditions  # noqa: E402
from ew3w.evaluation import evaluate_wells, fit_fold, load_cache  # noqa: E402
from ew3w.features import FeatureConfig, build_features  # noqa: E402
from ew3w.metrics import outcomes_frame, poisson_rate_ci  # noqa: E402
from ew3w.model import HEALTH_CODE_COLS, GBDTSimTrained, GBDTSimTrainedNoDropout  # noqa: E402
from ew3w.schema import SHARED_SENSORS, TRANSIENT_OFFSET  # noqa: E402
from ew3w.splits import make_folds  # noqa: E402

P2 = "P2_sim_plus_site_normal"
FA_BUDGET = 1 / 7
DETECTORS = [PhysicsRules, GBDTSimTrained, GBDTSimTrainedNoDropout]
LABEL = {"physics_rules": "Physics rules (primary)", "gbdt_sim_trained": "GBDT, sim-trained, dropout",
         "gbdt_sim_trained_no_dropout": "GBDT, sim-trained, no dropout"}
COLS = sorted(set(VALUE_COLS) | set(HEALTH_CODE_COLS))
META_KEEP = ["class", "state", "event_class", "instance_id", "segment", "well", "source", "n_ok"]


def blind_onsets(test: pd.DataFrame) -> tuple[int, int]:
    """(event instances whose onset row has < 2 usable sensors, scorable event instances)."""
    ev = test["event_class"].to_numpy()
    lab = test["class"].to_numpy(dtype=np.float64)
    inst = test["instance_id"].astype(str).to_numpy()
    nok = test["n_ok"].to_numpy()
    blind = total = 0
    for iid in pd.unique(inst[ev > 0]):
        k = np.flatnonzero(inst == iid)
        c = int(ev[k[0]])
        hit = np.flatnonzero(np.isin(lab[k], [c, TRANSIENT_OFFSET + c]))
        if hit.size == 0:
            continue
        tr = np.flatnonzero(lab[k] == TRANSIENT_OFFSET + c)
        i_on = k[tr[0]] if tr.size else k[hit[0]]
        total += 1
        blind += int(nok[i_on] < 2)
    return blind, total


@dataclass
class Acc:
    """Running totals for one (condition, detector) across folds. Picklable for checkpoints."""
    outcomes: list = field(default_factory=list)
    fa: list = field(default_factory=list)
    nan_live: int = 0
    live: int = 0


def score_fold(fitted, fold_k, test, alert, condition, store, blind):
    """Score one fold's (already degraded) test rows with every frozen detector; add to totals."""
    for name, per_fold in fitted.items():
        sc, cal = per_fold[fold_k]
        p = cal.to_percentile(sc.score(test))
        o, f = evaluate_wells(test, p, cal, alert)
        a = store.setdefault((condition, name), Acc())
        a.outcomes += o
        a.fa += f
        not_deg = test["n_ok"].to_numpy() >= 2
        a.nan_live += int((np.isnan(p) & not_deg).sum())
        a.live += int(not_deg.sum())
    b, t = blind_onsets(test)
    prev = blind.get(condition, (0, 0))
    blind[condition] = (prev[0] + b, prev[1] + t)


def summarise(store, blind, cond_meta):
    """(outcome rows, one summary row per condition x detector) from the running totals."""
    outs, summ = [], []
    for (condition, name), a in store.items():
        mode, k = cond_meta[condition]
        df = outcomes_frame(a.outcomes)
        lead = df[df.onset_observable & ~df.censored]
        eps = sum(x.episodes for x in a.fa)
        expo = sum(x.exposure_s for x in a.fa)
        r, lo, hi = poisson_rate_ci(eps, expo / 86_400)
        b, t = blind[condition]
        summ.append(dict(condition=condition, mode=mode, k=k, detector=name,
                         warned=lead.warned_before_established.mean(), n_warned=len(lead),
                         recall=df.detected.mean(), fa_per_well_day=r, fa_lo=lo, fa_hi=hi,
                         time_in_alarm=sum(x.time_in_alarm_s for x in a.fa) / expo,
                         silent_blind_pct=100 * a.nan_live / max(a.live, 1), blind_onsets=b, events=t))
        outs.append(df.assign(condition=condition, mode=mode, k=k, detector=name))
    return outs, summ


def cluster_ci(values: pd.Series, wells: pd.Series, n_boot: int, seed: int = 0) -> tuple[float, float, float]:
    """Mean of per-event values with a well-cluster bootstrap CI."""
    v = values.to_numpy(dtype=float)
    w = wells.astype(str).to_numpy()
    uniq = np.unique(w)
    idx = {u: np.flatnonzero(w == u) for u in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[u] for u in rng.choice(uniq, uniq.size)])
        boots.append(v[pick].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(v.mean()), float(lo), float(hi)


def by_k_table(outcomes: pd.DataFrame, summary: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Per detector and number of missing sensors: warned and recall averaged over every
    subset of that size, per event first (each event weighs once), then well-bootstrapped."""
    rows = []
    miss = outcomes[outcomes["mode"].isin(["clean", "missing"])]
    for (det, k), g in miss.groupby(["detector", "k"]):
        lead = g[g.onset_observable & ~g.censored]
        we = lead.groupby("instance_id", observed=True).agg(v=("warned_before_established", "mean"), well=("well", "first"))
        re = g.groupby("instance_id", observed=True).agg(v=("detected", "mean"), well=("well", "first"))
        w = cluster_ci(we.v, we.well, n_boot)
        r = cluster_ci(re.v, re.well, n_boot)
        s = summary[(summary.detector == det) & (summary.k == k) & summary["mode"].isin(["clean", "missing"])]
        rows.append(dict(detector=det, k=k, subsets=len(s), warned=w[0], warned_lo=w[1], warned_hi=w[2],
                         recall=r[0], recall_lo=r[1], recall_hi=r[2],
                         fa_mean=s.fa_per_well_day.mean(), fa_min=s.fa_per_well_day.min(), fa_max=s.fa_per_well_day.max(),
                         silent_blind_pct=s.silent_blind_pct.mean(), blind_onset_share=(s.blind_onsets / s.events).mean()))
    return pd.DataFrame(rows)


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label y-positions apart (keeping order) so no two are closer than `gap`."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        out[b] = max(out[b], out[a] + gap)
    return out.tolist()


def plot(byk: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reference palette slots 1-3 (validated all-pairs, light surface #fcfcfb); aqua is
    # below 3:1 contrast, so every line is direct-labeled and marker shapes differ.
    color = {"physics_rules": "#2a78d6", "gbdt_sim_trained": "#eb6834", "gbdt_sim_trained_no_dropout": "#1baf7a"}
    marker = {"physics_rules": "o", "gbdt_sim_trained": "s", "gbdt_sim_trained_no_dropout": "^"}
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": ink2, "xtick.color": ink2,
                         "ytick.color": ink2, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=surf)
    offs = {"physics_rules": -0.08, "gbdt_sim_trained": 0.0, "gbdt_sim_trained_no_dropout": 0.08}
    for ax, (col, lo, hi, panel_title, ylab) in zip(axes, [
        ("warned", "warned_lo", "warned_hi", "Warned before the event is established", "share of events (95% CI)"),
        ("fa_mean", "fa_min", "fa_max", "False alarms on normal operation", "per well-day (range over subsets)"),
    ]):
        ax.set_facecolor(surf)
        ends = []
        for det, g in byk.groupby("detector"):
            g = g.sort_values("k")
            x = g.k.to_numpy() + offs[det]
            y = g[col].to_numpy()
            ax.errorbar(x, y, yerr=[y - g[lo].to_numpy(), g[hi].to_numpy() - y], color=color[det], lw=2,
                        marker=marker[det], ms=8, mec=surf, mew=1.5, capsize=0, elinewidth=1.2, label=LABEL[det])
            at = int(np.flatnonzero(g.k.to_numpy() == 3)[0]) if (g.k == 3).any() else -1
            ends.append((det, x[at], y[at]))  # label at k=3: at k=4 every line sits on 0
        if col == "fa_mean":
            top = max(byk[hi].max(), FA_BUDGET) * 1.15
            ax.set_ylim(0, top)
        else:
            top = 1.0
            ax.set_ylim(0, 1)
        short = {"physics_rules": "Physics rules", "gbdt_sim_trained": "GBDT + dropout",
                 "gbdt_sim_trained_no_dropout": "GBDT, no dropout"}
        ys = _spread([e[2] for e in ends], gap=0.07 * top)
        for (det, x_end, y_end), y_lab in zip(ends, ys):
            ax.annotate(short[det], (x_end, y_end), xytext=(3.35, y_lab + 0.06 * top), textcoords="data", va="center",
                        fontsize=9, color=ink2, arrowprops=dict(arrowstyle="-", color=grid, lw=0.8))
        if col == "fa_mean":
            ax.axhline(FA_BUDGET, color=ink2, lw=1, ls=(0, (4, 3)))
            ax.annotate("budget: 1 per well-week", (5.55, FA_BUDGET), xytext=(0, 4), textcoords="offset points",
                        ha="right", fontsize=9, color=ink2)
        ax.set_title(panel_title, loc="left", color=ink, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylab)
        ax.set_xlabel("sensors missing (of the 5 shared)")
        ax.set_xticks(range(0, 5))
        ax.set_xlim(-0.4, 5.6)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", color=ink, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=160, facecolor=surf)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.environ.get("EW3W_DATASET"), help="needed for frozen conditions")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--folds", default="data/well_folds.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--skip-frozen", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from reports/phase4_checkpoint.pkl")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--title", default="Physics rules stay ahead at every level of sensor loss; failures silence alarms rather than add false ones",
                    help="chart headline (states the Phase 4 finding; the run is deterministic)")
    ap.add_argument("--replot", action="store_true",
                    help="rebuild the report and chart from reports/phase4_*.csv without re-running anything")
    a = ap.parse_args()
    out = Path(a.out)
    if a.replot:
        outcomes = pd.read_csv(out / "phase4_outcomes.csv")
        summary = pd.read_csv(out / "phase4_conditions.csv")
        write_report(outcomes, summary, out, a.title, a.boot, include_frozen=(summary["mode"] == "frozen").any())
        print(f"Rebuilt {out / 'phase4_degradation.md'} and the chart from saved results")
        return
    cfg = FeatureConfig()
    cache = Path(a.cache) if a.cache else Path("data/features") / cfg.key
    if not cache.exists():
        sys.exit(f"Feature cache {cache} not found: run scripts/build_features.py first.")
    if not a.skip_frozen and not a.dataset:
        sys.exit("Frozen conditions rebuild features from raw data: set EW3W_DATASET, or pass --skip-frozen.")
    folds = make_folds(pd.read_csv(a.folds, index_col=0)["fold"].to_dict())
    alert = HysteresisConfig(period_s=cfg.block_s)
    t0 = time.perf_counter()

    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "phase4_checkpoint.pkl"
    state = {"done": set(), "store": {}, "blind": {}}
    if a.resume and ckpt.exists():
        state = pickle.loads(ckpt.read_bytes())
        print(f"Resuming: {len(state['done'])} work units already done", flush=True)

    def save() -> None:
        tmp = ckpt.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(state))
        os.replace(tmp, ckpt)

    conds = [("clean", "clean", ())] + [("missing:" + "+".join(c), "missing", c) for c in missing_conditions()]
    cond_meta = {n: (m, len(c)) for n, m, c in conds}
    if not a.skip_frozen:
        cond_meta.update({f"frozen:{s}": ("frozen", 1) for s in SHARED_SENSORS})

    # Fit once per fold on CLEAN training wells (deterministic, so a resumed run refits identically).
    real, synth = load_cache(cache, COLS, synth_event_cap=GBDTSimTrained().per_group)
    print(f"Loaded real {len(real):,} / synthetic {len(synth):,} rows", flush=True)
    fitted = defaultdict(dict)
    for S in DETECTORS:
        for fold in folds:
            sc, cal = fit_fold(S, P2, real, synth, fold, alert, FA_BUDGET)
            fitted[S.name][fold.k] = (sc, cal)
            print(f"  fit {S.name} fold {fold.k}: on-pct {cal.on_percentile:.6f} "
                  f"(train FA {cal.train_fa_per_day:.3f}/day, in alarm {100 * cal.train_time_in_alarm:.2f}%)", flush=True)
    del synth
    gc.collect()

    # Missing-sensor conditions: one fold at a time, so only that fold's rows are ever copied.
    for fold in folds:
        unit = ("missing", fold.k)
        if unit in state["done"]:
            continue
        base = real[real["well"].isin(fold.test_wells)]
        for name, mode, sensors in conds:
            score_fold(fitted, fold.k, drop_sensors(base, sensors), alert, name, state["store"], state["blind"])
        state["done"].add(unit)
        save()
        print(f"[missing] fold {fold.k} done ({len(conds)} conditions, {time.perf_counter() - t0:.0f} s)", flush=True)
        del base
        gc.collect()
    del real
    gc.collect()

    if not a.skip_frozen:
        by_well = defaultdict(list)
        for m in discover(a.dataset):
            if m.source == "real":
                by_well[m.well].append(m)
        for j, sensor in enumerate(SHARED_SENSORS):
            for fold in folds:
                unit = ("frozen", sensor, fold.k)
                if unit in state["done"]:
                    continue
                parts = []
                for w in fold.test_wells:
                    tl = freeze_raw(stitch_well(by_well[w], list(SHARED_SENSORS)), sensor, seed=j)
                    f = build_features(tl, cfg, detect_frozen=True)
                    f["well"], f["source"] = w, "real"
                    parts.append(f[[*COLS, *META_KEEP]])
                    del tl
                test = pd.concat(parts)
                for c in ("instance_id", "well", "source"):
                    test[c] = test[c].astype("category")
                score_fold(fitted, fold.k, test, alert, f"frozen:{sensor}", state["store"], state["blind"])
                state["done"].add(unit)
                save()
                print(f"[frozen {sensor}] fold {fold.k} done ({time.perf_counter() - t0:.0f} s)", flush=True)
                del parts, test
                gc.collect()

    outs, summ = summarise(state["store"], state["blind"], cond_meta)
    for x in sorted(summ, key=lambda r: (r["mode"] != "clean", r["k"], r["condition"])):
        print(f"  {x['condition']:<40} {x['detector']:<28} warned {x['warned']:.2f}  FA {x['fa_per_well_day']:.3f}")
    outcomes = pd.concat(outs, ignore_index=True)
    summary = pd.DataFrame(summ)
    summary.to_csv(out / "phase4_conditions.csv", index=False)
    outcomes.to_csv(out / "phase4_outcomes.csv", index=False)
    write_report(outcomes, summary, out, a.title, a.boot, include_frozen=not a.skip_frozen)
    ckpt.unlink(missing_ok=True)  # finished: the report and CSVs now hold everything
    print(f"Done in {time.perf_counter() - t0:.0f} s -> {out / 'phase4_degradation.md'}")


def write_report(outcomes: pd.DataFrame, summary: pd.DataFrame, out: Path, title: str, n_boot: int,
                 include_frozen: bool) -> None:
    byk = by_k_table(outcomes, summary, n_boot)
    plot(byk, out / "phase4_degradation.png", title)

    def pct(x):
        return f"{100 * x:.0f}%"

    md = ["# Phase 4: sensor-degradation study", "",
          "Detectors trained and thresholded once per fold on CLEAN training wells (P2: simulated events + "
          "site normal history; <= 1 false alarm per well-week and <= 1% time in alarm), then frozen. "
          "Failures are injected into the held-out wells only. Warned/recall CIs: well-cluster bootstrap over "
          "events, each event averaged over every sensor subset of that size first. False alarms: mean and "
          f"range over subsets. Bootstrap: {n_boot} resamples.", "",
          "![Performance vs missing sensors](phase4_degradation.png)", "",
          "## By number of missing sensors", "",
          "| Detector | Missing | Subsets | Warned [95% CI] | Recall [95% CI] | FA / well-day mean [min, max] | Silent blindness | Onsets while DEGRADED |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in byk.sort_values(["detector", "k"]).iterrows():
        md.append(f"| {LABEL[r.detector]} | {r.k} | {r.subsets} | {r.warned:.2f} [{r.warned_lo:.2f}, {r.warned_hi:.2f}] | "
                  f"{r.recall:.2f} [{r.recall_lo:.2f}, {r.recall_hi:.2f}] | {r.fa_mean:.3f} [{r.fa_min:.3f}, {r.fa_max:.3f}] | "
                  f"{r.silent_blind_pct:.1f}% | {pct(r.blind_onset_share)} |")
    md += ["", "**Silent blindness** = share of rows where the detector produces no score (its inputs are "
           "unavailable) while the system is NOT in DEGRADED, so the screen shows an all-clear the detector "
           "cannot back up. **Onsets while DEGRADED** = events that begin while fewer than 2 sensors are usable.", "",
           "**Design consequence:** the DEGRADED rule (fewer than 2 usable sensors) is detector-blind. Physics rules need "
           "specific sensor combinations, so with 2-3 sensors missing they often cannot evaluate at all while the "
           "status still reads normal. The status must also go DEGRADED whenever the detector has no score. That "
           "changes only what the operator is shown, not when alerts fire (a missing score already blocks new "
           "alerts and never clears an active one), so every number in this report is unaffected.", "",
           "## Single sensor missing", "",
           "| Sensor | " + " | ".join(LABEL[d.name] for d in DETECTORS) + " |",
           "|---|" + "---|" * len(DETECTORS)]
    one = summary[(summary["mode"] == "missing") & (summary.k == 1)]
    for s in SHARED_SENSORS:
        cells = []
        for d in DETECTORS:
            r = one[(one.condition == f"missing:{s}") & (one.detector == d.name)].iloc[0]
            cells.append(f"warned {r.warned:.2f}, FA {r.fa_per_well_day:.3f}, silent {r.silent_blind_pct:.0f}%")
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    if include_frozen:
        md += ["", "## Single sensor frozen before onset (raw injection, features rebuilt)", "",
               "| Sensor | " + " | ".join(LABEL[d.name] for d in DETECTORS) + " |",
               "|---|" + "---|" * len(DETECTORS)]
        fz = summary[summary["mode"] == "frozen"]
        clean = summary[summary["mode"] == "clean"].set_index("detector")
        for s in SHARED_SENSORS:
            cells = []
            for d in DETECTORS:
                r = fz[(fz.condition == f"frozen:{s}") & (fz.detector == d.name)].iloc[0]
                cells.append(f"warned {r.warned:.2f} ({r.warned - clean.loc[d.name, 'warned']:+.2f}), "
                             f"FA {r.fa_per_well_day:.3f} ({r.fa_per_well_day - clean.loc[d.name, 'fa_per_well_day']:+.3f})")
            md.append(f"| {s} | " + " | ".join(cells) + " |")
        md += ["", "Differences in brackets are against the clean condition."]
    md += ["", "## Clean reference", "", summary[summary["mode"] == "clean"][
        ["detector", "warned", "n_warned", "recall", "fa_per_well_day", "fa_lo", "fa_hi", "time_in_alarm", "silent_blind_pct"]
    ].round(3).to_markdown(index=False), ""]
    (out / "phase4_degradation.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
