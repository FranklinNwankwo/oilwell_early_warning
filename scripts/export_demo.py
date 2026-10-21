"""Phase 5: freeze the demo's detectors and cut its replay clips. Run once (needs the
dataset and the feature cache); the app itself needs neither.

Usage (PowerShell, from the repo root, with EW3W_DATASET set):
    python scripts/export_demo.py            # ~5 min, ~3 GB RAM peak

What it does:
  1. Fits physics rules (primary) and the simulation-trained GBDT (comparison) under P2
     on the TRAINING wells of one fold (default fold 1), exactly as Phases 4 did
     (evaluation.fit_fold), and freezes their thresholds: models/detectors.pkl.
  2. Cuts replay clips from that fold's HELD-OUT wells, which neither detector has seen:
     a curated set of real events (both warn, only one warns, both miss), one quiet
     stretch of normal operation and the first physics-rules false alarm in a normal
     instance. Each clip carries 7.5 h of history before the replay starts, so the
     streaming features equal the batch features from the first replayed row.
     demo_data/*.parquet + demo_data/manifest.json.
  3. Streams every clip through ew3w.stream.EarlyWarningEngine and scores it with the
     Phase 4 metric code; prints the result next to the Phase 4 (batch) result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sklearn  # noqa: E402

from ew3w.alerting import HysteresisAlerter, HysteresisConfig, alert_active  # noqa: E402
from ew3w.baselines import VALUE_COLS, PhysicsRules  # noqa: E402
from ew3w.data import discover, stitch_well  # noqa: E402
from ew3w.evaluation import _runs, fit_fold, load_cache  # noqa: E402
from ew3w.features import FeatureConfig  # noqa: E402
from ew3w.metrics import evaluate_event_instance  # noqa: E402
from ew3w.model import HEALTH_CODE_COLS, GBDTSimTrained  # noqa: E402
from ew3w.schema import EVENT_NAMES, SHARED_SENSORS, TRANSIENT_OFFSET  # noqa: E402
from ew3w.splits import make_folds  # noqa: E402
from ew3w.stream import EarlyWarningEngine, FrozenDetector, TailCalibration, save_detectors  # noqa: E402

P2 = "P2_sim_plus_site_normal"
FA_BUDGET = 1 / 7
LABELS = {"physics_rules": "Physics rules (primary)", "gbdt_sim_trained": "Learned model, simulation-trained (comparison)"}
CFG = FeatureConfig()
ALERT = HysteresisConfig(period_s=CFG.block_s)

# Curated from Phase 4 (clean condition, fold 1 held-out wells). The stories are the
# measured Phase 4 outcomes; step 3 re-checks every one of them by streaming.
EPISODES = [
    ("dhsv_both_warn", "2/WELL-00011_20140515083000",
     "Spurious DHSV closure: both detectors warn about 3 hours before the event is established."),
    ("bsw_rules_only", "1/WELL-00001_20140124083303",
     "Abrupt BSW increase: physics rules warn hours ahead; the learned model never alarms."),
    ("productivity_both_warn", "5/WELL-00016_20180405005319",
     "Rapid productivity loss: both warn about 45 minutes before establishment."),
    ("productivity_model_only", "5/WELL-00016_20180426135108",
     "Rapid productivity loss: here the learned model warns and the physics rules miss."),
    ("scaling_rules_only", "7/WELL-00001_20170226130146",
     "Scaling in the production choke (slow): physics rules warn about 4 hours ahead."),
    ("service_hydrate_miss", "9/WELL-00016_20150418132043",
     "Hydrate in the service line: both miss. It acts on gas-lift sensors outside the shared set."),
]
PRE_ROLL_S = 3600  # replay starts this long before the onset
POST_S = 1800  # ... and ends this long after establishment
MAX_REPLAY_S = 10 * 3600


def _fmt_lead(o) -> str:
    if not o["detected"]:
        return "missed"
    if o["lead_min"] is None:
        return "detected (no lead defined)"
    return f"warned {o['lead_min']:.0f} min before established" if o["warned"] else f"late ({-o['lead_min']:.0f} min after)"


def cut(tl: pd.DataFrame, replay_start: pd.Timestamp, replay_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Clip = [replay_start - history, replay_end], never crossing a data gap before the
    replay starts (state resets there anyway, exactly as in batch)."""
    seg = tl.loc[:replay_start, "segment"].iloc[-1]
    seg_start = tl.index[tl["segment"].to_numpy() == seg][0]
    start = max(seg_start, replay_start - pd.Timedelta(seconds=CFG.history_s + 60))
    clip = tl.loc[start:replay_end, [*SHARED_SENSORS, "class", "state", "event_class", "instance_id"]].copy()
    clip["instance_id"] = clip["instance_id"].astype(str)
    return clip, max(replay_start, clip.index[0])  # replay_start may fall in a data gap


def stream_clip(clip: pd.DataFrame, detectors: list[FrozenDetector]) -> pd.DataFrame:
    eng = EarlyWarningEngine(detectors, CFG, detect_frozen=True, alert=ALERT)
    t = pd.DatetimeIndex(clip.index).as_unit("ns").asi8
    ticks = eng.push_many(t, clip[list(SHARED_SENSORS)].to_numpy())
    idx = pd.DatetimeIndex([k.t_ns for k in ticks]).as_unit("ns")
    out = pd.DataFrame({"segment": [k.segment for k in ticks], "n_ok": [k.n_ok for k in ticks]}, index=idx)
    for d in detectors:
        out[f"{d.name}__status"] = [int(k.status[d.name]) for k in ticks]
    meta = clip.loc[idx, ["class", "instance_id"]]
    out["class"] = meta["class"].to_numpy()
    out["instance_id"] = meta["instance_id"].to_numpy()
    return out


def outcome(rows: pd.DataFrame, det: str, iid: str, event_class: int) -> dict:
    k = rows["instance_id"].to_numpy() == iid
    r = rows[k]
    t = pd.DatetimeIndex(r.index).as_unit("ns").asi8 / 1e9
    act = alert_active(r[f"{det}__status"].to_numpy())
    o = evaluate_event_instance(t, r["class"].to_numpy(np.float64), act, event_class, instance_id=iid,
                                segments=r["segment"].to_numpy(), period_s=CFG.block_s)
    lead = None if not np.isfinite(o.lead_s) else round(o.lead_s / 60, 1)
    return {"detected": bool(o.detected), "warned": bool(o.warned_before_established), "lead_min": lead}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.environ.get("EW3W_DATASET"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--folds", default="data/well_folds.csv")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--models", default="models")
    ap.add_argument("--clips", default="demo_data")
    a = ap.parse_args()
    if not a.dataset:
        sys.exit("Set EW3W_DATASET (or pass --dataset): the clips are cut from the raw 1 Hz data.")
    cache = Path(a.cache) if a.cache else Path("data/features") / CFG.key
    if not cache.exists():
        sys.exit(f"Feature cache {cache} not found: run scripts/build_features.py first.")
    folds = make_folds(pd.read_csv(a.folds, index_col=0)["fold"].to_dict())
    fold = next(f for f in folds if f.k == a.fold)
    t0 = time.perf_counter()

    # ---- 1. fit + freeze (training wells only)
    cols = sorted(set(VALUE_COLS) | set(HEALTH_CODE_COLS))
    real, synth = load_cache(cache, cols, synth_event_cap=GBDTSimTrained().per_group)
    print(f"Loaded real {len(real):,} / synthetic {len(synth):,} rows ({time.perf_counter() - t0:.0f} s)", flush=True)
    detectors, fulls = [], {}
    for S in (PhysicsRules, GBDTSimTrained):
        sc, cal = fit_fold(S, P2, real, synth, fold, ALERT, FA_BUDGET)
        fulls[S.name] = (sc, cal)
        detectors.append(FrozenDetector(S.name, LABELS[S.name], sc, TailCalibration.from_calibration(cal), info={
            "on_percentile": cal.on_percentile, "off_percentile": cal.off_percentile,
            "train_fa_per_day": round(cal.train_fa_per_day, 3), "train_time_in_alarm": round(cal.train_time_in_alarm, 4),
            "meets_budget": bool(cal.meets_budget)}))
        print(f"  {S.name}: on {cal.on_percentile:.6f} off {cal.off_percentile:.6f}  "
              f"train FA {cal.train_fa_per_day:.3f}/day  in alarm {100 * cal.train_time_in_alarm:.2f}%", flush=True)

    # ---- normal-operation clips, chosen from batch scores on the held-out wells
    test = real[real["well"].isin(fold.test_wells) & (real["event_class"] == 0)]
    del synth
    quiet = fa_pick = None
    for w in sorted(test["well"].astype(str).unique()):
        d = test[test["well"].astype(str) == w]
        runs = _runs(d, CFG.block_s)
        st = {}
        for name, (sc, cal) in fulls.items():
            cfg = HysteresisConfig(cal.on_percentile, cal.off_percentile, ALERT.on_delay_s, ALERT.off_delay_s, ALERT.period_s)
            st[name] = alert_active(HysteresisAlerter(cfg).run(cal.to_percentile(sc.score(d)), degraded=d["n_ok"].to_numpy() < 2,
                                                                segments=runs))
        lab0 = d["class"].to_numpy() == 0
        if fa_pick is None:
            on = st["physics_rules"] & lab0
            first = np.flatnonzero(on & ~np.r_[False, on[:-1]])
            if first.size:
                fa_pick = (w, d.index[first[0]])
        # quiet: 6 h of labeled normal, same run, no alert from either detector, >= 3 usable sensors
        n = int(6 * 3600 / CFG.block_s)
        good = lab0 & ~st["physics_rules"] & ~st["gbdt_sim_trained"] & (d["n_ok"].to_numpy() >= 3)
        if quiet is None and len(d) > n:
            ok = pd.Series(good.astype(int)).rolling(n).sum().to_numpy() == n
            same = pd.Series(runs).rolling(n).apply(lambda r: r[0] == r[-1], raw=True).to_numpy() == 1
            hit = np.flatnonzero(ok & same)
            if hit.size:
                quiet = (w, d.index[hit[0] - n + 1], d.index[hit[0]])
    del real, test

    # ---- 2. clips from the raw data
    metas = discover(a.dataset)
    outcomes4 = pd.read_csv("reports/phase4_outcomes.csv")
    outcomes4 = outcomes4[outcomes4["condition"] == "clean"]
    clip_dir = Path(a.clips)
    clip_dir.mkdir(parents=True, exist_ok=True)
    wells = {}

    def well_tl(w):
        if w not in wells:
            wells.clear()  # one well in memory at a time
            wells[w] = stitch_well([m for m in metas if m.well == w], SHARED_SENSORS)
        return wells[w]

    manifest = []
    for slug, iid, story in EPISODES:
        w = iid.split("/")[1].split("_")[0]
        if w not in fold.test_wells:
            sys.exit(f"{iid} is not in fold {fold.k}'s held-out wells; the EPISODES list assumes fold 1")
        c = int(iid.split("/")[0])
        tl = well_tl(w)
        inst = tl[tl["instance_id"].astype(str) == iid]
        lab = inst["class"].to_numpy()
        tr = inst.index[lab == TRANSIENT_OFFSET + c]
        ss = inst.index[lab == c]
        onset = tr[0] if len(tr) else ss[0]
        est = ss[0] if len(ss) else None
        # always include establishment (lead is measured to it); for slow events start the
        # replay later instead of cutting the end (earlier rows still stream, as warm-up)
        end = min((est if est is not None else inst.index[-1]) + pd.Timedelta(seconds=POST_S), inst.index[-1])
        start = max(onset - pd.Timedelta(seconds=PRE_ROLL_S), end - pd.Timedelta(seconds=MAX_REPLAY_S))
        clip, replay_start = cut(tl, start, end)
        exp = {}
        for det in LABELS:
            r = outcomes4[(outcomes4["instance_id"] == iid) & (outcomes4["detector"] == det)].iloc[0]
            exp[det] = {"detected": bool(r["detected"]), "warned": bool(r["warned_before_established"]),
                        "lead_min": None if not np.isfinite(r["lead_s"]) else round(r["lead_s"] / 60, 1)}
        manifest.append(dict(id=slug, file=f"{slug}.parquet", kind="event", title=f"{EVENT_NAMES[c]} ({w}, {onset:%Y-%m-%d})",
                             story=story, well=w, instance_id=iid, event_class=c, event_name=EVENT_NAMES[c],
                             replay_start=str(replay_start), onset=str(onset),
                             established=None if est is None else str(est), end=str(clip.index[-1]), phase4=exp))
        clip.to_parquet(clip_dir / f"{slug}.parquet", compression="zstd")

    if quiet:
        w, s, e = quiet
        clip, rs = cut(well_tl(w), s, e)
        clip.to_parquet(clip_dir / "normal_quiet.parquet", compression="zstd")
        manifest.append(dict(id="normal_quiet", file="normal_quiet.parquet", kind="normal",
                             title=f"Normal operation ({w}, {rs:%Y-%m-%d})", well=w, replay_start=str(rs), end=str(clip.index[-1]),
                             story="Six hours of normal operation: neither detector alarms. Use this one to break sensors."))
    if fa_pick:
        w, t_alarm = fa_pick
        tl = well_tl(w)
        s = max(t_alarm - pd.Timedelta(hours=2), tl.index[0])
        clip, rs = cut(tl, s, t_alarm + pd.Timedelta(hours=2))
        clip.to_parquet(clip_dir / "false_alarm.parquet", compression="zstd")
        manifest.append(dict(id="false_alarm", file="false_alarm.parquet", kind="false_alarm",
                             title=f"A false alarm ({w}, {t_alarm:%Y-%m-%d})", well=w, replay_start=str(rs), alarm_at=str(t_alarm),
                             end=str(clip.index[-1]),
                             story="Normal operation where the physics rules DO alarm: one of the ~1 per well-week the "
                                   "threshold allows. Shown on purpose."))

    titles = [m["title"] for m in manifest]
    if len(set(titles)) != len(titles):
        sys.exit(f"episode titles must be unique (the app picks by title): {titles}")

    # ---- 3. stream every clip, compare with Phase 4
    print("\nStreaming check (engine on the clip vs Phase 4 batch result):", flush=True)
    mismatches = 0
    for m in manifest:
        clip = pd.read_parquet(clip_dir / m["file"])
        rows = stream_clip(clip, detectors)
        replay = rows[rows.index >= pd.Timestamp(m["replay_start"])]
        m["streamed"] = {}
        for d in detectors:
            if m["kind"] == "event":
                got = outcome(rows, d.name, m["instance_id"], m["event_class"])
                same = got == m["phase4"][d.name]
                mismatches += not same
                m["streamed"][d.name] = got
                print(f"  {m['id']:<26} {d.name:<18} stream: {_fmt_lead(got):<38} phase 4: {_fmt_lead(m['phase4'][d.name]):<38}"
                      f" {'OK' if same else 'DIFFERENT'}", flush=True)
            else:
                act = alert_active(replay[f'{d.name}__status'].to_numpy())
                starts = np.flatnonzero(np.diff(np.r_[0, act.astype(int)]) == 1)
                m["streamed"][d.name] = {"alert_episodes": int(starts.size)}
                print(f"  {m['id']:<26} {d.name:<18} alert episodes during replay: {starts.size}", flush=True)
                if m["kind"] == "false_alarm" and d.name == "physics_rules" and starts.size:
                    # describe the alarm from what the stream did (the batch pick above only
                    # looked at labeled-normal rows, so it marks where the labels start)
                    t_on = replay.index[starts[0]]
                    ends = np.flatnonzero(np.diff(np.r_[act.astype(int), 0]) == -1)
                    t_off = replay.index[ends[0]]
                    held = (t_off - t_on).total_seconds() / 60
                    cleared = ends[0] < len(act) - 1
                    since = (t_on - clip.index[0]).total_seconds() / 60
                    lab0 = clip.index[clip["class"].to_numpy() == 0][0]
                    m["alarm_at"] = str(t_on)
                    m["labels_start"] = str(lab0)
                    m["story"] = (f"Normal operation where the physics rules DO alarm (the threshold allows ~1 per "
                                  f"well-week). Raised {since:.0f} min after data resumes, before the 6 h baselines "
                                  f"exist: the oscillation rule's scale falls back to its 0.5 bar floor, so ordinary "
                                  f"choke-pressure noise reads larger than it will once a baseline forms. Then held "
                                  f"{held:.0f} min{'' if cleared else ' (still active at clip end)'}: the score stays "
                                  f"above the clearing threshold, set one decade below the alarm threshold. "
                                  f"Shown on purpose; see the README's known issues.")
                    print(f"      raised {t_on} ({since:.0f} min after the data starts), held {held:.0f} min, "
                          f"{'cleared' if cleared else 'still active at clip end'}; normal labels start {lab0}")
        m["size_kb"] = round((clip_dir / m["file"]).stat().st_size / 1024)

    meta = {"fold": fold.k, "protocol": P2, "train_wells": sorted(fold.train_wells), "test_wells": sorted(fold.test_wells),
            "feature_key": CFG.key, "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
            "alert": {"on_delay_s": ALERT.on_delay_s, "off_delay_s": ALERT.off_delay_s, "period_s": ALERT.period_s},
            "fa_budget_per_day": FA_BUDGET}
    Path(a.models).mkdir(parents=True, exist_ok=True)
    save_detectors(Path(a.models) / "detectors.pkl", detectors, meta)
    # pinned to the versions that wrote the pickle, for hosting (Streamlit Community Cloud
    # reads requirements.txt next to the app's entry file)
    import plotly
    import pyarrow
    import scipy
    import streamlit
    Path("app/requirements.txt").write_text("\n".join([
        f"numpy=={np.__version__}", f"pandas=={pd.__version__}", f"scikit-learn=={sklearn.__version__}",
        f"scipy=={scipy.__version__}", f"pyarrow=={pyarrow.__version__}", f"streamlit=={streamlit.__version__}",
        f"plotly=={plotly.__version__}", ""]), encoding="utf-8")
    (clip_dir / "manifest.json").write_text(json.dumps({"meta": meta, "episodes": manifest}, indent=2), encoding="utf-8")
    size = (Path(a.models) / "detectors.pkl").stat().st_size / 1e6
    print(f"\nDetectors -> {Path(a.models) / 'detectors.pkl'} ({size:.1f} MB); "
          f"{len(manifest)} clips -> {clip_dir} ({sum(m['size_kb'] for m in manifest) / 1024:.1f} MB)")
    print(f"{'All event outcomes match Phase 4.' if not mismatches else f'{mismatches} outcome(s) differ from Phase 4 (see above).'} "
          f"Done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
