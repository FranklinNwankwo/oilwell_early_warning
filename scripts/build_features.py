"""Build and cache causal features for every 3W instance.

Usage (PowerShell, from the repo root, with EW3W_DATASET set):
    python scripts/build_features.py                 # everything (real + simulated + drawn)
    python scripts/build_features.py --only real     # one source
    python scripts/build_features.py --force         # rebuild even if cached

Layout (one folder per feature-config key, so a config change never mixes with old files):
    data/features/<key>/config.json
    data/features/<key>/real/WELL-00001.parquet       one stitched timeline per real well
    data/features/<key>/simulated/class_2.parquet     all simulated instances of a class
    data/features/<key>/drawn/class_7.parquet
    data/features/<key>/manifest.json                 rows, sizes, build times

Real wells are stitched (overlapping lead-ins removed, gaps split into segments) so a
well's normal and event instances share one continuous causal history, exactly as a
live system would see them. Simulated / drawn instances are independent recordings.
Files are written atomically (temp file, then rename), so an interrupted run never
leaves a half-written file that a later run would mistake for a finished one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ew3w.data import discover, load_instance, stitch_well  # noqa: E402
from ew3w.features import FeatureConfig, build_features  # noqa: E402
from ew3w.schema import SHARED_SENSORS  # noqa: E402

SENSORS = list(SHARED_SENSORS)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, compression="zstd")
    os.replace(tmp, path)


def build_real(metas, out: Path, cfg: FeatureConfig, force: bool, log: dict) -> None:
    by_well = defaultdict(list)
    for m in metas:
        if m.source == "real":
            by_well[m.well].append(m)
    for i, (well, ms) in enumerate(sorted(by_well.items()), 1):
        path = out / "real" / f"{well}.parquet"
        if path.exists() and not force:
            print(f"  [{i}/{len(by_well)}] {well}: cached", flush=True)
            continue
        t0 = time.perf_counter()
        tl = stitch_well(ms, SENSORS)
        f = build_features(tl, cfg, detect_frozen=True)
        f["well"] = well
        f["source"] = "real"
        _write(f, path)
        dt = time.perf_counter() - t0
        log[str(path.relative_to(out))] = dict(rows=len(f), instances=len(ms), seconds=round(dt, 1))
        print(f"  [{i}/{len(by_well)}] {well}: {len(ms)} instances, {len(tl):,} s -> {len(f):,} rows ({dt:.0f} s)", flush=True)


def build_synthetic(metas, source: str, out: Path, cfg: FeatureConfig, force: bool, log: dict) -> None:
    by_cls = defaultdict(list)
    for m in metas:
        if m.source == source:
            by_cls[m.event_class].append(m)
    for cls, ms in sorted(by_cls.items()):
        path = out / source / f"class_{cls}.parquet"
        if path.exists() and not force:
            print(f"  {source} class {cls}: cached", flush=True)
            continue
        t0 = time.perf_counter()
        parts = []
        for j, m in enumerate(ms, 1):
            f = build_features(load_instance(m, SENSORS), cfg, detect_frozen=False)
            f["well"] = m.stem  # each synthetic instance is its own group for CV purposes
            f["source"] = source
            parts.append(f)
            if j % 100 == 0:
                print(f"    {source} class {cls}: {j}/{len(ms)}", flush=True)
        f = pd.concat(parts)
        _write(f, path)
        dt = time.perf_counter() - t0
        log[str(path.relative_to(out))] = dict(rows=len(f), instances=len(ms), seconds=round(dt, 1))
        print(f"  {source} class {cls}: {len(ms)} instances -> {len(f):,} rows ({dt:.0f} s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.environ.get("EW3W_DATASET"),
                    help="3W repo root or its 'dataset' folder (default: EW3W_DATASET)")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--only", choices=["real", "simulated", "drawn"], action="append",
                    help="build only these sources (repeatable); default all")
    ap.add_argument("--force", action="store_true", help="rebuild files that already exist")
    a = ap.parse_args()
    if not a.dataset:
        ap.error("no dataset path: pass --dataset or set the EW3W_DATASET environment variable")

    cfg = FeatureConfig()
    out = Path(a.out) / cfg.key
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"key": cfg.key, **cfg.__dict__, "features": cfg.feature_names()}, indent=2))
    metas = discover(a.dataset)
    sources = a.only or ["real", "simulated", "drawn"]
    print(f"Feature config {cfg.key}: {len(cfg.feature_names())} features, {cfg.block_s} s rows -> {out}")

    man_path = out / "manifest.json"
    log = json.loads(man_path.read_text()) if man_path.exists() else {}
    t0 = time.perf_counter()
    try:
        if "real" in sources:
            print("Real wells (stitched timelines):")
            build_real(metas, out, cfg, a.force, log)
        for src in ("simulated", "drawn"):
            if src in sources:
                print(f"{src.capitalize()} instances:")
                build_synthetic(metas, src, out, cfg, a.force, log)
    finally:
        man_path.write_text(json.dumps(log, indent=2))
    size = sum(p.stat().st_size for p in out.rglob("*.parquet")) / 1e6
    print(f"Done in {time.perf_counter() - t0:.0f} s. Cache: {size:,.0f} MB in {out}")


if __name__ == "__main__":
    main()
