"""Phase 5: cost of the streaming service per 1 Hz reading, on a real replay clip.

Usage: python scripts/bench_stream.py        (needs models/ and demo_data/ from export_demo.py)

Three configurations, all through ew3w.stream: features only, + physics rules, + both
detectors. 'push' = one reading at a time (live); 'push_many' = the same readings with
scoring batched (warm-up). Reports microseconds per reading and the implied headroom
over real time (1 reading per second per well).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ew3w.features import FeatureConfig  # noqa: E402
from ew3w.schema import SHARED_SENSORS  # noqa: E402
from ew3w.stream import EarlyWarningEngine, FeatureStream, load_detectors  # noqa: E402


def main() -> None:
    dets, _ = load_detectors(ROOT / "models" / "detectors.pkl")
    clip = pd.read_parquet(ROOT / "demo_data" / "bsw_rules_only.parquet")
    t = pd.DatetimeIndex(clip.index).as_unit("ns").asi8
    X = clip[list(SHARED_SENSORS)].to_numpy(np.float64)
    n = len(t)
    print(f"clip: {n:,} readings ({n / 3600:.1f} h of 1 Hz data)")

    def run(label, fn):
        t0 = time.perf_counter()
        fn()
        us = 1e6 * (time.perf_counter() - t0) / n
        print(f"  {label:<44} {us:8.0f} us/reading   {1e6 / us:8,.0f}x real time")

    def feats():
        fs = FeatureStream(FeatureConfig(), detect_frozen=True)
        for ti, x in zip(t, X):
            fs.push(ti, x)

    def live(ds):
        def f():
            e = EarlyWarningEngine(ds)
            for ti, x in zip(t, X):
                e.push(ti, x)
        return f

    def batched(ds):
        return lambda: EarlyWarningEngine(ds).push_many(t, X)

    run("features only (push)", feats)
    run("features + physics rules (push)", live(dets[:1]))
    run("features + both detectors (push)", live(dets))
    run("features + both detectors (push_many)", batched(dets))


if __name__ == "__main__":
    main()
