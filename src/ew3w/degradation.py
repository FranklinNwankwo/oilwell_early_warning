"""Phase 4: inject sensor failures into held-out wells and re-score with FROZEN detectors.

Two failure modes, both applied only to test data (detectors and thresholds come from
clean training wells via evaluation.fit_fold and are never refit):

  MISSING from the start   The sensor never reports. Every feature that depends on it
                           (its own windows, and pair signals that use it) is NaN, its
                           health is MISSING, its usable fraction 0, its staleness NaN.
                           Because the features of each signal are computed independently,
                           this is an EXACT transform of cached features; the test suite
                           checks it against a full rebuild from raw data.
  FROZEN before onset      The sensor keeps reporting its last value from a random moment
                           (within the hour before each event's onset, or anywhere inside
                           a normal instance) to the end of that instance. This changes
                           features over time (the level drifts away from the baseline,
                           change and variability collapse to zero, after 1 h the health
                           check flags FROZEN), so it is injected into the RAW 1 Hz data
                           and the features are rebuilt with the production pipeline.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .features import PAIR_SIGNALS, SIGNALS, FeatureConfig
from .quality import Health
from .schema import SHARED_SENSORS, TRANSIENT_OFFSET

_CFG = FeatureConfig()


def affected_columns(sensor: str, cfg: FeatureConfig = _CFG) -> list[str]:
    """Value-feature columns that depend on `sensor` (its own + pair signals using it)."""
    sigs = {sensor} | {n for n, a, b in PAIR_SIGNALS if sensor in (a, b)}
    return [c for c in cfg.feature_names() if c.split("__")[0] in sigs and c.split("__")[1][:3] in ("lvl", "dlt", "std")]


def drop_sensors(df: pd.DataFrame, sensors: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """Copy of feature rows with `sensors` MISSING from the start (see module docstring).
    Columns that are not present in `df` are skipped; n_ok is recomputed from the health
    codes, so all 5 '<sensor>__health' columns must be present."""
    if not sensors:
        return df
    unknown = set(sensors) - set(SHARED_SENSORS)
    if unknown:
        raise KeyError(f"not shared sensors: {unknown}")
    health_cols = [f"{s}__health" for s in SHARED_SENSORS]
    missing = [c for c in health_cols if c not in df]
    if missing:
        raise KeyError(f"drop_sensors needs all health columns; missing {missing}")
    out = df.copy()
    for s in sensors:
        cols = [c for c in affected_columns(s) if c in out]
        out[cols] = np.nan
        out[f"{s}__health"] = np.float32(Health.MISSING)
        if f"{s}__ok_300" in out:
            out[f"{s}__ok_300"] = np.float32(0.0)
        if f"{s}__age" in out:
            out[f"{s}__age"] = np.nan
    out["n_ok"] = (out[health_cols].to_numpy() == Health.OK).sum(axis=1).astype(np.float32)
    return out


def missing_conditions(max_k: int = 4) -> list[tuple[str, ...]]:
    """Every subset of the 5 shared sensors of size 1..max_k (5 + 10 + 10 + 5 = 30 for max_k=4).
    k = 5 is omitted: with nothing observable the alerter is DEGRADED by construction."""
    return [c for k in range(1, max_k + 1) for c in combinations(SHARED_SENSORS, k)]


def freeze_raw(timeline: pd.DataFrame, sensor: str, seed: int = 0, max_before_onset_s: int = 3600) -> pd.DataFrame:
    """Copy of a stitched 1 Hz well timeline where `sensor` freezes once per instance.

    Event instance: freeze time uniform in [onset - max_before_onset_s, onset] (clipped to
    the instance's first labeled sample); classes 3/4 have no observable onset, so they
    freeze at their first labeled sample. Normal instance: uniform over its labeled span.
    The held value is the sensor's own value at the freeze time; if that is not finite
    (already missing / invalid) the instance is left untouched. Frozen to instance end."""
    tl = timeline.copy()
    x = tl[sensor].to_numpy(dtype=np.float64, copy=True)
    cls = tl["class"].to_numpy(dtype=np.float64)
    ev = tl["event_class"].to_numpy()
    inst = tl["instance_id"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    t = (pd.DatetimeIndex(tl.index).as_unit("s").asi8).astype(np.int64)
    starts = np.flatnonzero(np.r_[True, inst[1:] != inst[:-1]])
    ends = np.r_[starts[1:], len(inst)]
    for a, b in zip(starts, ends):
        lab = np.flatnonzero(~np.isnan(cls[a:b])) + a
        if lab.size == 0:
            continue
        c = int(ev[a])
        if c == 0:
            i_f = int(rng.integers(lab[0], lab[-1] + 1))
        else:
            on = np.flatnonzero(cls[a:b] == TRANSIENT_OFFSET + c)
            if on.size == 0:
                i_f = int(lab[0])
            else:
                i_on = int(on[0]) + a
                lo_t = max(t[lab[0]], t[i_on] - max_before_onset_s)
                lo = int(np.searchsorted(t[a:b], lo_t)) + a
                i_f = int(rng.integers(lo, i_on + 1))
        if np.isfinite(x[i_f]):
            x[i_f:b] = x[i_f]
    tl[sensor] = x.astype(np.float32)
    return tl


__all__ = ["affected_columns", "drop_sensors", "missing_conditions", "freeze_raw", "SIGNALS"]
