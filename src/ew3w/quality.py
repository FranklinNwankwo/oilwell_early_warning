"""Causal sensor-health detection: MISSING and FROZEN, per sensor, per sample.

Why this exists: in real 3W data, missing and frozen variables are the norm, not the
exception (3W 2.0.0 paper, Table 7: 65.9% of real instance-variables missing, 9.8%
frozen). The audit shows P-PDG is missing or frozen in 92% of real normal instances
while it is healthy in every simulated one. Valve-state ESTADO-* columns are
legitimately constant for hours, so "frozen" detection applies only to continuous
sensors.

Everything here is causal (uses only past samples), so the exact same code runs in
offline evaluation and in the streaming service: there is no train/serve skew.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from .schema import VALVE_STATE_SENSORS


class Health(IntEnum):
    OK = 0
    MISSING = 1  # NaN: not recorded
    FROZEN = 2  # value stuck (continuous sensors only)
    INVALID = 3  # outside physical range (historian fill value); stored as +inf by data.load_instance


@dataclass(frozen=True)
class FrozenRule:
    """A continuous sensor is FROZEN once its value has not changed by more than
    `abs_tol` for `min_run_s` consecutive samples.

    Why 3600 s and not less: many healthy real P-TPT / T-TPT series are stored by the
    historian every ~10 min and held in between, so they show thousands of constant
    runs of exactly ~601 s (audit of all real instances). A 600 s rule flags those
    healthy low-rate sensors as frozen. Genuine stalls look different: P-PDG's
    constant runs have a median of ~21,400 s and 85% span the whole instance.
    1 hour sits well above the hold interval and still catches every long stall.
    """

    min_run_s: int = 3600
    abs_tol: float = 0.0  # exact repeat == frozen; raise if the historian adds jitter


def frozen_mask(x: np.ndarray, rule: FrozenRule = FrozenRule()) -> np.ndarray:
    """Boolean mask, True where the sensor is (causally) known to be frozen.

    A run of unchanged values is only flagged from the sample at which its length
    reaches `min_run_s`, never retroactively: that is what a live system can know.
    NaN breaks a run.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    same = np.zeros(n, dtype=bool)
    with np.errstate(invalid="ignore"):  # inf - inf (INVALID samples) -> NaN -> not "same"
        same[1:] = np.abs(np.diff(x)) <= rule.abs_tol  # NaN comparisons -> False
    # run length of consecutive "same" samples ending at i (count of repeats + 1)
    run_start = ~same
    run_id = np.cumsum(run_start)
    first_idx = np.flatnonzero(run_start)
    run_len = np.arange(n) - first_idx[run_id - 1] + 1
    out = run_len >= rule.min_run_s
    out &= ~np.isnan(x)
    return out


def health_matrix(
    df: pd.DataFrame,
    sensors: list[str],
    rule: FrozenRule = FrozenRule(),
    segment_col: str | None = "segment",
    detect_frozen: bool = True,
) -> pd.DataFrame:
    """Per-sample Health code (int8) for each sensor. Resets at timeline gaps.

    Pass detect_frozen=False for simulated and hand-drawn instances: the 3W 2.0.0
    paper states they contain no frozen variables, but simulators output exactly
    constant values at steady state, so the rule would flag healthy samples
    (measured: 1-7% of synthetic samples even at 3600 s). Frozen sensors in training
    are produced deliberately by sensor-dropout augmentation instead.
    """
    out = pd.DataFrame(index=df.index)
    segs = df[segment_col].to_numpy() if segment_col and segment_col in df else np.zeros(len(df), dtype=int)
    boundaries = np.flatnonzero(np.diff(segs)) + 1
    for s in sensors:
        x = df[s].to_numpy(dtype=np.float64)
        h = np.where(np.isnan(x), Health.MISSING, Health.OK).astype(np.int8)
        if detect_frozen and s not in VALVE_STATE_SENSORS:
            fz = np.concatenate([frozen_mask(part, rule) for part in np.split(x, boundaries)]) if len(x) else np.zeros(0, bool)
            h[fz] = Health.FROZEN
        h[np.isinf(x)] = Health.INVALID
        out[s] = h
    return out


def usable_mask(health: pd.DataFrame) -> pd.DataFrame:
    """True where the sensor value may be fed to a model."""
    return health == Health.OK


def instance_health_summary(health: pd.DataFrame) -> pd.Series:
    """Dominant health state per sensor over the instance (for reporting/stratifying)."""
    return health.apply(lambda c: Health(int(np.bincount(c, minlength=len(Health)).argmax())).name)
