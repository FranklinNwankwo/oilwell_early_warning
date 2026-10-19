"""Causal window features for the 5 shared sensors.

Design decisions, each forced by a fact in reports/data_audit.md:

* ONLY the 5 shared sensors (P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP), the only
  continuous sensors present in simulated and hand-drawn data, so every protocol
  (sim-only to real+sim) sees the same feature space.
* NO absolute levels. Absolute pressure identifies the well (and the well identifies
  the label, see the missingness leak test). Every value feature is expressed relative
  to the signal's own lagged trailing baseline, in units of a robust scale.
* A scale FLOOR per signal kind. Simulated steady states are noise-free, so a raw
  robust scale goes to ~0 in simulation but not in real data; without a floor the same
  physical change would score orders of magnitude higher in sim than in real wells.
* Unusable samples (MISSING / FROZEN / INVALID from quality.py) are masked to NaN
  before any statistic, and per-sensor health is exported as features, because real
  P-PDG is usable in only 18% of normal operation.
* Staleness ("age": seconds since the value last changed) is a feature, because many
  healthy real P-TPT / T-TPT series are historian-held for ~601 s between updates, so
  short-window slopes read zero and then jump.
* 10 s blocks, aligned to absolute epoch time (not to the segment start), so a live
  service that keeps only a trailing buffer produces the same blocks, and a block is
  emitted only once it is complete. Everything is causal: a row uses samples up to its
  own timestamp only. tests/test_features.py proves prefix invariance and that an
  8-hour trailing buffer reproduces full-history features.

Output: one row per 10 s block, timestamped at the block's LAST sample.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .quality import FrozenRule, Health, health_matrix
from .schema import SAMPLE_PERIOD_S, SHARED_SENSORS

# Derived pressure-difference signals: (name, minuend, subtrahend). Both must be OK.
PAIR_SIGNALS: tuple[tuple[str, str, str], ...] = (
    ("dP_PDG_TPT", "P-PDG", "P-TPT"),  # across the production column (DHSV, hydrate, productivity)
    ("dP_TPT_CKP", "P-TPT", "P-MON-CKP"),  # along the production line to the choke (line hydrate, restriction)
)
SIGNALS: tuple[str, ...] = (*SHARED_SENSORS, *(p[0] for p in PAIR_SIGNALS))
META_COLUMNS = ("class", "state", "event_class", "instance_id", "segment")


def _kind(signal: str) -> str:
    return "temperature" if signal.startswith("T-") else "pressure"


@dataclass(frozen=True)
class FeatureConfig:
    block_s: int = 10
    windows_s: tuple[int, ...] = (60, 300, 1800)
    baseline_s: int = 6 * 3600  # robust level/scale over the trailing 6 h ...
    baseline_lag_s: int = 1800  # ... ending 30 min ago, so a fast change is not absorbed into its own baseline
    baseline_min_s: int = 1800  # need >= 30 min of usable history before a baseline exists
    min_valid_frac: float = 0.5  # a window statistic needs >= 50% usable blocks
    pressure_floor: float = 5.0e4  # Pa (0.5 bar): minimum robust scale for pressures and pressure differences
    temperature_floor: float = 0.5  # degC
    clip: float = 50.0  # |feature| cap, in robust-scale units
    max_age_s: int = 3600  # staleness cap; == FrozenRule().min_run_s, beyond it the sensor is FROZEN
    # Round value features to 3 decimals (robust-scale units: 0.001 = 50 Pa or 0.0005 degC,
    # far below sensor noise). Shrinks the cache 3.5x (zstd cannot compress float noise),
    # and is applied here, not at write time, so batch and live features stay identical.
    decimals: int = 3
    version: str = "v1"

    def __post_init__(self) -> None:
        if any(w % self.block_s for w in self.windows_s):
            raise ValueError("every window must be a multiple of block_s")
        if self.baseline_s % self.block_s or self.baseline_lag_s % self.block_s:
            raise ValueError("baseline_s and baseline_lag_s must be multiples of block_s")

    def floor(self, signal: str) -> float:
        return self.temperature_floor if _kind(signal) == "temperature" else self.pressure_floor

    @property
    def key(self) -> str:
        """Cache key: any change to any parameter invalidates cached features."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return f"{self.version}-{hashlib.sha256(blob).hexdigest()[:10]}"

    @property
    def history_s(self) -> int:
        """Trailing history a live service must keep to reproduce batch features exactly."""
        need = max(self.baseline_s + self.baseline_lag_s, 2 * max(self.windows_s))
        return need + self.max_age_s + 2 * self.block_s

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for s in SIGNALS:
            for w in self.windows_s:
                names += [f"{s}__lvl_{w}", f"{s}__dlt_{w}", f"{s}__std_{w}"]
        for s in SHARED_SENSORS:
            names += [f"{s}__ok_300", f"{s}__age", f"{s}__health"]
        names.append("n_ok")
        return names


# --------------------------------------------------------------------------- helpers
def _age_seconds(x: np.ndarray, cap: int) -> np.ndarray:
    """Seconds since the value last changed (causal), NaN where x is NaN, capped."""
    n = x.size
    if n == 0:
        return np.zeros(0)
    with np.errstate(invalid="ignore"):
        changed = np.ones(n, dtype=bool)
        changed[1:] = ~(x[1:] == x[:-1])  # NaN != NaN -> counts as a change (restart)
    last = np.maximum.accumulate(np.where(changed, np.arange(n), 0))
    age = (np.arange(n) - last).astype(np.float64) * SAMPLE_PERIOD_S
    age = np.minimum(age, cap)
    age[np.isnan(x)] = np.nan
    return age


def _block_layout(t_epoch_s: np.ndarray, block_s: int) -> tuple[np.ndarray, np.ndarray]:
    """For a contiguous 1 Hz segment: index of the first sample of each COMPLETE
    epoch-aligned block, and those blocks' ids. Partial blocks at either end are dropped."""
    bid = t_epoch_s // block_s
    starts = np.flatnonzero(np.r_[True, bid[1:] != bid[:-1]])
    lens = np.diff(np.r_[starts, t_epoch_s.size])
    full = lens == block_s
    return starts[full], bid[starts[full]]


def _segment_features(seg: pd.DataFrame, ok: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame | None:
    """Features for ONE contiguous 1 Hz segment. `ok` = usable mask per shared sensor."""
    B = cfg.block_s
    # Explicit unit conversion: pandas 3 may hold timestamps as us or ns; never assume.
    t = pd.DatetimeIndex(seg.index).as_unit("s").asi8.astype(np.int64)
    starts, bids = _block_layout(t, B)
    if starts.size == 0:
        return None
    nb = starts.size
    take = starts[:, None] + np.arange(B)[None, :]  # (nb, B) sample indices of each complete block
    # Complete blocks are consecutive in time only where block ids are consecutive;
    # inside one contiguous segment that is always true except at dropped partial ends.
    if np.any(np.diff(bids) != 1):
        raise AssertionError("non-consecutive blocks inside a contiguous segment")
    end_idx = take[:, -1]

    vals: dict[str, np.ndarray] = {}
    for s in SHARED_SENSORS:
        x = seg[s].to_numpy(dtype=np.float64)
        vals[s] = np.where(ok[s].to_numpy(), x, np.nan)
    for name, a, b in PAIR_SIGNALS:
        vals[name] = vals[a] - vals[b]  # NaN unless both usable

    out: dict[str, np.ndarray] = {}
    kb = cfg.baseline_s // B
    kmin = cfg.baseline_min_s // B
    klag = cfg.baseline_lag_s // B
    for s in SIGNALS:
        xb = vals[s][take]  # (nb, B)
        cnt = np.isfinite(xb).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            bm = np.where(cnt >= B / 2, np.nansum(xb, axis=1) / np.maximum(cnt, 1), np.nan)
        ser = pd.Series(bm)
        roll = ser.rolling(kb, min_periods=kmin)
        med = roll.median().shift(klag).to_numpy()
        q75 = roll.quantile(0.75).shift(klag).to_numpy()
        q25 = roll.quantile(0.25).shift(klag).to_numpy()
        scale = np.fmax((q75 - q25) / 1.349, cfg.floor(s))  # fmax: NaN quantiles -> floor, med stays NaN
        for w in cfg.windows_s:
            k = w // B
            mp = max(1, int(np.ceil(k * cfg.min_valid_frac)))
            r = ser.rolling(k, min_periods=mp)
            m = r.mean().to_numpy()
            sd = r.std(ddof=0).to_numpy() if k > 1 else np.full(nb, np.nan)
            m_prev = np.r_[np.full(min(k, nb), np.nan), m[:-k]] if nb > k else np.full(nb, np.nan)
            with np.errstate(invalid="ignore"):
                out[f"{s}__lvl_{w}"] = np.clip((m - med) / scale, -cfg.clip, cfg.clip)
                out[f"{s}__dlt_{w}"] = np.clip((m - m_prev) / scale, -cfg.clip, cfg.clip)
                out[f"{s}__std_{w}"] = np.clip(sd / scale, 0.0, cfg.clip)

    n_ok = np.zeros(nb, dtype=np.float64)
    for s in SHARED_SENSORS:
        okb = ok[s].to_numpy()[take].mean(axis=1)
        out[f"{s}__ok_300"] = pd.Series(okb).rolling(300 // B, min_periods=1).mean().to_numpy()
        out[f"{s}__age"] = _age_seconds(vals[s], cfg.max_age_s)[end_idx]
        out[f"{s}__health"] = seg[f"_h_{s}"].to_numpy()[end_idx].astype(np.float64)
        n_ok += ok[s].to_numpy()[end_idx]
    out["n_ok"] = n_ok

    feats = pd.DataFrame(out, index=seg.index[end_idx])[cfg.feature_names()].round(cfg.decimals).astype(np.float32)
    for c in META_COLUMNS:
        if c in seg:
            feats[c] = seg[c].to_numpy()[end_idx]
    return feats


def build_features(
    timeline: pd.DataFrame,
    cfg: FeatureConfig = FeatureConfig(),
    *,
    detect_frozen: bool,
    frozen_rule: FrozenRule = FrozenRule(),
) -> pd.DataFrame:
    """Causal features for a 1 Hz timeline (one instance, or a stitched well timeline).

    `timeline` needs the 5 shared sensors, a DatetimeIndex, and optionally the META
    columns. A 'segment' column (from data.stitch_well) splits it at data gaps; without
    one, gaps are detected here. State never crosses a segment boundary.
    detect_frozen: True for real data, False for simulated / hand-drawn (see quality.py).
    """
    missing = [s for s in SHARED_SENSORS if s not in timeline]
    if missing:
        raise KeyError(f"timeline lacks shared sensors {missing}")
    tl = timeline
    if "segment" not in tl:
        gaps = tl.index.to_series().diff().dt.total_seconds().fillna(SAMPLE_PERIOD_S).to_numpy()
        tl = tl.assign(segment=np.cumsum(gaps != SAMPLE_PERIOD_S).astype(np.int32))
    health = health_matrix(tl, list(SHARED_SENSORS), frozen_rule, segment_col="segment", detect_frozen=detect_frozen)
    ok = health == Health.OK
    tl = tl.assign(**{f"_h_{s}": health[s].to_numpy() for s in SHARED_SENSORS})
    parts = []
    seg_ids = tl["segment"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, seg_ids[1:] != seg_ids[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        f = _segment_features(tl.iloc[a:b], ok.iloc[a:b], cfg)
        if f is not None:
            parts.append(f)
    if not parts:
        empty = pd.DataFrame(
            columns=[*cfg.feature_names(), *[c for c in META_COLUMNS if c in tl]],
            index=pd.DatetimeIndex([], name="timestamp", dtype="datetime64[ns]"),
        )
        return empty.astype({c: np.float32 for c in cfg.feature_names()})
    return pd.concat(parts)
