"""Phase 5: the streaming service. One 1 Hz reading in, at most one scored row out.

This is the deployed form of the evaluated system, not a look-alike:

  FeatureStream   incremental version of features.build_features. Keeps bounded state
                  (one trailing ring of 10 s block means, ~6.5 h, plus per-sensor run
                  counters), so cost per reading is constant no matter how long the well
                  has been running. tests/test_stream.py checks it against the batch
                  pipeline row by row, including gaps, NaN, +inf and frozen sensors.
  TailCalibration compact, EXACT stand-in for evaluation.Calibration. Alert decisions
                  only compare percentiles against on/off thresholds >= 0.9, so only the
                  top of the training-normal score distribution has to be kept exactly;
                  below it, percentiles are interpolated for display only.
  EarlyWarningEngine  FeatureStream -> every detector's score -> percentile -> its own
                  hysteresis alerter (alerting.HysteresisAlerter.step, the same state
                  machine the evaluation ran). DEGRADED for the alerter is exactly the
                  evaluated rule (fewer than 2 usable sensors). On top of it, the
                  DISPLAYED status also reads DEGRADED whenever a detector has no score:
                  the Phase 4 fix for silent blindness. It changes what the operator
                  sees, never when an alert fires.
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .alerting import HysteresisAlerter, HysteresisConfig, Status
from .evaluation import Calibration
from .features import PAIR_SIGNALS, SIGNALS, FeatureConfig
from .quality import FrozenRule, Health
from .schema import SHARED_SENSORS

_PAIR_IDX = [(SHARED_SENSORS.index(a), SHARED_SENSORS.index(b)) for _, a, b in PAIR_SIGNALS]


class FeatureStream:
    """Incremental build_features for ONE well. push() one reading per second.

    Segment rule (same as batch): any step other than exactly +1 s starts a new segment
    and all state is reset. Readings must be time-ordered.
    """

    def __init__(self, cfg: FeatureConfig = FeatureConfig(), *, detect_frozen: bool,
                 frozen_rule: FrozenRule = FrozenRule()):
        self.cfg, self.detect_frozen, self.rule = cfg, detect_frozen, frozen_rule
        B = cfg.block_s
        self._B = B
        self._kb, self._kmin, self._klag = cfg.baseline_s // B, cfg.baseline_min_s // B, cfg.baseline_lag_s // B
        self._ks = [w // B for w in cfg.windows_s]
        self._mps = [max(1, int(np.ceil(k * cfg.min_valid_frac))) for k in self._ks]
        self._L = max(self._kb + self._klag, 2 * max(self._ks))  # blocks of history ever read
        self._floor = np.array([cfg.floor(s) for s in SIGNALS])
        self.names = cfg.feature_names()
        self._prev_t: int | None = None
        self.segment = -1
        self.health = np.full(len(SHARED_SENSORS), Health.MISSING, dtype=np.int8)
        self._reset()

    # ------------------------------------------------------------------ state
    def _reset(self) -> None:
        n5, n7, B = len(SHARED_SENSORS), len(SIGNALS), self._B
        self.segment += 1
        self._i = -1  # sample index inside the segment
        self._raw_prev = np.full(n5, np.nan)
        self._run = np.zeros(n5, dtype=np.int64)
        self._v_prev = np.full(n5, np.nan)
        self._last_change = np.zeros(n5, dtype=np.int64)
        self._age = np.full(n5, np.nan)
        self._bid: int | None = None
        self._n_in = 0
        self._buf = np.full((n7, B), np.nan)  # (signal, sample) like the batch (block, sample) rows
        self._okbuf = np.zeros((n5, B), dtype=bool)
        self._hist = np.full((2 * self._L, n7), np.nan)  # ring of block means
        self._j0 = 0  # absolute block index of _hist[0]
        self._nb = 0  # blocks emitted in this segment
        self._okhist = np.zeros((300 // B, n5))
        self._nok_blocks = 0

    def _window(self, a: int, b: int) -> np.ndarray:
        """Block means for absolute block indices [a, b) (a clipped at 0)."""
        a = max(a, 0)
        return self._hist[a - self._j0 : b - self._j0]

    # ------------------------------------------------------------------ input
    def push(self, t_ns: int, x: Sequence[float]) -> np.ndarray | None:
        """Add one reading (values of the 5 shared sensors, NaN = not recorded, +inf =
        invalid). Returns the feature vector (float32, order = cfg.feature_names()) when
        this reading completes an epoch-aligned 10 s block, else None."""
        t = int(t_ns) // 1_000_000_000
        if self._prev_t is not None and t <= self._prev_t:
            raise ValueError("readings must be strictly time-ordered")
        if self._prev_t is not None and t - self._prev_t != 1:
            self._reset()
        self._prev_t = t
        self._i += 1
        i = self._i
        x = np.asarray(x, dtype=np.float32).astype(np.float64)
        # --- health (quality.health_matrix, one sample at a time)
        h = np.where(np.isnan(x), Health.MISSING, Health.OK).astype(np.int8)
        if self.detect_frozen:
            with np.errstate(invalid="ignore"):
                same = (np.abs(x - self._raw_prev) <= self.rule.abs_tol) if i > 0 else np.zeros(x.size, bool)
            self._run = np.where(same, self._run + 1, 1)
            h[(self._run >= self.rule.min_run_s) & ~np.isnan(x)] = Health.FROZEN
        h[np.isinf(x)] = Health.INVALID
        self._raw_prev = x
        self.health = h
        ok = h == Health.OK
        v = np.where(ok, x, np.nan)
        # --- staleness (features._age_seconds)
        with np.errstate(invalid="ignore"):
            changed = np.ones(v.size, bool) if i == 0 else ~(v == self._v_prev)
        self._last_change = np.where(changed, i, self._last_change)
        self._v_prev = v
        age = np.minimum((i - self._last_change).astype(np.float64), self.cfg.max_age_s)
        age[np.isnan(v)] = np.nan
        # --- block assembly
        sig = np.concatenate([v, [v[a] - v[b] for a, b in _PAIR_IDX]])
        bid = t // self._B
        if bid != self._bid:
            self._bid, self._n_in = bid, 0
        self._buf[:, self._n_in] = sig
        self._okbuf[:, self._n_in] = ok
        self._n_in += 1
        if t % self._B != self._B - 1 or self._n_in != self._B:
            return None  # block not finished, or it began before this segment (partial: dropped)
        return self._emit(age, ok, h)

    def _emit(self, age: np.ndarray, ok: np.ndarray, h: np.ndarray) -> np.ndarray:
        cfg, B = self.cfg, self._B
        xb = self._buf.copy()  # (7, B) contiguous: same reduction as the batch (nb, B) rows
        cnt = np.isfinite(xb).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            bm = np.where(cnt >= B / 2, np.nansum(xb, axis=1) / np.maximum(cnt, 1), np.nan)
        # append to the ring
        j = self._nb
        if j - self._j0 == self._hist.shape[0]:
            self._hist[: self._L] = self._hist[-self._L :]
            self._hist[self._L :] = np.nan
            self._j0 += self._L
        self._hist[j - self._j0] = bm
        self._nb += 1

        n7 = len(SIGNALS)
        med = np.full(n7, np.nan)
        q25 = np.full(n7, np.nan)
        q75 = np.full(n7, np.nan)
        e = j - self._klag  # baseline window ends at block e (inclusive)
        if e >= 0:
            W = self._window(e - self._kb + 1, e + 1)
            good = np.isfinite(W).sum(axis=0) >= self._kmin
            if good.any():
                with np.errstate(invalid="ignore"):
                    q = np.nanquantile(W[:, good], [0.25, 0.5, 0.75], axis=0)
                q25[good], med[good], q75[good] = q
        scale = np.fmax((q75 - q25) / 1.349, self._floor)

        feats: dict[str, Any] = {}
        per_w = []
        for k, mp in zip(self._ks, self._mps):
            Wk = self._window(j - k + 1, j + 1)
            c = np.isfinite(Wk).sum(axis=0)
            with np.errstate(invalid="ignore"), _quiet():
                m = np.where(c >= mp, np.nanmean(Wk, axis=0), np.nan)
                sd = np.where(c >= mp, np.nanstd(Wk, axis=0), np.nan) if k > 1 else np.full(n7, np.nan)
            if j >= k:
                Wp = self._window(j - 2 * k + 1, j - k + 1)
                cp = np.isfinite(Wp).sum(axis=0)
                with np.errstate(invalid="ignore"), _quiet():
                    m_prev = np.where(cp >= mp, np.nanmean(Wp, axis=0), np.nan)
            else:
                m_prev = np.full(n7, np.nan)
            with np.errstate(invalid="ignore"):
                per_w.append((np.clip((m - med) / scale, -cfg.clip, cfg.clip),
                              np.clip((m - m_prev) / scale, -cfg.clip, cfg.clip),
                              np.clip(sd / scale, 0.0, cfg.clip)))
        for si, s in enumerate(SIGNALS):
            for w, (lvl, dlt, std) in zip(cfg.windows_s, per_w):
                feats[f"{s}__lvl_{w}"] = lvl[si]
                feats[f"{s}__dlt_{w}"] = dlt[si]
                feats[f"{s}__std_{w}"] = std[si]
        okb = self._okbuf.mean(axis=1)
        self._okhist = np.roll(self._okhist, -1, axis=0)
        self._okhist[-1] = okb
        self._nok_blocks = min(self._nok_blocks + 1, self._okhist.shape[0])
        ok300 = self._okhist[-self._nok_blocks :].mean(axis=0)
        for k, s in enumerate(SHARED_SENSORS):
            feats[f"{s}__ok_300"] = ok300[k]
            feats[f"{s}__age"] = age[k]
            feats[f"{s}__health"] = float(h[k])
        feats["n_ok"] = float(ok.sum())
        vec = np.array([feats[n] for n in self.names], dtype=np.float64)
        return np.round(vec, cfg.decimals).astype(np.float32)


class _quiet:
    """Silence 'Mean of empty slice' / 'Degrees of freedom' RuntimeWarnings (handled by counts)."""

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


# ---------------------------------------------------------------------- calibration
@dataclass
class TailCalibration:
    """Exact replacement for Calibration.to_percentile wherever the result is >= keep_from,
    i.e. everywhere an alert decision is made (on and off thresholds are >= 0.9).

    Kept: every training-normal score from rank floor(keep_from * n) up (ties included),
    plus a coarse quantile table for display below that. Below the kept tail the
    percentile is < keep_from <= off_percentile, so every on/off comparison is unchanged.
    tests/test_stream.py checks both claims against the full Calibration."""

    n: int
    offset: int  # number of training-normal scores below tail[0]
    tail: np.ndarray
    low_q: np.ndarray  # percentile grid below the tail (display only)
    low_v: np.ndarray  # score values at low_q
    on_percentile: float
    off_percentile: float
    keep_from: float = 0.85
    meets_budget: bool = True

    @classmethod
    def from_calibration(cls, cal: Calibration, keep_from: float = 0.85, n_display: int = 171) -> "TailCalibration":
        s = np.asarray(cal.sorted_normal, dtype=np.float64)
        n = s.size
        if not (cal.off_percentile > keep_from):
            raise ValueError("keep_from must be below the off-threshold, or alert decisions would change")
        cut = int(np.floor(keep_from * n))
        offset = int(np.searchsorted(s, s[min(cut, n - 1)], side="left"))  # keep ties of the cut value whole
        q = np.linspace(0.0, offset / n, n_display)
        v = s[np.minimum((q * n).astype(int), n - 1)]
        return cls(n, offset, s[offset:].copy(), q, v, cal.on_percentile, cal.off_percentile, keep_from,
                   bool(getattr(cal, "meets_budget", True)))

    def to_percentile(self, raw: np.ndarray | float) -> np.ndarray:
        raw = np.atleast_1d(np.asarray(raw, dtype=np.float64))
        lo = self.offset + np.searchsorted(self.tail, raw, side="left")
        hi = self.offset + np.searchsorted(self.tail, raw, side="right")
        p = (lo + hi) / (2.0 * self.n)
        below = raw < self.tail[0]
        if below.any():  # display only; always < offset / n <= keep_from
            approx = np.interp(raw[below], self.low_v, self.low_q)
            p[below] = np.minimum(approx, np.nextafter(self.offset / self.n, 0.0))
        p[~np.isfinite(raw)] = np.nan
        return p


# ---------------------------------------------------------------------- detectors
@dataclass
class FrozenDetector:
    """A fitted scorer + its frozen threshold, as saved by scripts/export_demo.py."""

    name: str
    label: str
    scorer: Any  # has .columns and .score(DataFrame)
    calibration: TailCalibration
    info: dict = field(default_factory=dict)

    def alert_config(self, base: HysteresisConfig) -> HysteresisConfig:
        c = self.calibration
        return HysteresisConfig(c.on_percentile, c.off_percentile, base.on_delay_s, base.off_delay_s, base.period_s)


def save_detectors(path: str | Path, detectors: list[FrozenDetector], meta: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump({"detectors": detectors, "meta": meta}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_detectors(path: str | Path) -> tuple[list[FrozenDetector], dict]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["detectors"], d["meta"]


# ---------------------------------------------------------------------- engine
MIN_OK = 2  # evaluated DEGRADED rule (evaluation.evaluate_wells): fewer than 2 usable sensors


@dataclass
class Tick:
    """One scored 10 s row."""

    t_ns: int
    segment: int
    features: np.ndarray
    n_ok: int
    health: np.ndarray
    raw: dict[str, float]
    pct: dict[str, float]
    status: dict[str, Status]  # the evaluated alerter state
    shown: dict[str, str]  # what the operator sees (adds DEGRADED when a detector is blind)
    reason: dict[str, str]


def shown_status(status: Status, raw: float, n_ok: int) -> tuple[str, str]:
    """Display rule (Phase 4 fix). Alerts are never changed; only a quiet NORMAL that is
    really 'cannot see' is relabelled."""
    if status == Status.DEGRADED:
        return "DEGRADED", f"only {n_ok} usable sensor{'s' if n_ok != 1 else ''} (needs {MIN_OK})"
    if status == Status.NORMAL and not math.isfinite(raw):
        return "DEGRADED", "this detector's sensors are unavailable: it cannot score"
    return status.name, ""


class EarlyWarningEngine:
    def __init__(self, detectors: list[FrozenDetector], cfg: FeatureConfig = FeatureConfig(), *,
                 detect_frozen: bool = True, alert: HysteresisConfig | None = None):
        self.cfg = cfg
        self.features = FeatureStream(cfg, detect_frozen=detect_frozen)
        self.detectors = detectors
        base = alert or HysteresisConfig(period_s=cfg.block_s)
        self.alerters = {d.name: HysteresisAlerter(d.alert_config(base)) for d in detectors}
        self._segment = self.features.segment
        self.names = self.features.names

    def _frame(self, rows: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(rows.astype(np.float32), columns=self.names)

    def _scores(self, rows: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        df = self._frame(rows)
        out = {}
        for d in self.detectors:
            raw = np.asarray(d.scorer.score(df), dtype=np.float64)
            out[d.name] = (raw, d.calibration.to_percentile(raw))
        return out

    def _tick(self, t_ns: int, seg: int, row: np.ndarray, health: np.ndarray, sc: dict, i: int) -> Tick:
        if seg != self._segment:  # data gap: alert state must not cross it (evaluation._runs)
            for a in self.alerters.values():
                a.reset()
            self._segment = seg
        n_ok = int(row[-1])
        raw, pct, status, shown, reason = {}, {}, {}, {}, {}
        for d in self.detectors:
            r, p = float(sc[d.name][0][i]), float(sc[d.name][1][i])
            st = self.alerters[d.name].step(p, degraded=n_ok < MIN_OK)
            raw[d.name], pct[d.name], status[d.name] = r, p, st
            shown[d.name], reason[d.name] = shown_status(st, r, n_ok)
        return Tick(t_ns, seg, row, n_ok, health, raw, pct, status, shown, reason)

    def push(self, t_ns: int, x: Sequence[float]) -> Tick | None:
        """One reading in; a scored Tick out when it completes a 10 s block."""
        row = self.features.push(t_ns, x)
        if row is None:
            return None
        return self._tick(int(t_ns), self.features.segment, row, self.features.health.copy(),
                          self._scores(row[None, :]), 0)

    def push_many(self, t_ns: np.ndarray, X: np.ndarray) -> list[Tick]:
        """Same as calling push() for every reading, but scores the finished rows in one
        call per detector (scorers are row-wise, so results are identical; used to warm
        up hours of history quickly). tests/test_stream.py checks the equivalence."""
        rows, meta = [], []
        for t, x in zip(np.asarray(t_ns, dtype=np.int64), np.asarray(X)):
            r = self.features.push(int(t), x)
            if r is not None:
                rows.append(r)
                meta.append((int(t), self.features.segment, self.features.health.copy()))
        if not rows:
            return []
        R = np.vstack(rows)
        sc = self._scores(R)
        return [self._tick(t, seg, R[i], h, sc, i) for i, (t, seg, h) in enumerate(meta)]


def ticks_frame(ticks: list[Tick], detector_names: Sequence[str]) -> pd.DataFrame:
    """Compact table of ticks (for charts and tests)."""
    d: dict[str, Any] = {"t_ns": [t.t_ns for t in ticks], "segment": [t.segment for t in ticks],
                         "n_ok": [t.n_ok for t in ticks]}
    for n in detector_names:
        d[f"{n}__raw"] = [t.raw[n] for t in ticks]
        d[f"{n}__pct"] = [t.pct[n] for t in ticks]
        d[f"{n}__status"] = [int(t.status[n]) for t in ticks]
        d[f"{n}__shown"] = [t.shown[n] for t in ticks]
    return pd.DataFrame(d)
