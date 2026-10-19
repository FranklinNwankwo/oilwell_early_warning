"""Alerting with hysteresis: turns a per-second risk score into operator alerts.

A raw threshold on a noisy score flickers on and off, and every flicker is a pager
event. This state machine uses two thresholds and two persistence timers:

  NORMAL --(score >= on_threshold for on_delay_s)--> ALERT
  ALERT  --(score <= off_threshold for off_delay_s)--> NORMAL

with off_threshold < on_threshold (the dead-band). Additional, explicit rules:
  * NaN score (model could not score) is "no evidence": it resets a pending raise
    and does not count toward clearing an active alert.
  * DEGRADED input (too few healthy sensors) blocks *new* alerts and is reported as
    its own status, so the operator sees "I can't see" instead of silence that looks
    like "all clear". An already-active alert is held, not silently dropped.

The same `step()` code path is used by `run()` for offline evaluation and by the
streaming replay service, so evaluated behaviour == deployed behaviour.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .schema import SAMPLE_PERIOD_S


class Status(IntEnum):
    NORMAL = 0
    PENDING = 1  # above on_threshold, waiting out on_delay_s
    ALERT = 2
    CLEARING = 3  # alert active, below off_threshold, waiting out off_delay_s
    DEGRADED = 4  # not alerting AND input health insufficient to trust a "normal"


@dataclass(frozen=True)
class HysteresisConfig:
    on_threshold: float = 0.8
    off_threshold: float = 0.5
    on_delay_s: int = 60
    off_delay_s: int = 300
    period_s: int = SAMPLE_PERIOD_S  # seconds between scores: 1 for raw 1 Hz, FeatureConfig.block_s for features

    def __post_init__(self) -> None:
        if self.period_s <= 0:
            raise ValueError("period_s must be > 0")
        # Scores are NOT assumed to be probabilities: the z-score and isolation-forest
        # baselines produce unbounded scores, so only the dead-band ordering is enforced.
        if not (math.isfinite(self.on_threshold) and math.isfinite(self.off_threshold)):
            raise ValueError("thresholds must be finite")
        if not self.off_threshold < self.on_threshold:
            raise ValueError("require off_threshold < on_threshold (the dead-band)")
        if self.on_delay_s < 0 or self.off_delay_s < 0:
            raise ValueError("delays must be >= 0")

    @property
    def on_steps(self) -> int:
        """Consecutive scores >= on_threshold needed to raise."""
        return max(1, int(round(self.on_delay_s / self.period_s)))

    @property
    def off_steps(self) -> int:
        """Consecutive scores <= off_threshold needed to clear."""
        return max(1, int(round(self.off_delay_s / self.period_s)))


@dataclass
class HysteresisAlerter:
    cfg: HysteresisConfig = field(default_factory=HysteresisConfig)
    _active: bool = False
    _above: int = 0
    _below: int = 0

    def reset(self) -> None:
        """Call at every timeline gap / new well: state must not leak across segments."""
        self._active, self._above, self._below = False, 0, 0

    @property
    def active(self) -> bool:
        return self._active

    def step(self, score: float, degraded: bool = False) -> Status:
        c = self.cfg
        score = float(score)
        finite = math.isfinite(score)  # math, not numpy: ~10x faster per scalar call
        if not self._active:
            if degraded or not finite:
                self._above = 0
                return Status.DEGRADED if degraded else Status.NORMAL
            if score >= c.on_threshold:
                self._above += 1
                if self._above >= c.on_steps:
                    self._active, self._above, self._below = True, 0, 0
                    return Status.ALERT
                return Status.PENDING
            self._above = 0
            return Status.NORMAL
        # active alert
        if finite and score <= c.off_threshold:
            self._below += 1
            if self._below >= c.off_steps:
                self._active, self._below = False, 0
                return Status.DEGRADED if degraded else Status.NORMAL
            return Status.CLEARING
        self._below = 0  # NaN or above off_threshold: hold
        return Status.ALERT

    def run(self, scores: np.ndarray, degraded: np.ndarray | None = None, segments: np.ndarray | None = None) -> np.ndarray:
        """Batch evaluation with identical semantics to repeated step(). Returns Status codes."""
        scores = np.asarray(scores, dtype=np.float64)
        n = scores.size
        deg = np.zeros(n, dtype=bool) if degraded is None else np.asarray(degraded, dtype=bool)
        seg = np.zeros(n, dtype=np.int64) if segments is None else np.asarray(segments)
        out = np.empty(n, dtype=np.int8)
        if deg.shape != scores.shape or seg.shape != scores.shape:
            raise ValueError("scores, degraded and segments must have identical shapes")
        self.reset()
        sc, dg, sg = scores.tolist(), deg.tolist(), seg.tolist()  # python scalars: fast loop
        prev = sg[0] if n else 0
        for i in range(n):
            if sg[i] != prev:
                self.reset()
                prev = sg[i]
            out[i] = self.step(sc[i], dg[i])
        return out


def alert_active(status: np.ndarray) -> np.ndarray:
    status = np.asarray(status)
    return (status == Status.ALERT) | (status == Status.CLEARING)


def alert_episodes(active: np.ndarray, segments: np.ndarray | None = None) -> np.ndarray:
    """(start_idx, end_idx_exclusive) of every contiguous alert episode.

    An episode is closed at a segment boundary: an alert cannot span a data gap.
    """
    a = np.asarray(active, dtype=bool)
    if a.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    seg = np.zeros(a.size, dtype=np.int64) if segments is None else np.asarray(segments)
    brk = np.zeros(a.size, dtype=bool)
    brk[1:] = seg[1:] != seg[:-1]
    starts = np.flatnonzero(a & (np.r_[True, ~a[:-1]] | brk))
    ends = np.flatnonzero(a & (np.r_[~a[1:], True] | np.r_[brk[1:], True])) + 1
    return np.stack([starts, ends], axis=1)
