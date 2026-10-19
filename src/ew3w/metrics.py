"""Event-level early-warning metrics. Accuracy is deliberately absent.

Definitions (all times in seconds, all computed from ALERTS, i.e. after hysteresis,
never from raw per-sample predictions):

For one event instance of class c:
  onset        first sample labeled transient (100 + c). If the instance has no
               transient label, the first sample labeled c, and onset_observable=False
               (always the case for classes 3 and 4).
  established  first sample labeled c (steady state). None => right-censored: the
               event never became established inside the recording (common for
               class 7 Scaling in PCK, where 31 of 36 real instances never do).
  window       [onset - grace_s, event_end). Alerts in the window are detections.
               grace_s applies only when the onset is observable (not classes 3/4).
  t_alert      time of the first active alert sample in the window. If an alert raised
               before the window is still active when the window opens, t_alert is
               the window start: early alerts get at most grace_s of credit, so a
               detector cannot buy lead time by alarming constantly.
  delay_s      t_alert - onset (negative values down to -grace_s are allowed).
  lead_s       established - t_alert. > 0 means the operator was warned BEFORE the
               event became established: this is the headline number.
  pre_onset_alerts
               alert episodes raised on normal-labeled samples before the window.
               Reported per instance, not mixed into the false-alarm rate, because
               expert labels can be late and these may be genuine precursors.

False alarms (normal instances only, the clean negative population):
  An alert EPISODE whose first sample is a normal-labeled (class 0) sample.
  rate = episodes / labeled-normal well-days, with an exact Poisson CI.
  Exposure counts only labeled class-0 seconds of a de-duplicated timeline.
  Reported overall and split by well state (Open vs any non-Open or unknown state),
  because alarms during shut-in / restart are operationally different.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .alerting import alert_episodes
from .schema import NO_TRANSIENT_CLASSES, SAMPLE_PERIOD_S, STATE_OPEN, TRANSIENT_OFFSET


@dataclass(frozen=True)
class EventOutcome:
    instance_id: str
    event_class: int
    well: str | None
    source: str
    onset_observable: bool
    detected: bool
    delay_s: float  # NaN if not detected
    lead_s: float  # NaN if not detected or censored
    censored: bool  # established phase never observed
    warned_before_established: bool  # detected and lead_s > 0 (False if censored)
    pre_onset_alerts: int
    event_duration_s: float


def _first(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def evaluate_event_instance(
    t_s: np.ndarray,
    labels: np.ndarray,
    active: np.ndarray,
    event_class: int,
    *,
    instance_id: str = "",
    well: str | None = None,
    source: str = "real",
    grace_s: float = 600.0,
    segments: np.ndarray | None = None,
    period_s: float = SAMPLE_PERIOD_S,
) -> EventOutcome:
    """Score one event instance. period_s = seconds per row (1 raw, 10 for features)."""
    t_s = np.asarray(t_s, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if not (t_s.shape == labels.shape == active.shape):
        raise ValueError("t_s, labels, active must have identical shapes")
    if not 1 <= event_class <= 9:
        raise ValueError("event_class must be 1..9")
    is_tr = labels == TRANSIENT_OFFSET + event_class
    is_ss = labels == event_class
    i_tr, i_ss = _first(is_tr), _first(is_ss)
    if i_tr is None and i_ss is None:
        raise ValueError(f"{instance_id}: no samples labeled {event_class} or {TRANSIENT_OFFSET + event_class}")
    onset_observable = i_tr is not None and event_class not in NO_TRANSIENT_CLASSES
    i_on = i_tr if i_tr is not None else i_ss
    t_on = t_s[i_on]
    ev_idx = np.flatnonzero(is_tr | is_ss)
    # One instance = one contiguous event run. A second run means the caller passed a
    # whole stitched well timeline (or several instances): slice by instance_id first,
    # otherwise only the first event would be scored, silently.
    if ev_idx.size and np.any(np.diff(ev_idx) > 1):
        raise ValueError(f"{instance_id}: event labels form more than one contiguous run; pass one instance")
    i_end = int(ev_idx[-1]) + 1
    t_end = t_s[i_end - 1] + period_s
    censored = i_ss is None
    t_est = np.nan if censored else t_s[i_ss]

    # The grace window exists because transient labels may lag the physics. With no
    # observable onset (classes 3/4) the recording opens mid-event and the samples
    # before it belong to another instance, so no early credit is given.
    win_start_t = t_on - (grace_s if onset_observable else 0.0)
    in_win = (t_s >= win_start_t) & (np.arange(t_s.size) < i_end)
    det_idx = _first(in_win & active)
    detected = det_idx is not None
    if detected:
        t_alert = max(t_s[det_idx], win_start_t)
        delay = t_alert - t_on
        # Without an observable onset (classes 3/4) "lead before established" is
        # meaningless: the recording starts with the event already established.
        lead = np.nan if (censored or not onset_observable) else t_est - t_alert
    else:
        delay = lead = np.nan

    eps = alert_episodes(active, segments)
    pre = 0
    for s, _ in eps:
        if t_s[s] < win_start_t and labels[s] == 0:
            pre += 1
    return EventOutcome(
        instance_id=instance_id,
        event_class=event_class,
        well=well,
        source=source,
        onset_observable=onset_observable,
        detected=detected,
        delay_s=float(delay),
        lead_s=float(lead),
        censored=censored,
        warned_before_established=bool(detected and np.isfinite(lead) and lead > 0),
        pre_onset_alerts=pre,
        event_duration_s=float(t_end - t_on),
    )


@dataclass(frozen=True)
class FalseAlarmCounts:
    well: str | None
    episodes_open: int
    episodes_not_open: int
    exposure_open_s: float
    exposure_not_open_s: float
    time_in_alarm_s: float

    @property
    def episodes(self) -> int:
        return self.episodes_open + self.episodes_not_open

    @property
    def exposure_s(self) -> float:
        return self.exposure_open_s + self.exposure_not_open_s


def count_false_alarms(
    labels: np.ndarray,
    state: np.ndarray,
    active: np.ndarray,
    *,
    well: str | None = None,
    segments: np.ndarray | None = None,
    period_s: float = SAMPLE_PERIOD_S,
) -> FalseAlarmCounts:
    """For a NORMAL-instance timeline. Only class-0 samples count as exposure; an
    episode counts if it STARTS on a class-0 sample (unlabeled lead-in is ignored).
    Each row stands for period_s seconds of exposure (1 for raw data, 10 for feature rows)."""
    labels = np.asarray(labels, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    normal = labels == 0
    is_open = state == STATE_OPEN
    eps = alert_episodes(active, segments)
    e_open = e_not = 0
    for s, _ in eps:
        if normal[s]:
            if is_open[s]:
                e_open += 1
            else:
                e_not += 1
    return FalseAlarmCounts(
        well=well,
        episodes_open=e_open,
        episodes_not_open=e_not,
        exposure_open_s=float((normal & is_open).sum() * period_s),
        exposure_not_open_s=float((normal & ~is_open).sum() * period_s),
        time_in_alarm_s=float((normal & active).sum() * period_s),
    )


def poisson_rate_ci(k: int, exposure_days: float, alpha: float = 0.05) -> tuple[float, float, float]:
    """Exact (Garwood) CI for a Poisson rate. Returns (rate, lo, hi) per day."""
    if exposure_days <= 0:
        return (np.nan, np.nan, np.nan)
    lo = 0.0 if k == 0 else stats.chi2.ppf(alpha / 2, 2 * k) / 2
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    return (k / exposure_days, lo / exposure_days, hi / exposure_days)


def false_alarm_report(counts: list[FalseAlarmCounts], alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for name, k, expo in [
        ("all states", sum(c.episodes for c in counts), sum(c.exposure_s for c in counts)),
        ("Open", sum(c.episodes_open for c in counts), sum(c.exposure_open_s for c in counts)),
        ("not Open or unknown", sum(c.episodes_not_open for c in counts), sum(c.exposure_not_open_s for c in counts)),
    ]:
        days = expo / 86_400
        r, lo, hi = poisson_rate_ci(k, days, alpha)
        rows.append(dict(scope=name, false_alarms=k, well_days=round(days, 2), fa_per_well_day=r, ci_lo=lo, ci_hi=hi))
    tia = sum(c.time_in_alarm_s for c in counts) / max(sum(c.exposure_s for c in counts), 1.0)
    df = pd.DataFrame(rows)
    df.attrs["time_in_alarm_fraction"] = tia
    return df


def outcomes_frame(outcomes: list[EventOutcome]) -> pd.DataFrame:
    return pd.DataFrame([asdict(o) for o in outcomes])


def summarize_events(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class event metrics. Lead/delay medians are over detected instances only,
    and n_lead counts how many instances actually contribute (watch for n_lead < 5)."""

    def agg(g: pd.DataFrame) -> pd.Series:
        obs = g[g.onset_observable]
        unc = obs[~obs.censored]  # instances where "warned before established" is defined
        det = g[g.detected]
        return pd.Series(
            {
                "n_instances": len(g),
                "n_wells": g.well.nunique(),
                "event_recall": g.detected.mean(),
                "n_uncensored": len(unc),
                "warned_before_established": unc.warned_before_established.mean() if len(unc) else np.nan,
                "median_delay_min": obs[obs.detected].delay_s.median() / 60 if len(obs) else np.nan,
                "median_lead_min": det.lead_s.median() / 60,
                "n_lead": int(det.lead_s.notna().sum()),
                "pre_onset_alerts_per_inst": g.pre_onset_alerts.mean(),
            }
        )

    per_class = df.groupby("event_class").apply(agg, include_groups=False)
    overall = agg(df).to_frame("all").T
    macro = per_class[["event_recall", "warned_before_established"]].mean().to_frame("macro").T
    return pd.concat([per_class, overall, macro])


def well_bootstrap_ci(
    df: pd.DataFrame, stat, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Cluster bootstrap over wells: instances from one well are not independent, and
    with 3-15 wells per class an instance-level bootstrap would be badly overconfident."""
    rng = np.random.default_rng(seed)
    wells = df.well.fillna(df.instance_id).to_numpy()
    uniq = np.unique(wells)
    groups = {w: df[wells == w] for w in uniq}
    point = float(stat(df))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        v = stat(pd.concat([groups[w] for w in pick]))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (point, np.nan, np.nan)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return (point, float(lo), float(hi))
