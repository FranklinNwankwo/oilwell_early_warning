"""Evaluation harness: one code path for every baseline and model, every protocol.

For each protocol (P1-P4) and each well-grouped fold:

  1. TRAIN   the scorer on the protocol's training rows (splits.PROTOCOLS).
  2. CALIBRATE on the protocol's training NORMAL rows only:
       a. map raw scores to "normal percentile" with the ECDF of training-normal scores,
          so every scorer (probabilities, z-scores, anomaly scores) shares one scale;
       b. pick the LOWEST on-threshold whose false-alarm rate on training-normal
          timelines is <= the budget (default 1 per well-week). off-threshold sits one
          decade lower (on = 0.999 -> off = 0.99), giving a dead-band.
     Nothing from the test wells is touched; P1 calibrates on simulated normal data.
  3. TEST    score the held-out wells' stitched timelines, run the alerter over each
     contiguous segment, then score every scorable event instance (metrics.py) and
     count false alarms on normal instances.

Outcomes are pooled over folds (every well is tested exactly once) and summarised with
well-cluster bootstrap CIs and exact Poisson CIs.
"""
from __future__ import annotations

import glob
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .alerting import HysteresisAlerter, HysteresisConfig, alert_active
from .metrics import (
    EventOutcome,
    FalseAlarmCounts,
    count_false_alarms,
    evaluate_event_instance,
    false_alarm_report,
    outcomes_frame,
    summarize_events,
    well_bootstrap_ci,
)
from .splits import Fold

META = ["class", "state", "event_class", "instance_id", "segment", "well", "source", "n_ok"]
# On-threshold candidates in normal-percentile space: 0.99 ... 0.999999 (41 steps in log(1-q)).
# Lower bound 0.99: below it the off-threshold (one decade lower) falls toward 0 and an
# alert can never clear; a never-clearing alarm counts as ONE false-alarm episode and
# games the episode budget (seen in Phase 2's first run: 50% time in alarm at "0.22 FA/day").
# Upper end: on some folds only the 1-in-a-million tail keeps alarms under budget.
PERCENTILE_GRID = 1.0 - np.logspace(-2, -6, 41)
TIME_IN_ALARM_BUDGET = 0.01  # max fraction of normal time in alarm (~15 min/day)


def off_of(on: float) -> float:
    """Off-threshold one decade below on (0.999 -> 0.99): the hysteresis dead-band."""
    return 1.0 - min(1.0, 10.0 * (1.0 - on))


# --------------------------------------------------------------------------- data
def load_cache(cache_dir: str | Path, columns: list[str], synth_normal_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(real, synthetic) feature rows with only `columns` + META, memory-lean.

    synth_normal_only: keep only normal-labeled synthetic rows (enough for scorers that
    never train on events), filtered per file so the full table is never in memory."""
    cache_dir = Path(cache_dir)
    if not (cache_dir / "config.json").exists():
        raise FileNotFoundError(f"{cache_dir} is not a feature cache (run scripts/build_features.py)")
    cols = list(dict.fromkeys([*columns, *META]))

    def read(pattern: str, normal_only: bool = False) -> pd.DataFrame:
        files = sorted(glob.glob(str(cache_dir / pattern)))
        if not files:
            raise FileNotFoundError(f"no files match {cache_dir / pattern}")
        parts = []
        for f in files:
            d = pd.read_parquet(f, columns=cols)
            parts.append(d[d["class"] == 0] if normal_only else d)
        df = pd.concat(parts)
        for c in ("instance_id", "well", "source"):
            df[c] = df[c].astype("category")
        return df

    synth = pd.concat([read("simulated/*.parquet", synth_normal_only), read("drawn/*.parquet", synth_normal_only)])
    for c in ("instance_id", "well", "source"):
        synth[c] = synth[c].astype("category")
    return read("real/*.parquet"), synth


def event_target(df: pd.DataFrame) -> np.ndarray:
    """1 = event in progress (transient or established), 0 = normal, NaN = unlabeled."""
    c = df["class"].to_numpy(dtype=np.float64)
    return np.where(np.isnan(c), np.nan, (c != 0).astype(np.float64))


def training_rows(protocol: str, real: pd.DataFrame, synth: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    """Row-level version of splits.training_selection (labeled rows only)."""
    tr_real = real[real["well"].isin(fold.train_wells)]
    if protocol == "P1_sim_only":
        rows = synth
    elif protocol == "P2_sim_plus_site_normal":
        rows = pd.concat([synth, tr_real[tr_real["event_class"] == 0]])
    elif protocol == "P3_real":
        rows = tr_real
    elif protocol == "P4_real_plus_sim":
        rows = pd.concat([synth, tr_real])
    else:
        raise ValueError(protocol)
    return rows[rows["class"].notna()]


def calibration_normal(protocol: str, real: pd.DataFrame, synth: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    """Normal rows that set the score scale and the threshold (never test wells).
    Real: rows of NORMAL instances only (the clean negative population, as at test time).
    Simulated: normal-labeled lead-ins (the only normal operation simulation has)."""
    syn_norm = synth[synth["class"] == 0]
    tr_real = real[real["well"].isin(fold.train_wells)]
    real_norm = tr_real[(tr_real["event_class"] == 0) & (tr_real["class"] == 0)]
    if protocol == "P1_sim_only":
        return syn_norm
    if protocol in ("P2_sim_plus_site_normal", "P3_real"):
        return real_norm
    return pd.concat([syn_norm, real_norm])


# --------------------------------------------------------------------------- scorers
class Scorer(Protocol):
    name: str
    columns: list[str]
    needs_events: bool  # False: fit() uses normal rows only, so event rows are never loaded or copied
    fitted: bool  # True: learns from data, so it is calibrated on out-of-fold scores

    def fit(self, train: pd.DataFrame, normal: pd.DataFrame) -> "Scorer": ...

    def score(self, df: pd.DataFrame) -> np.ndarray: ...


@dataclass
class Calibration:
    sorted_normal: np.ndarray
    on_percentile: float
    off_percentile: float
    train_fa_per_day: float
    train_normal_days: float
    train_time_in_alarm: float = np.nan
    meets_budget: bool = True  # False: even the strictest threshold broke a training budget

    def to_percentile(self, raw: np.ndarray) -> np.ndarray:
        """Mid-rank ECDF. Ties matter: a scorer that outputs one constant (e.g. health-only
        trained on simulation, where every sensor is always healthy) must map to 0.5, not to
        1.0, or it would sit permanently above every threshold and alarm non-stop."""
        raw = np.asarray(raw, dtype=np.float64)
        n = max(self.sorted_normal.size, 1)
        lo = np.searchsorted(self.sorted_normal, raw, side="left")
        hi = np.searchsorted(self.sorted_normal, raw, side="right")
        p = (lo + hi) / (2.0 * n)
        p[~np.isfinite(raw)] = np.nan
        return p


def _ns(index: pd.Index) -> np.ndarray:
    return pd.DatetimeIndex(index).as_unit("ns").asi8


def _runs(df: pd.DataFrame, period_s: int) -> np.ndarray:
    """Run id for alerter state: breaks at every well / segment change and wherever
    consecutive rows are not exactly one period apart (e.g. a subset of rows such as
    normal instances only, or concatenated synthetic instances)."""
    key = df["well"].astype(str).to_numpy()
    seg = df["segment"].to_numpy()
    t = _ns(df.index)
    brk = np.r_[True, (key[1:] != key[:-1]) | (seg[1:] != seg[:-1]) | (np.diff(t) != period_s * 1_000_000_000)]
    return np.cumsum(brk)


def _fa_rate(p: np.ndarray, labels: np.ndarray, state: np.ndarray, deg: np.ndarray, runs: np.ndarray,
             on: float, off: float, alert_cfg: HysteresisConfig) -> tuple[float, float, float]:
    """(false alarms per day, exposure days, time-in-alarm fraction) on normal rows."""
    cfg = HysteresisConfig(on, off, alert_cfg.on_delay_s, alert_cfg.off_delay_s, alert_cfg.period_s)
    st = HysteresisAlerter(cfg).run(p, degraded=deg, segments=runs)
    c = count_false_alarms(labels, state, alert_active(st), segments=runs, period_s=alert_cfg.period_s)
    days = c.exposure_s / 86_400
    if days <= 0:
        return np.inf, 0.0, 1.0
    return c.episodes / days, days, c.time_in_alarm_s / c.exposure_s


def calibrate(raw_normal: np.ndarray, normal: pd.DataFrame, alert_cfg: HysteresisConfig,
              fa_budget_per_day: float, tia_budget: float = TIME_IN_ALARM_BUDGET,
              max_ecdf: int = 2_000_000, seed: int = 0) -> Calibration:
    """ECDF on training-normal scores + LOWEST on-threshold meeting BOTH budgets:
    false-alarm episodes per day <= fa_budget_per_day AND time in alarm <= tia_budget.

    The check runs the alerter over the calibration normal rows themselves, split into
    contiguous runs. Within the grid (>= 0.99) both quantities are non-increasing in the
    threshold in practice, so bisection; tests/test_evaluation.py checks the choice."""
    finite = raw_normal[np.isfinite(raw_normal)]
    if finite.size == 0:
        raise ValueError("scorer produced no finite scores on calibration normal rows")
    rng = np.random.default_rng(seed)
    ref = rng.choice(finite, max_ecdf, replace=False) if finite.size > max_ecdf else finite
    cal = Calibration(np.sort(ref), np.nan, np.nan, np.nan, np.nan)
    p = cal.to_percentile(raw_normal)
    labels = normal["class"].to_numpy(dtype=np.float64)
    state = normal["state"].to_numpy(dtype=np.float64)
    deg = normal["n_ok"].to_numpy() < 2 if "n_ok" in normal else np.zeros(len(normal), bool)
    runs = _runs(normal, alert_cfg.period_s)

    def check(i: int) -> tuple[bool, float, float, float]:
        on = PERCENTILE_GRID[i]
        r, d, tia = _fa_rate(p, labels, state, deg, runs, on, off_of(on), alert_cfg)
        return (r <= fa_budget_per_day and tia <= tia_budget), r, d, tia

    lo, hi = 0, len(PERCENTILE_GRID) - 1
    ok_hi, rate, days, tia = check(hi)
    best = hi
    if ok_hi:  # bisect for the lowest threshold that still meets both budgets
        while lo <= hi:
            mid = (lo + hi) // 2
            ok, r, _, t = check(mid)
            if ok:
                best, rate, tia, hi = mid, r, t, mid - 1
            else:
                lo = mid + 1
    on = float(PERCENTILE_GRID[best])
    return Calibration(cal.sorted_normal, on, off_of(on), float(rate), float(days), float(tia), bool(ok_hi))


# --------------------------------------------------------------------------- test
def evaluate_wells(test: pd.DataFrame, p: np.ndarray, cal: Calibration, alert_cfg: HysteresisConfig,
                   grace_s: float = 600.0) -> tuple[list[EventOutcome], list[FalseAlarmCounts]]:
    """Alerter over each held-out well's stitched timeline, then event + FA scoring."""
    outcomes: list[EventOutcome] = []
    fas: list[FalseAlarmCounts] = []
    cfg = HysteresisConfig(cal.on_percentile, cal.off_percentile, alert_cfg.on_delay_s, alert_cfg.off_delay_s, alert_cfg.period_s)
    wells = test["well"].astype(str).to_numpy()
    for w in pd.unique(wells):
        m = wells == w
        d = test[m]
        runs = _runs(d, alert_cfg.period_s)
        deg = d["n_ok"].to_numpy() < 2
        act = alert_active(HysteresisAlerter(cfg).run(p[m], degraded=deg, segments=runs))
        lab = d["class"].to_numpy(dtype=np.float64)
        ev = d["event_class"].to_numpy()
        norm_inst = ev == 0
        if norm_inst.any():
            fas.append(count_false_alarms(np.where(norm_inst, lab, np.nan), d["state"].to_numpy(dtype=np.float64),
                                          act, well=w, segments=runs, period_s=alert_cfg.period_s))
        tn = _ns(d.index)
        t_s = (tn - tn[0]) / 1e9
        inst = d["instance_id"].astype(str).to_numpy()
        for iid in pd.unique(inst[ev > 0]):
            k = inst == iid
            c = int(ev[k][0])
            if not np.isin(lab[k], [c, 100 + c]).any():
                continue  # unscorable (e.g. 43 class-9 instances with no event label)
            outcomes.append(evaluate_event_instance(t_s[k], lab[k], act[k], c, instance_id=iid, well=w, source="real",
                                                    grace_s=grace_s, segments=runs[k], period_s=alert_cfg.period_s))
    return outcomes, fas


@dataclass
class RunResult:
    scorer: str
    protocol: str
    outcomes: pd.DataFrame
    fa: list[FalseAlarmCounts]
    calibrations: dict[int, Calibration] = field(default_factory=dict)
    seconds: float = 0.0
    test_percentiles: dict[int, np.ndarray] = field(default_factory=dict)  # per fold, for operating curves


def out_of_fold_scores(make_scorer: Callable[[], Scorer], train: pd.DataFrame, normal: pd.DataFrame,
                       inner_folds: int = 3, seed: int = 0) -> np.ndarray:
    """Scores for the calibration-normal rows from models that never saw those rows' wells.

    A fitted scorer calibrated on its own training rows is optimistic: a classifier that
    memorises its training wells scores their normal data low, the threshold comes out too
    permissive, and unseen wells over-alarm (Phase 2: health_only P3 was 0.1% in alarm on
    training normal vs 50% on test wells). So the training wells are split into
    `inner_folds` groups; each group's normal rows are scored by a model fit without it."""
    cal_w = normal["well"].astype(str).to_numpy()
    tr_w = train["well"].astype(str).to_numpy()
    groups = np.array_split(np.random.default_rng(seed).permutation(np.unique(cal_w)), min(inner_folds, len(np.unique(cal_w))))
    oof = np.full(len(normal), np.nan)
    for hold in groups:
        m = np.isin(cal_w, hold)
        inner = make_scorer().fit(train[~np.isin(tr_w, hold)], normal[~m])
        oof[m] = inner.score(normal[m])
    return oof


def run_protocol(make_scorer: Callable[[], Scorer], protocol: str, real: pd.DataFrame, synth: pd.DataFrame,
                 folds: list[Fold], alert_cfg: HysteresisConfig, fa_budget_per_day: float,
                 grace_s: float = 600.0, log: Callable[[str], None] = print,
                 inner_folds: int = 3, seed: int = 0) -> RunResult:
    """Fitted scorers (attribute fitted=True, the default) are calibrated on out-of-fold
    scores within the training wells; rule-based scorers (fitted=False) on their own scores."""
    t0 = time.perf_counter()
    outs: list[EventOutcome] = []
    fas: list[FalseAlarmCounts] = []
    cals: dict[int, Calibration] = {}
    tps: dict[int, np.ndarray] = {}
    for fold in folds:
        sc = make_scorer()
        normal = calibration_normal(protocol, real, synth, fold)
        train = training_rows(protocol, real, synth, fold) if sc.needs_events else normal
        sc.fit(train, normal)
        raw_cal = (out_of_fold_scores(make_scorer, train, normal, inner_folds, seed)
                   if getattr(sc, "fitted", True) else sc.score(normal))
        cal = calibrate(raw_cal, normal, alert_cfg, fa_budget_per_day)
        cals[fold.k] = cal
        test = real[real["well"].isin(fold.test_wells)]
        p_test = cal.to_percentile(sc.score(test))
        tps[fold.k] = p_test.astype(np.float32)
        o, f = evaluate_wells(test, p_test, cal, alert_cfg, grace_s)
        outs += o
        fas += f
        log(f"    fold {fold.k}: on-pct {cal.on_percentile:.6f}, train FA {cal.train_fa_per_day:.3f}/day, "
            f"in alarm {100 * cal.train_time_in_alarm:.2f}% over {cal.train_normal_days:.1f} d"
            f"{'' if cal.meets_budget else ' [BUDGET NOT MET]'}; test {len(o)} events")
    return RunResult(make_scorer().name, protocol, outcomes_frame(outs), fas, cals, time.perf_counter() - t0, tps)


def operating_curve(r: RunResult, real: pd.DataFrame, folds: list[Fold], alert_cfg: HysteresisConfig,
                    grid: np.ndarray | None = None, grace_s: float = 600.0) -> pd.DataFrame:
    """Warned-before-established vs false alarms per well-day across on-thresholds, on the
    test folds. For REFERENCE ONLY: the reported operating point is the one chosen on
    training data (run_protocol); picking a point from this curve would be test-set tuning."""
    grid = PERCENTILE_GRID[::5] if grid is None else grid
    rows = []
    for on in grid:
        outs: list[EventOutcome] = []
        fas: list[FalseAlarmCounts] = []
        for fold in folds:
            c = r.calibrations[fold.k]
            cal = Calibration(c.sorted_normal, float(on), off_of(float(on)), np.nan, np.nan)
            test = real[real["well"].isin(fold.test_wells)]
            o, f = evaluate_wells(test, r.test_percentiles[fold.k].astype(np.float64), cal, alert_cfg, grace_s)
            outs += o
            fas += f
        df = outcomes_frame(outs)
        d = df[df.onset_observable & ~df.censored]
        fa = false_alarm_report(fas).iloc[0]
        rows.append(dict(on_percentile=on, warned_before_established=d.warned_before_established.mean(),
                         event_recall=df.detected.mean(), fa_per_well_day=fa.fa_per_well_day))
    return pd.DataFrame(rows)


def headline(r: RunResult, n_boot: int = 2000) -> dict:
    """One results-table row: warned-before-established, recall, lead, FA with CIs."""
    df = r.outcomes
    lead_def = df[df.onset_observable & ~df.censored]
    fa = false_alarm_report(r.fa)
    allst = fa.iloc[0]

    def warned(d: pd.DataFrame) -> float:
        d = d[d.onset_observable & ~d.censored]
        return d.warned_before_established.mean() if len(d) else np.nan

    def med_lead(d: pd.DataFrame) -> float:
        x = d.lead_s.dropna()
        return x.median() / 60 if len(x) else np.nan

    w = well_bootstrap_ci(df, warned, n_boot=n_boot)
    lead = well_bootstrap_ci(df[df.detected], med_lead, n_boot=n_boot) if df.detected.any() else (np.nan,) * 3
    return {
        "scorer": r.scorer,
        "protocol": r.protocol,
        "warned_before_established": w[0],
        "warned_ci": (w[1], w[2]),
        "n_lead_defined": len(lead_def),
        "event_recall": df.detected.mean(),
        "n_events": len(df),
        "median_lead_min": lead[0],
        "lead_ci": (lead[1], lead[2]),
        "fa_per_well_day": allst.fa_per_well_day,
        "fa_ci": (allst.ci_lo, allst.ci_hi),
        "false_alarms": int(allst.false_alarms),
        "normal_well_days": allst.well_days,
        "time_in_alarm": fa.attrs["time_in_alarm_fraction"],
        "class4_recall": df[df.event_class == 4].detected.mean() if (df.event_class == 4).any() else np.nan,
        "seconds": r.seconds,
    }


def per_class(r: RunResult) -> pd.DataFrame:
    return summarize_events(r.outcomes)
