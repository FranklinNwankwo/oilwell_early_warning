"""Phase 5: live replay of the early-warning system on held-out wells.

    streamlit run app/demo.py

Everything shown is computed live by ew3w.stream.EarlyWarningEngine, one 1 Hz reading at
a time, with detectors and thresholds frozen by scripts/export_demo.py on OTHER wells.
The event labels are hidden until the replay passes them, as an operator would see it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ew3w.alerting import alert_active  # noqa: E402
from ew3w.features import FeatureConfig  # noqa: E402
from ew3w.metrics import evaluate_event_instance  # noqa: E402
from ew3w.quality import Health  # noqa: E402
from ew3w.schema import SHARED_SENSORS  # noqa: E402
from ew3w.stream import MIN_OK, EarlyWarningEngine, load_detectors  # noqa: E402

CFG = FeatureConfig()
SERIES = {"physics_rules": "#2a78d6", "gbdt_sim_trained": "#eb6834"}  # categorical slots 1, 2
STATUS = {"NORMAL": ("#0ca30c", "●"), "PENDING": ("#fab219", "◐"), "ALERT": ("#d03b3b", "▲"),
          "CLEARING": ("#ec835a", "◑"), "DEGRADED": ("#8a8984", "■")}
HEALTH = {Health.OK: ("OK", "#0ca30c"), Health.MISSING: ("MISSING", "#8a8984"),
          Health.FROZEN: ("FROZEN", "#fab219"), Health.INVALID: ("INVALID", "#d03b3b")}
UNITS = {s: ("bar", 1e-5) if s.startswith("P") else ("°C", 1.0) for s in SHARED_SENSORS}
SPEEDS = {"60× (1 min per s)": 60, "300×": 300, "600× (10 min per s)": 600, "1800× (30 min per s)": 1800}
FRAME_S = 0.5  # wall-clock seconds between redraws
RARITY_CAP = 1e6

st.set_page_config(page_title="3W early warning: live replay", layout="wide")


# ------------------------------------------------------------------ resources
@st.cache_resource
def resources():
    dets, meta = load_detectors(ROOT / "models" / "detectors.pkl")
    man = json.loads((ROOT / "demo_data" / "manifest.json").read_text(encoding="utf-8"))
    return dets, meta, man["episodes"]


@st.cache_data
def load_clip(file: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "demo_data" / file)


def start_episode(ep: dict) -> None:
    dets, _, _ = resources()
    clip = load_clip(ep["file"])
    eng = EarlyWarningEngine(dets, CFG, detect_frozen=True)
    t = pd.DatetimeIndex(clip.index).as_unit("ns").asi8
    X = clip[list(SHARED_SENSORS)].to_numpy(np.float64)
    i0 = int(np.searchsorted(t, pd.Timestamp(ep["replay_start"]).value))
    S = st.session_state
    S.update(ep=ep, eng=eng, t=t, X=X, pos=0, rows=[], playing=False, overrides={}, held={}, cpu_s=0.0, cpu_n=0,
             cls=clip["class"].to_numpy(np.float64), inst=clip["instance_id"].astype(str).to_numpy(), replay_i0=i0)
    advance(i0)  # warm-up: the hours before the replay window, processed at full speed


def apply_overrides(X: np.ndarray, start: int) -> np.ndarray:
    """Sensor failures chosen in the sidebar, applied to the readings as they arrive."""
    S = st.session_state
    X = X.copy()
    for s, mode in S.overrides.items():
        j = SHARED_SENSORS.index(s)
        if mode == "dead":
            X[:, j] = np.nan
        elif mode == "stuck":
            if s not in S.held:
                prev = S.X[start - 1, j] if start > 0 else np.nan
                S.held[s] = prev
            X[:, j] = S.held[s]
    return X


def advance(n: int) -> None:
    S = st.session_state
    a, b = S.pos, min(S.pos + n, len(S.t))
    if b <= a:
        return
    Xf = apply_overrides(S.X[a:b], a)
    c0 = time.perf_counter()
    ticks = S.eng.push_many(S.t[a:b], Xf)
    S.cpu_s += time.perf_counter() - c0
    S.cpu_n += b - a
    for k in ticks:
        i = a + int(np.searchsorted(S.t[a:b], k.t_ns))
        S.rows.append({"t": k.t_ns, "i": i, "segment": k.segment, "n_ok": k.n_ok, "health": k.health,
                       "x": Xf[i - a], **{f"{n}__{f}": getattr(k, f)[n] for n in k.raw for f in ("raw", "pct", "shown")},
                       **{f"{n}__status": int(k.status[n]) for n in k.raw},
                       **{f"{n}__reason": k.reason[n] for n in k.raw}})
    S.pos = b


# ------------------------------------------------------------------ views
def rarity(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.minimum(1.0 / (1.0 - p), RARITY_CAP)


def spans(mask: np.ndarray, x: pd.DatetimeIndex) -> list[tuple]:
    m = np.r_[False, mask, False]
    d = np.diff(m.astype(int))
    return [(x[s], x[min(e, len(x) - 1)]) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def lamp(det, row) -> str:
    name = det.name
    if row is None:
        shown, reason, p = "DEGRADED", "warming up", np.nan
    else:
        shown, reason, p = row[f"{name}__shown"], row[f"{name}__reason"], row[f"{name}__pct"]
    color, icon = STATUS[shown]
    on = 1.0 / (1.0 - det.calibration.on_percentile)
    if shown == "DEGRADED":
        detail = reason
    else:
        detail = f"score: 1 in {rarity(np.array([p]))[0]:,.0f} normal readings · alarm at 1 in {on:,.0f}"
    return (f"<div style='border-left:6px solid {color};padding:6px 12px;margin-bottom:6px'>"
            f"<div style='font-size:0.85rem;opacity:0.75'>{det.label}</div>"
            f"<div style='font-size:1.6rem;font-weight:600'><span style='color:{color}'>{icon}</span> {shown}</div>"
            f"<div style='font-size:0.85rem;opacity:0.75'>{detail}</div></div>")


def health_chips(row) -> str:
    out = []
    for j, s in enumerate(SHARED_SENSORS):
        h = Health(int(row["health"][j])) if row is not None else Health.MISSING
        label, color = HEALTH[h]
        out.append(f"<span style='display:inline-block;margin:2px 6px 2px 0;padding:2px 8px;border-radius:4px;"
                   f"border:1px solid {color}'><b>{s}</b> {label}</span>")
    return "".join(out)


def charts(rows: pd.DataFrame, dets, ep: dict, now: pd.Timestamp, reveal: bool):
    x = pd.DatetimeIndex(rows["t"].astype("int64"))
    n = len(SHARED_SENSORS)
    fig = make_subplots(rows=n + 1, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[1] * n + [2.2],
                        subplot_titles=[f"{s} ({UNITS[s][0]})" for s in SHARED_SENSORS]
                        + ["Detector score: how unusual vs this site's normal history (1 in N readings, log scale)"])
    X = np.vstack(rows["x"].to_numpy())
    H = np.vstack(rows["health"].to_numpy())
    for j, s in enumerate(SHARED_SENSORS):
        y = X[:, j] * UNITS[s][1]
        y = np.where(np.isfinite(y), y, np.nan)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=1.5, color="#52514e"), name=s,
                                 showlegend=False, hovertemplate=f"{s}: %{{y:.2f}} {UNITS[s][0]}<extra></extra>"),
                      row=j + 1, col=1)
        if not np.isfinite(y).any():
            fig.add_annotation(text="no data from this sensor", xref="x domain", yref="y domain", x=0.5, y=0.5,
                               showarrow=False, font=dict(size=12, color="#8a8984"), row=j + 1, col=1)
        for a, b in spans(H[:, j] == Health.FROZEN, x):
            fig.add_vrect(x0=a, x1=b, fillcolor="#fab219", opacity=0.18, line_width=0, row=j + 1, col=1)
    sr = n + 1
    for d in dets:
        r = rarity(rows[f"{d.name}__pct"].to_numpy(np.float64))
        fig.add_trace(go.Scatter(x=x, y=r, mode="lines", line=dict(width=2, color=SERIES[d.name]), name=d.label,
                                 hovertemplate="1 in %{y:,.0f}<extra>" + d.label + "</extra>"), row=sr, col=1)
        on = 1.0 / (1.0 - d.calibration.on_percentile)
        fig.add_hline(y=on, line=dict(color=SERIES[d.name], width=1, dash="dash"), row=sr, col=1)
        fig.add_annotation(text=f"alarm threshold, {'rules' if d.name == 'physics_rules' else 'model'}",
                           xref="x domain", x=0.0, xanchor="left", y=np.log10(on), yanchor="bottom",  # log axis: log10 units
                           showarrow=False, font=dict(size=11, color="#52514e"), row=sr, col=1)
        act = alert_active(rows[f"{d.name}__status"].to_numpy())
        for a, b in spans(act, x):
            fig.add_vrect(x0=a, x1=b, fillcolor=SERIES[d.name], opacity=0.12, line_width=0, row=sr, col=1)
    blind = (rows["n_ok"].to_numpy() < MIN_OK)
    for a, b in spans(blind, x):
        fig.add_vrect(x0=a, x1=b, fillcolor="#8a8984", opacity=0.15, line_width=0)
    for key, label, side in (("onset", "event starts (label) ", "right"), ("established", "event established (label) ", "right"),
                             ("alarm_at", " false alarm raised", "left")):
        t = ep.get(key)
        if t and (reveal or pd.Timestamp(t) <= now) and pd.Timestamp(t) >= pd.Timestamp(ep["replay_start"]):
            fig.add_vline(x=pd.Timestamp(t), line=dict(color="#0b0b0b", width=1.2, dash="dot"))
            fig.add_annotation(x=pd.Timestamp(t), y=1.0, yref="paper", yanchor="bottom", text=label, showarrow=False,
                               xanchor=side, font=dict(size=11))
        elif key == "onset" and t and pd.Timestamp(t) < pd.Timestamp(ep["replay_start"]) and (reveal or now >= pd.Timestamp(t)):
            fig.add_annotation(x=pd.Timestamp(ep["replay_start"]), y=1.0, yref="paper", yanchor="bottom", xanchor="left",
                               text="(event already in progress when the replay starts)", showarrow=False, font=dict(size=11))
    fig.update_yaxes(type="log", row=sr, col=1, range=[0, 6.2], tickvals=[1, 10, 100, 1e3, 1e4, 1e5, 1e6],
                     ticktext=["1", "10", "100", "1k", "10k", "100k", "≥1M"])
    t0 = pd.Timestamp(ep["replay_start"])
    t1 = pd.Timestamp(ep["end"])
    fig.update_xaxes(range=[t0, t1], showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_layout(height=920, margin=dict(l=10, r=10, t=60, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.035, x=0, bgcolor="rgba(0,0,0,0)"),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    for a in fig.layout.annotations:
        a.font.size = 12
    return fig


def outcome_table(rows: pd.DataFrame, dets, ep: dict) -> pd.DataFrame | None:
    if ep["kind"] != "event":
        return None
    k = rows["inst"].to_numpy() == ep["instance_id"]
    r = rows[k]
    if r.empty or not np.isin(r["cls"].to_numpy(), [ep["event_class"], 100 + ep["event_class"]]).any():
        return None
    out = []
    for d in dets:
        o = evaluate_event_instance(r["t"].to_numpy(np.float64) / 1e9, r["cls"].to_numpy(np.float64),
                                    alert_active(r[f"{d.name}__status"].to_numpy()), ep["event_class"],
                                    segments=r["segment"].to_numpy(), period_s=CFG.block_s)
        if not o.detected:
            live = "no alarm"
        elif np.isfinite(o.lead_s):
            live = f"alarm {o.lead_s / 60:.0f} min BEFORE established" if o.lead_s > 0 else f"alarm {-o.lead_s / 60:.0f} min after"
        else:
            live = "alarm (lead not defined yet)"
        p4 = ep["phase4"][d.name]
        ref = ("no alarm" if not p4["detected"] else
               f"{p4['lead_min']:.0f} min before" if p4["warned"] else "late")
        out.append({"Detector": d.label, "This replay": live, "Phase 4 evaluation (no failures)": ref})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ page
if not (ROOT / "models" / "detectors.pkl").exists():
    st.error("models/detectors.pkl not found. Run `python scripts/export_demo.py` once (needs the dataset).")
    st.stop()
dets, meta, episodes = resources()
S = st.session_state
import sklearn  # noqa: E402

if sklearn.__version__ != meta["sklearn"]:
    st.warning(f"The detectors were saved with scikit-learn {meta['sklearn']}; this environment has "
               f"{sklearn.__version__}. Results may differ slightly. Re-run `python scripts/export_demo.py` "
               f"to re-freeze them here.")
by_title = {e["title"]: e for e in episodes}

with st.sidebar:
    st.header("Replay")
    title = st.selectbox("Episode (held-out wells)", list(by_title), key="episode")
    if "ep" not in S or S.ep["id"] != by_title[title]["id"]:
        with st.spinner("Warming up: streaming the hours before the replay window..."):
            start_episode(by_title[title])
    ep = S.ep
    st.caption(ep["story"])
    speed = SPEEDS[st.radio("Speed", list(SPEEDS), index=2)]
    c1, c2 = st.columns(2)
    if c1.button("⏸ Pause" if S.playing else "▶ Play", width="stretch"):
        S.playing = not S.playing
    if c2.button("↺ Restart", width="stretch"):
        start_episode(ep)
    st.divider()
    st.subheader("Break a sensor")
    st.caption("Applied to readings from now on. A stuck sensor keeps reporting its last value: the health "
               "check needs 1 hour of no change to call it FROZEN, so watch what happens before that.")
    for s in SHARED_SENSORS:
        choice = st.selectbox(s, ["working", "dead (no data)", "stuck (frozen)"], key=f"ov_{ep['id']}_{s}")
        mode = {"working": None, "dead (no data)": "dead", "stuck (frozen)": "stuck"}[choice]
        if mode is None:
            S.overrides.pop(s, None)
            S.held.pop(s, None)
        else:
            if S.overrides.get(s) != mode:
                S.held.pop(s, None)
            S.overrides[s] = mode
    st.divider()
    reveal = st.checkbox("Show event labels now (answer key)", value=False)
    st.caption(f"Detectors fit on fold {meta['fold']}'s training wells only; these episodes come from its "
               f"held-out wells: {', '.join(meta['test_wells'])}.")

st.title("Early warning for offshore oil wells: live replay")
st.caption("Real Petrobras 3W data, replayed one reading per second. Two detectors run side by side: "
           "physics rules (the primary detector) and a model trained only on simulated events. "
           "Thresholds were set on other wells for at most 1 false alarm per well-week.")


@st.fragment(run_every=FRAME_S)
def live():
    if S.playing:
        advance(int(speed * FRAME_S))
        if S.pos >= len(S.t):
            S.playing = False
    rows = pd.DataFrame(S.rows)
    now = pd.Timestamp(int(S.t[min(S.pos, len(S.t)) - 1]))
    shown = rows[rows["t"] >= pd.Timestamp(ep["replay_start"]).value] if len(rows) else rows
    last = rows.iloc[-1] if len(rows) else None
    top = st.columns([1.2, 1.2, 2.2])
    for col, d in zip(top, dets):
        col.markdown(lamp(d, last), unsafe_allow_html=True)
    with top[2]:
        prog = (S.pos - S.replay_i0) / max(len(S.t) - S.replay_i0, 1)
        st.markdown(f"**{now:%Y-%m-%d %H:%M:%S}** · well {ep['well']} · "
                    f"{'playing' if S.playing else 'paused' if S.pos < len(S.t) else 'finished'}")
        st.progress(min(max(prog, 0.0), 1.0))
        st.markdown(health_chips(last), unsafe_allow_html=True)
        if S.overrides:
            st.markdown("Injected: " + ", ".join(f"**{s}** {m}" for s, m in S.overrides.items()))
    if len(shown) >= 2:
        shown = shown.assign(cls=S.cls[shown["i"].to_numpy()], inst=S.inst[shown["i"].to_numpy()])
        st.plotly_chart(charts(shown, dets, ep, now, reveal), width="stretch", key="chart")
        all_rows = rows.assign(cls=S.cls[rows["i"].to_numpy()], inst=S.inst[rows["i"].to_numpy()])
        est = ep.get("established")
        if ep["kind"] == "event" and (reveal or (est and now >= pd.Timestamp(est)) or S.pos >= len(S.t)):
            tab = outcome_table(all_rows, dets, ep)
            if tab is not None:
                st.subheader("Result (scored with the evaluation code)")
                st.dataframe(tab, hide_index=True, width="stretch")
    else:
        st.info("Press ▶ Play.")
    if S.cpu_n:
        us = 1e6 * S.cpu_s / S.cpu_n
        st.caption(f"Processing cost: {us:,.0f} µs per reading including both detectors "
                   f"(capacity ≈ {1e6 / us:,.0f}× real time on this machine). "
                   "Shaded: blue/orange = that detector's alarm; amber = sensor flagged FROZEN; grey = DEGRADED.")


live()
