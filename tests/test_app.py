"""Phase 5: the demo app runs headless, replays, and shows a dead sensor as DEGRADED.
Skipped when streamlit is not installed (pip install -e ".[demo]") or the demo
artifacts are missing (python scripts/export_demo.py)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("streamlit")
pytest.importorskip("plotly")
if not (ROOT / "models" / "detectors.pkl").exists() or not (ROOT / "demo_data" / "manifest.json").exists():
    pytest.skip("demo artifacts missing: run scripts/export_demo.py", allow_module_level=True)

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "app" / "demo.py")


def _app(episode_prefix: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    box = at.sidebar.selectbox[0]
    title = next(o for o in box.options if o.startswith(episode_prefix))
    box.select(title).run()
    assert not at.exception, at.exception
    return at


def _html(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def test_replay_advances_without_errors():
    at = _app("Spurious Closure of DHSV")
    t0 = at.session_state["pos"]
    at.sidebar.button[0].click().run()  # Play
    for _ in range(3):
        at.run()
    assert not at.exception, at.exception
    assert at.session_state["pos"] > t0
    assert "Physics rules (primary)" in _html(at)


def _lamp(at: AppTest, label: str) -> str:
    return _html(at).split(label)[1].split("</div></div>")[0]


def test_silent_blindness_is_shown_as_degraded():
    """The Phase 4 fix. In the hour before a productivity-loss event all 5 sensors are
    healthy; kill P-TPT, T-TPT and P-MON-CKP. Two sensors still work (P-PDG, T-JUS-CKP),
    so the evaluated rule (< 2 usable) says 'not degraded', but no physics rule can be
    evaluated from those two, so the rules cannot score. Before the fix this read
    NORMAL. It must read DEGRADED, and say why."""
    at = _app("Rapid Productivity Loss (WELL-00016, 2018-04-05)")
    for s in ("P-TPT", "T-TPT", "P-MON-CKP"):
        at.sidebar.selectbox(key=f"ov_productivity_both_warn_{s}").select("dead (no data)").run()
    at.sidebar.button[0].click().run()
    for _ in range(8):  # 8 frames x 300 s = 40 min of data, still before the onset
        at.run()
    assert not at.exception, at.exception
    rows = at.session_state["rows"]
    assert rows[-1]["n_ok"] == 2
    rules = _lamp(at, "Physics rules (primary)")
    assert "DEGRADED" in rules and "cannot score" in rules
    assert "NORMAL" in _lamp(at, "Learned model")  # the model still scores from temperatures


def test_stuck_sensor_is_flagged_frozen_one_hour_after_its_last_change():
    """A stuck sensor keeps reporting a plausible value, so it looks healthy until its
    value has not changed for 1 h (quality.FrozenRule). The hour counts from the last
    REAL change: healthy P-TPT is often historian-held, so the run may start before the
    failure."""
    import numpy as np

    from ew3w.quality import Health
    at = _app("Normal operation")
    at.sidebar.radio[0].set_value("1800× (30 min per s)").run()  # 900 s of data per frame
    at.sidebar.selectbox(key="ov_normal_quiet_P-TPT").select("stuck (frozen)").run()
    start = at.session_state["pos"]
    at.sidebar.button[0].click().run()
    for _ in range(6):
        at.run()
    assert not at.exception, at.exception
    S = at.session_state
    x = S["X"][: start, 1]
    last_change = start - 1 - int(np.flatnonzero(x[::-1] != x[-1])[0])  # last sample before the run of x[-1]
    t_run = S["t"][last_change + 1]
    for r in S["rows"]:
        if r["i"] < start:
            continue
        age = (r["t"] - t_run) / 1e9
        if age < 3600 - 20:
            assert int(r["health"][1]) == Health.OK, age
        elif age > 3600 + 20:
            assert int(r["health"][1]) == Health.FROZEN, age
    assert (S["rows"][-1]["t"] - t_run) / 1e9 > 3600 + 20  # the replay got past the flag
