"""tact-clock: every turn carries the real current date and time, not the session's start.

Loads the bundled plugin through the real ``PluginManager`` against a temp HERMES_HOME (timezone
Asia/Riyadh, as the Tact deployment pins) and drives Hermes's own per-turn hook collector
(``agent.turn_context._collect_pre_llm_call_context``) and user-message composition.
"""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RIYADH = ZoneInfo("Asia/Riyadh")


@pytest.fixture
def clock(tmp_path, monkeypatch):
    import hermes_time
    import hermes_yaml as yaml
    from hermes_cli import plugins as pmod

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["tact-clock"]},
                                                      "timezone": "Asia/Riyadh"}))
    hermes_time.reset_cache()
    mgr = pmod.PluginManager()
    mgr.discover_and_load()
    loaded = mgr._plugins["tact-clock"]
    assert loaded.enabled, loaded.error
    monkeypatch.setattr(pmod, "_plugin_manager", mgr)
    yield hermes_time
    hermes_time.reset_cache()


def _turn_context(history):
    from agent.turn_context import _collect_pre_llm_call_context
    agent = SimpleNamespace(_persist_disabled=False, session_id="s1", model="m", platform="telegram",
                            _parent_session_id="", _user_id="7")
    return _collect_pre_llm_call_context(agent, effective_task_id="t", turn_id="u", original_user_message="كم الساعة؟",
                                         messages=history, conversation_history=history)


def test_a_late_turn_in_a_long_session_carries_the_current_time(clock, monkeypatch):
    from agent.turn_context import compose_user_api_content
    import hermes_yaml as yaml
    managed = yaml.safe_load((REPO_ROOT / "tact" / "managed-config.yaml").read_text())
    assert "tact-clock" in managed["plugins"]["enabled"] and managed["timezone"] == "Asia/Riyadh"

    started = datetime(2026, 10, 1, 9, 0, tzinfo=RIYADH)
    monkeypatch.setattr(clock, "now", lambda: started)
    first = _turn_context([])
    assert first == ("[Current date and time: Thursday, 01 October 2026, 09:00 (Asia/Riyadh, UTC+03:00). Use this "
                     "for the time, today's date and any relative date (tomorrow, next week, in 2 hours); never guess the time.]")

    # Seven hours and many turns later, the same session: the turn says 16:08, not the start time.
    history = [{"role": "user", "content": "مرحبا"}, {"role": "assistant", "content": "أهلاً"}] * 40
    monkeypatch.setattr(clock, "now", lambda: datetime(2026, 10, 1, 16, 8, tzinfo=RIYADH))
    late = _turn_context(history)
    assert "Thursday, 01 October 2026, 16:08 (Asia/Riyadh, UTC+03:00)" in late and "09:00" not in late
    # It rides the current turn's user message only; earlier turns keep the bytes they were sent with.
    assert compose_user_api_content("كم الساعة؟", "", late) == f"كم الساعة؟\n\n{late}"
    assert "Current date and time" not in str(history)


def test_the_clock_uses_the_configured_timezone_not_the_servers(clock):
    line = _turn_context([])
    now = datetime.now(RIYADH)
    assert "(Asia/Riyadh, UTC+03:00)" in line and now.strftime("%d %B %Y") in line
