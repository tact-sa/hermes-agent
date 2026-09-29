"""tact-meetings: a logged meeting becomes an advance reminder + start reminder in the cron store.

Loads the bundled plugin through the real ``PluginManager`` against a temp HERMES_HOME, dispatches
the ``meeting`` tool through the registry, and runs the generated job scripts with cron's own
script runner, so the cron jobs, the scripts and the SQLite store are all the real ones.
"""

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

RIYADH = ZoneInfo("Asia/Riyadh")


@pytest.fixture
def meetings(tmp_path, monkeypatch):
    import hermes_yaml as yaml
    from hermes_cli import plugins as pmod

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "4242")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["tact-meetings"]}}))
    mgr = pmod.PluginManager()
    mgr.discover_and_load()
    loaded = mgr._plugins["tact-meetings"]
    assert loaded.enabled, loaded.error
    monkeypatch.setattr(pmod, "_plugin_manager", mgr)
    return mgr, loaded.module


def _call(mgr, **args):
    from tools.registry import registry
    return json.loads(registry.dispatch("meeting", args, scope=mgr.scope_key))


def _jobs_for(meeting_id):
    from cron.jobs import list_jobs
    jobs = {j["name"]: j for j in list_jobs(include_disabled=True)}
    return jobs.get(f"Meeting #{meeting_id} card"), jobs.get(f"Meeting #{meeting_id} reminder")


def _script_output(job):
    from cron.scheduler_script import _run_job_script
    ok, out = _run_job_script(job["script"])
    assert ok, out
    return out.strip()


def _start_in(minutes):
    return (datetime.now(RIYADH) + timedelta(minutes=minutes)).replace(second=0, microsecond=0)


def test_logged_meeting_schedules_card_before_and_reminder_at_start(meetings):
    mgr, _ = meetings
    start = _start_in(60)
    res = _call(mgr, action="log", start=start.strftime("%Y-%m-%dT%H:%M"), attendees="Khalid",
                topic="Q4 budget", red_flags=["he'll push for cuts", "avoid committing to dates"],
                key_points=["we need 2 more hires", "highlight Q3 results"])
    assert res["success"], res
    card, reminder = _jobs_for(res["meeting_id"])

    assert datetime.fromisoformat(card["next_run_at"]) == start - timedelta(minutes=10)
    assert datetime.fromisoformat(reminder["next_run_at"]) == start
    for job in (card, reminder):
        assert job["no_agent"] and job["deliver"] == "origin"
        assert job["origin"]["platform"] == "telegram" and job["origin"]["chat_id"] == "4242"

    card_text = _script_output(card)
    assert card_text.startswith("⏰ Reminder: meeting in 10 minutes\n👥 With: Khalid\n")
    assert "- avoid committing to dates" in card_text and "BATTLE CARD" not in card_text
    assert _script_output(reminder).endswith("Top priority: we need 2 more hires")
    confirmation = res["confirmation"]
    assert confirmation.startswith("✅ Logged\n👥 With: Khalid\n🕒 Time: ")
    assert "🎯 Key points: we need 2 more hires; highlight Q3 results" in confirmation
    assert "NOT MENTIONED" not in confirmation
    assert confirmation.endswith(f"/cancel {res['meeting_id']} to cancel")


def test_missing_fields_say_not_mentioned_and_missing_time_logs_nothing(meetings):
    mgr, _ = meetings
    res = _call(mgr, action="log", start=_start_in(45).strftime("%Y-%m-%dT%H:%M"), attendees="Khalid")
    assert res["success"], res
    card, reminder = _jobs_for(res["meeting_id"])
    card_text = _script_output(card)

    assert "🎯 Key points:\n- NOT MENTIONED" in card_text
    assert "🚩 Red flags:\n- NOT MENTIONED" in card_text
    assert "about NOT MENTIONED" in _script_output(reminder)
    assert _script_output(reminder).endswith("Top priority: NOT MENTIONED")
    assert "🎯 Key points: NOT MENTIONED" in res["confirmation"]
    assert "🚩 Red flags: NOT MENTIONED" in res["confirmation"]

    from cron.jobs import list_jobs
    before = len(list_jobs(include_disabled=True))
    assert not _call(mgr, action="log", start="", attendees="Khalid")["success"]
    assert len(list_jobs(include_disabled=True)) == before


def test_meeting_under_ten_minutes_sends_advance_reminder_now(meetings):
    mgr, _ = meetings
    start = _start_in(6)
    res = _call(mgr, action="log", start=start.strftime("%Y-%m-%dT%H:%M"), attendees="Sara")
    assert res["success"], res
    card, reminder = _jobs_for(res["meeting_id"])

    assert datetime.fromisoformat(card["next_run_at"]) <= datetime.now(RIYADH)
    assert datetime.fromisoformat(reminder["next_run_at"]) == start
    assert "advance reminder sending now" in res["confirmation"]
    assert _script_output(card).startswith("⏰ Reminder: meeting in 6 minutes")


def test_cancel_removes_jobs_and_silences_the_meeting(meetings):
    mgr, module = meetings
    res = _call(mgr, action="log", start=_start_in(90).strftime("%Y-%m-%dT%H:%M"), attendees="Omar",
                topic="vendor contract")
    meeting_id = res["meeting_id"]
    card, _ = _jobs_for(meeting_id)
    assert f"#{meeting_id} " in module._cmd_meetings("")

    assert "Cancelled" in module._cmd_cancel(str(meeting_id))
    assert _jobs_for(meeting_id) == (None, None)
    assert f"#{meeting_id} " not in module._cmd_meetings("")
    assert "No scheduled meeting" in module._cmd_cancel(str(meeting_id))
