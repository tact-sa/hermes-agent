"""tact-consult: the consultant instructions reach a turn only through ``/consult <question>``.

Loads the bundled plugin through the real ``PluginManager`` against a temp HERMES_HOME and drives
the real ``pre_gateway_dispatch`` hook chain and the real skill discovery, so "only via /consult"
is checked against what the model can actually reach (skill index, ``skill_view``, slash skills).
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTION = "Should we bid on the Riyadh municipality tender or focus on private clients?"


@pytest.fixture
def consult(tmp_path, monkeypatch):
    import hermes_yaml as yaml
    from hermes_cli import plugins as pmod

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["tact-consult"]}}))
    mgr = pmod.PluginManager()
    mgr.discover_and_load()
    loaded = mgr._plugins["tact-consult"]
    assert loaded.enabled, loaded.error
    monkeypatch.setattr(pmod, "_plugin_manager", mgr)
    return loaded.module


def _dispatch(text):
    """Run the gateway's own pre_gateway_dispatch step; the rewritten text, or None if untouched."""
    import asyncio
    from gateway.platforms.event import MessageEvent
    from gateway.run_inbound import GatewayInboundMixin
    event = MessageEvent(text=text)
    out = asyncio.run(GatewayInboundMixin._hm_pre_gateway_dispatch_hook(object(), event, None))
    if out.text == text:
        return None
    assert out.get_command() is None  # the rewritten turn goes to the agent, not command dispatch
    return out.text


def test_consult_is_registered_and_enabled_for_the_deployment(consult):
    import hermes_yaml as yaml
    from hermes_cli.plugins import get_plugin_command_handler

    assert get_plugin_command_handler("consult") is not None
    managed = yaml.safe_load((REPO_ROOT / "tact" / "managed-config.yaml").read_text())
    assert "tact-consult" in managed["plugins"]["enabled"]
    assert {"tact-guard", "tact-meetings"} <= set(managed["plugins"]["enabled"])
    assert "/consult" in (REPO_ROOT / "docker" / "SOUL.md").read_text()


def test_instructions_load_only_via_consult(consult):
    from agent.skill_commands import describe_skill_invocation, extract_user_instruction_from_skill_message
    from agent.skill_commands import get_skill_commands
    from tools.skills_tool import skill_view, skills_list

    turn = _dispatch(f"/consult {QUESTION}")
    assert turn is not None and "🎯 BOTTOM LINE" in turn and "🛑 KILL CRITERION" in turn
    assert turn.rstrip().endswith(QUESTION)
    # Same scaffold as a skill slash command: memory keeps just the question, transcripts show /consult.
    assert extract_user_instruction_from_skill_message(turn) == QUESTION
    assert describe_skill_invocation(turn) == f"/consult — {QUESTION}"
    assert _dispatch(f"/consult@TactBot {QUESTION}") == turn

    # Ordinary messages (even strategy ones, or other commands) are never rewritten...
    for text in (QUESTION, "consult me on the tender", "/meetings", "/consultant hi"):
        assert _dispatch(text) is None, text
    # ...and no model-reachable skill surface carries the instructions.
    assert "/consult" not in get_skill_commands()
    assert not json.loads(skill_view("consult")).get("success")
    assert "BOTTOM LINE" not in skills_list()


def test_bare_consult_replies_with_usage_hint(consult):
    for text in ("/consult", "/consult   ", "/consult@TactBot"):
        assert _dispatch(text) is None, text
    usage = consult._cmd_consult("")
    assert usage.startswith("Usage: /consult <question>") and "e.g. /consult " in usage
    assert consult._cmd_consult("   ") == usage
