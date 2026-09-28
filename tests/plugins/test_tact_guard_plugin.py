"""tact-guard: browser submits need a human Approve, secrets and in-page scripts are refused.

Loads the bundled plugin through the real ``PluginManager`` discovery against a temp HERMES_HOME
and drives its hooks exactly as ``model_tools`` does (post_tool_call feeds the snapshot, the
pre_tool_call directive decides).
"""

import json
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

SNAPSHOT = "\n".join([
    '- heading "Clinic appointments" [level=1] [ref=e1]',
    '- textbox "Email" [ref=e2]',
    '- textbox "Password" [ref=e3]',
    '- link "Next" [ref=e4]',
    '- button "Book now" [ref=e12]',
    '  - button "تأكيد الحجز" [ref=e13]',
])


@pytest.fixture
def manager(tmp_path, monkeypatch):
    import hermes_yaml as yaml
    from hermes_cli import plugins as pmod

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["tact-guard"]}}))
    mgr = pmod.PluginManager()
    mgr.discover_and_load()
    loaded = mgr._plugins["tact-guard"]
    assert loaded.enabled, loaded.error
    monkeypatch.setattr(pmod, "_plugin_manager", mgr)
    mgr.invoke_hook(
        "post_tool_call", tool_name="browser_navigate", args={"url": "https://clinic.example.sa/book"},
        result=json.dumps({"success": True, "url": "https://clinic.example.sa/book", "snapshot": SNAPSHOT}),
        task_id="t1", session_id="s1",
    )
    return pmod


def _directive(pmod, tool_name, args):
    return pmod._get_pre_tool_call_directive_details(tool_name, args, task_id="t1", session_id="s1")


@pytest.mark.parametrize("ref,label", [("@e12", "Book now"), ("@e13", "تأكيد الحجز")])
def test_submit_button_click_needs_fresh_approval(manager, ref, label):
    first = _directive(manager, "browser_click", {"ref": ref})
    second = _directive(manager, "browser_click", {"ref": ref})
    assert first.action == "approve"
    assert first.message == f'Click "{label}" on clinic.example.sa'
    assert first.rule_key != second.rule_key  # every submit asks again


def test_unknown_ref_needs_approval_and_plain_navigation_is_allowed(manager):
    assert _directive(manager, "browser_click", {"ref": "@e99"}).action == "approve"
    assert _directive(manager, "browser_click", {"ref": "@e4"}).action is None  # "Next"


def test_enter_needs_approval(manager):
    d = _directive(manager, "browser_press", {"key": "Enter"})
    assert d.action == "approve" and "clinic.example.sa" in d.message
    assert _directive(manager, "browser_press", {"key": "Tab"}).action is None


def test_secret_fields_and_page_scripts_are_blocked(manager):
    d = _directive(manager, "browser_type", {"ref": "@e3", "text": "hunter2"})
    assert d.action == "block" and "manager" in d.message
    assert _directive(manager, "browser_type", {"ref": "@e2", "text": "a@b.sa"}).action is None
    assert _directive(manager, "browser_console", {"expression": "document.forms[0].submit()"}).action == "block"


def test_managed_config_enables_tact_guard(tmp_path, monkeypatch):
    from gateway.run import _load_gateway_config
    from hermes_cli.managed_scope import invalidate_managed_cache

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  default: some/model\n")
    managed = tmp_path / "managed"
    managed.mkdir()
    shutil.copy(REPO_ROOT / "tact" / "managed-config.yaml", managed / "config.yaml")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    invalidate_managed_cache()
    try:
        cfg = _load_gateway_config(home / "config.yaml")
    finally:
        invalidate_managed_cache()
    assert "tact-guard" in cfg["plugins"]["enabled"]
    assert cfg["model"]["default"] == "some/model"  # user config still underneath
