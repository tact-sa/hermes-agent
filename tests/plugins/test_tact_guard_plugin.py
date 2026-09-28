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


# agent-browser 0.26 (Railway): ref shares one attribute bracket with other attributes.
SNAPSHOT_AB026 = "\n".join([
    '- heading "Web form" [level=1, ref=e1]',
    '- combobox "Dropdown (select) " [expanded=false, ref=e8]: One',
    '- option "One" [selected, ref=e9]',
    '- option "Two" [ref=e10]',
    '- checkbox " Checked checkbox" [checked=true, ref=e2]',
    '- radio " Checked radio" [checked=true, ref=e3]',
    '- button "Submit" [ref=e4]',
])


@pytest.fixture
def loaded_plugin(tmp_path, monkeypatch):
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
    return pmod, mgr, loaded.module


def _feed(mgr, snapshot, url):
    mgr.invoke_hook(
        "post_tool_call", tool_name="browser_navigate", args={"url": url},
        result=json.dumps({"success": True, "url": url, "snapshot": snapshot}),
        task_id="t1", session_id="s1",
    )


@pytest.fixture
def manager(loaded_plugin):
    pmod, mgr, _ = loaded_plugin
    _feed(mgr, SNAPSHOT, "https://clinic.example.sa/book")
    return pmod


@pytest.fixture
def web_form(loaded_plugin):
    pmod, mgr, _ = loaded_plugin
    extra = ['- button [ref=e20]', '- link "Home" [ref=e21]']
    _feed(mgr, "\n".join([SNAPSHOT_AB026, *extra]), "https://www.selenium.dev/selenium/web/web-form.html")
    return pmod


def _directive(pmod, tool_name, args, tool_call_id=""):
    return pmod._get_pre_tool_call_directive_details(tool_name, args, task_id="t1", session_id="s1",
                                                     tool_call_id=tool_call_id)


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


def test_agent_browser_attribute_brackets_parse(loaded_plugin):
    _, _, module = loaded_plugin
    assert module.parse_snapshot(SNAPSHOT_AB026) == {
        "e1": ("heading", "Web form"),
        "e8": ("combobox", "Dropdown (select)"),
        "e9": ("option", "One"),
        "e10": ("option", "Two"),
        "e2": ("checkbox", "Checked checkbox"),
        "e3": ("radio", "Checked radio"),
        "e4": ("button", "Submit"),
    }


@pytest.mark.parametrize("ref", ["@e2", "@e3", "@e8", "@e9", "@e10", "@e21"])
def test_select_toggle_and_plain_link_clicks_are_allowed(web_form, ref):
    assert _directive(web_form, "browser_click", {"ref": ref}).action is None


@pytest.mark.parametrize("ref", ["@e4", "@e20", "@e99"])  # "Submit", unlabeled button, unknown ref
def test_submit_capable_clicks_need_approval(web_form, ref):
    assert _directive(web_form, "browser_click", {"ref": ref}).action == "approve"


def test_password_vault_is_blocked(web_form):
    d = _directive(web_form, "browser_vault_fill", {"ref": "@e4"})
    assert d.action == "block"
    assert d.message == "Saved passwords are disabled. Ask the manager to log in themselves."


LINKS = "\n".join([
    '- link "Book a test drive" [ref=e30]',
    '- link "احجز موعد" [ref=e31]',
    '- link "Schedule a visit" [ref=e32]',
    '- link "Pay now" [ref=e33]',
    '- link "تأكيد الحجز" [ref=e34]',
    '- link "Cancel booking" [ref=e35]',
])


@pytest.mark.parametrize("ref,action", [
    ("@e30", None), ("@e31", None), ("@e32", None),  # links to a booking page only navigate
    ("@e33", "approve"), ("@e34", "approve"), ("@e35", "approve"),  # pay / confirm / cancel still ask
])
def test_links_ask_only_for_non_booking_submit_words(loaded_plugin, ref, action):
    pmod, mgr, _ = loaded_plugin
    _feed(mgr, LINKS, "https://dealer.example.sa/")
    assert _directive(pmod, "browser_click", {"ref": ref}).action == action


def _click_ran(mgr, ref, tool_call_id, status):
    mgr.invoke_hook("post_tool_call", tool_name="browser_click", args={"ref": ref}, result='{"success": true}',
                    task_id="t1", session_id="s1", tool_call_id=tool_call_id, status=status)


def test_approved_click_retry_is_allowed_for_120_seconds(loaded_plugin, monkeypatch):
    pmod, mgr, module = loaded_plugin
    clock = [1000.0]
    monkeypatch.setattr(module, "_now", lambda: clock[0])
    _feed(mgr, SNAPSHOT_AB026, "https://www.selenium.dev/selenium/web/web-form.html")

    assert _directive(pmod, "browser_click", {"ref": "@e4"}, "c1").action == "approve"
    _click_ran(mgr, "@e4", "c1", "ok")  # manager approved, the click ran
    clock[0] += 60
    # the page re-renders with new refs; the retry is keyed by label, not ref
    _feed(mgr, SNAPSHOT_AB026.replace("ref=e4", "ref=e44"), "https://www.selenium.dev/selenium/web/web-form.html")
    assert _directive(pmod, "browser_click", {"ref": "@e44"}, "c2").action is None
    assert _directive(pmod, "browser_click", {"ref": "@e99"}, "c3").action == "approve"  # unknown still asks

    clock[0] += 61  # 121s after the approval
    assert _directive(pmod, "browser_click", {"ref": "@e44"}, "c4").action == "approve"


def test_denied_or_other_domain_click_is_not_remembered(loaded_plugin, monkeypatch):
    pmod, mgr, module = loaded_plugin
    monkeypatch.setattr(module, "_now", lambda: 1000.0)
    _feed(mgr, SNAPSHOT_AB026, "https://a.example.sa/form")
    assert _directive(pmod, "browser_click", {"ref": "@e4"}, "c1").action == "approve"
    _click_ran(mgr, "@e4", "c1", "blocked")  # manager tapped Deny
    assert _directive(pmod, "browser_click", {"ref": "@e4"}, "c2").action == "approve"

    _click_ran(mgr, "@e4", "c2", "ok")  # approved on a.example.sa
    _feed(mgr, SNAPSHOT_AB026, "https://b.example.sa/form")
    assert _directive(pmod, "browser_click", {"ref": "@e4"}, "c3").action == "approve"


def test_managed_config_locks_down_telegram_but_keeps_browser_and_tact_guard(tmp_path, monkeypatch):
    """tact/managed-config.yaml, baked to /etc/hermes/config.yaml, enables tact-guard and strips
    every code/file/account toolset from the Telegram bot while browser, memory and cron remain."""
    from gateway.run import _load_gateway_config
    from hermes_cli import plugins as pmod
    from hermes_cli.managed_scope import invalidate_managed_cache
    from hermes_cli.tools_config import _get_platform_tools
    from model_tools import _select_tool_names

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
        assert "tact-guard" in cfg["plugins"]["enabled"]
        assert cfg["model"]["default"] == "some/model"  # user config still underneath

        tools = _select_tool_names(sorted(_get_platform_tools(cfg, "telegram")),
                                   cfg["agent"]["disabled_toolsets"], quiet_mode=True)
        assert not tools & {"terminal", "read_file", "write_file", "execute_code", "manage_connections"}
        assert {"browser_navigate", "browser_click", "browser_type", "memory", "cronjob_manage",
                "web_search", "session_search"} <= tools

        mgr = pmod.PluginManager()
        mgr.discover_and_load()
        loaded = mgr._plugins["tact-guard"]
        assert loaded.enabled, loaded.error
        assert {"pre_tool_call", "post_tool_call"} <= set(loaded.hooks_registered)
    finally:
        invalidate_managed_cache()
