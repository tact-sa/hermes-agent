"""Plugin approval rules (tact-guard booking clicks) render a plain Confirm/Cancel card.

``tools.approval.request_tool_approval`` marks its pending approval ``plugin_rule``; the runner
forwards that flag to the adapter, which renders "Please confirm" + the description + two buttons.
A dangerous-command approval (no flag) keeps the warning card with the command block and the
four-tier buttons. Driven through the real ``TurnRunner._approval_notify_sync`` into a real
``TelegramAdapter`` with a mocked bot.
"""

from __future__ import annotations

import asyncio
import html
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter

SESSION = "agent:main:telegram:dm:1"


def _adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
    adapter._app = MagicMock()
    return adapter


def _runner(adapter):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="12345", _status_thread_metadata=None,
        session_key=SESSION, source=SimpleNamespace(chat_id="12345", platform="telegram", session_key=SESSION),
    )

    class _Fut:
        def __init__(self, result): self._r = result
        def result(self, timeout=None): return self._r

    runner._schedule = lambda coro, _label: _Fut(asyncio.run(coro))
    runner._close_native_stream_boundary = lambda _why: None
    return runner


@pytest.fixture
def card(monkeypatch):
    """Send ``approval_data`` through the runner; return (text, button labels, adapter)."""
    monkeypatch.setattr("gateway.platforms.base_exec_approval.approval_timeout_seconds", lambda: 300)
    labels = []
    monkeypatch.setattr("plugins.platforms.telegram.adapter.InlineKeyboardButton",
                        lambda text, callback_data: labels.append(text) or text)
    monkeypatch.setattr("plugins.platforms.telegram.adapter.InlineKeyboardMarkup", lambda rows: rows)

    def send(approval_data):
        adapter = _adapter()
        _runner(adapter)._approval_notify_sync(dict(approval_data))
        return adapter._bot.send_message.call_args[1]["text"], list(labels), adapter
    return send


def test_plugin_approval_renders_plain_confirm_card(card):
    text, labels, _ = card({
        "command": "<browser_click> (plugin approval rule)", "pattern_key": "plugin_rule:tact-guard:click:x",
        "description": 'Click "Reserve Now" on automationintesting.online', "plugin_rule": True,
        "allow_permanent": True, "allow_session": True,
    })
    assert html.unescape(text) == ("🔐 <b>Please confirm</b>\n\n"
                                   'Click "Reserve Now" on automationintesting.online\n\n'
                                   "If you don't answer within 5 minutes, nothing will be done.")
    assert labels == ["✅ Confirm", "❌ Cancel"]


def test_dangerous_command_keeps_warning_card(card):
    text, labels, _ = card({
        "command": "rm -rf /tmp/x", "pattern_key": "k", "description": "recursive delete",
        "allow_permanent": True, "allow_session": True,
    })
    assert text.startswith("⚠️ <b>Hermes wants to run a command that needs your OK</b>")
    assert "<pre>rm -rf /tmp/x</pre>" in text and "Why it was flagged" in text
    assert "it will NOT run." in text
    assert labels == ["✅ Allow Once", "✅ Session", "✅ Always", "❌ Deny"]


@pytest.mark.asyncio
@pytest.mark.parametrize("choice,resolved", [("once", "✅ Confirmed"), ("deny", "❌ Cancelled")])
async def test_plugin_card_resolves_as_confirmed_or_cancelled(choice, resolved, monkeypatch):
    monkeypatch.setattr("gateway.platforms.base_exec_approval.approval_timeout_seconds", lambda: 300)
    adapter = _adapter()
    await adapter.send_exec_approval(chat_id="12345", command="<browser_click> (plugin approval rule)",
                                     session_key=SESSION, description='Click "Book" on x.sa', plugin_rule=True)
    approval_id = next(iter(adapter._approval_state))

    query = AsyncMock()
    query.data = f"ea:{choice}:{approval_id}"
    query.message = MagicMock(chat_id=12345)
    query.from_user = MagicMock(first_name="Manager", id="12345")
    update = MagicMock(callback_query=query)
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        with patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve:
            await adapter._handle_callback_query(update, MagicMock())

    resolve.assert_called_once_with(SESSION, choice)
    assert query.edit_message_text.call_args[1]["text"] == resolved.replace("!", "\\!")
    assert query.answer.call_args[1]["text"] == resolved
