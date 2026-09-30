"""tact-clock — the real current date and time on every turn.

Hermes puts only the DATE in the system prompt ("Conversation started: …"), kept byte-stable so the
prompt cache holds for the whole day, and tells the model to read the exact time from the terminal.
The Tact bot has no terminal (its toolsets are disabled), so asked "كم الساعة؟" it guessed.

``pre_llm_call`` context is appended to the current turn's user message only (never the system
prompt), and the gateway replays each past turn with the bytes it was sent with. So every turn
carries the time it was actually asked at, earlier turns keep theirs, and the cached prefix is
untouched: the cost is one short line per turn.
"""

from __future__ import annotations

from typing import Any, Dict


def clock_line() -> str:
    from hermes_time import get_timezone_name, now, safe_strftime
    current = now()
    offset = safe_strftime(current, "%z")
    zone = ", ".join(p for p in (get_timezone_name(), f"UTC{offset[:3]}:{offset[3:]}" if offset else "") if p)
    return (f"[Current date and time: {safe_strftime(current, '%A, %d %B %Y, %H:%M')}"
            f"{f' ({zone})' if zone else ''}. Use this for the time, today's date and any relative date "
            "(tomorrow, next week, in 2 hours); never guess the time.]")


def _on_pre_llm_call(**_: Any) -> Dict[str, str]:
    return {"context": clock_line()}


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
