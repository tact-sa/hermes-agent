"""Buttons behind one small seam, so panels never import a messaging library.

A panel sends or edits a message with *rows*: lists of ``(label, callback_id)``. Only this module
knows how a platform draws them; today that is Telegram inline keyboards. Limits are chosen so a
second platform (WhatsApp reply buttons / list rows) can take the same rows: callback ids stay
within ``MAX_CALLBACK`` characters and labels are cut to ``MAX_LABEL`` unless a caller allows more.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

from . import present

logger = logging.getLogger(__name__)

MAX_CALLBACK = 200
MAX_LABEL = 20
Row = Sequence[Tuple[str, str]]


def short(text: str, limit: int = MAX_LABEL) -> str:
    """*text* cut to *limit* characters with a trailing "…"."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:max(limit - 1, 1)].rstrip() + "…"


def _checked(rows: Sequence[Row], label_limit: int) -> List[List[Tuple[str, str]]]:
    out = []
    for row in rows:
        for _, callback_id in row:
            if len(callback_id) > MAX_CALLBACK:
                raise ValueError(f"callback id longer than {MAX_CALLBACK}: {callback_id[:30]}…")
        out.append([(short(label, label_limit), callback_id) for label, callback_id in row])
    return out


def has_buttons(adapter: Any, platform: str) -> bool:
    return platform == "telegram" and getattr(adapter, "_bot", None) is not None


def _telegram_markup(rows: List[List[Tuple[str, str]]]) -> Any:
    from telegram import InlineKeyboardButton as Button, InlineKeyboardMarkup
    return InlineKeyboardMarkup([[Button(label, callback_data=callback_id) for label, callback_id in row]
                                 for row in rows])


def is_manager(adapter: Any, user_id: str, chat_id: str) -> bool:
    """Whether the person who tapped is on the platform's allowlist (the manager); false when unknown."""
    check = getattr(adapter, "_is_callback_user_authorized", None)
    if not callable(check):
        return False
    try:
        return bool(check(str(user_id), chat_id=str(chat_id), chat_type="dm"))
    except Exception as exc:
        logger.debug("tact-recorder: manager check failed: %s", type(exc).__name__)
        return False


async def send(adapter: Any, chat_id: str, text: str, rows: Sequence[Row] = (),
               label_limit: int = MAX_LABEL) -> Optional[str]:
    """Send *text* with button *rows*; the new message's id (None where it can't be known)."""
    bot = getattr(adapter, "_bot", None)
    if bot is None or not rows:
        await present.send(adapter, chat_id, text)
        return None
    msg = await bot.send_message(chat_id=present._chat_arg(chat_id), text=text,
                                 reply_markup=_telegram_markup(_checked(rows, label_limit)))
    return str(msg.message_id)


async def edit(adapter: Any, chat_id: str, message_id: Any, text: str, rows: Sequence[Row] = (),
               label_limit: int = MAX_LABEL) -> bool:
    """Replace one of our messages in place (no rows removes its buttons); False when it can't be."""
    markup = _telegram_markup(_checked(rows, label_limit)) if rows else None
    return await present._edit(getattr(adapter, "_bot", None), chat_id, message_id, text, markup)
