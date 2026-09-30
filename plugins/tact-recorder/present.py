"""Showing a meeting in the chat and keeping it up to date.

On Telegram each task is its own message with ✅ / ✏️ / ❌ buttons, followed by one "Confirm all (n)"
message; ✏️ swaps a card's buttons for Name / Task / Deadline / Back. The Telegram message id of every card (and of the "Confirm all" message) is stored, so a
confirmation — by button or by text reply — edits THAT card in place: its status appears under it
and its buttons go (confirmed / removed) or stay (edited). Above ``MAX_CARD_TASKS`` tasks, or where
there are no inline buttons, all tasks go in one message and text replies do the confirming.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from . import brief as fmt
from . import store
from .confirm import Outcome

logger = logging.getLogger(__name__)

MAX_CARD_TASKS = 15


async def send(adapter: Any, chat_id: str, text: str) -> None:
    if adapter is None:
        logger.warning("tact-recorder: no adapter to reply on")
        return
    result = await adapter.send(chat_id, text)
    if result is not None and not getattr(result, "success", True):
        logger.warning("tact-recorder: reply failed: %s", getattr(result, "error", "unknown"))


def _bot(adapter: Any, platform: str) -> Any:
    return getattr(adapter, "_bot", None) if platform == "telegram" else None


def _chat_arg(chat_id: str) -> Any:
    return int(chat_id) if str(chat_id).lstrip("-").isdigit() else chat_id


def _card_markup(meeting_id: int, task: Dict[str, Any], lang: str) -> Any:
    if task["status"] != "pending":
        return None
    from telegram import InlineKeyboardButton as Button, InlineKeyboardMarkup
    labels = fmt.LABELS[lang]
    return InlineKeyboardMarkup([[
        Button(labels[label], callback_data=f"rec:{code}:{meeting_id}:{task['position']}")
        for label, code in (("btn_confirm", "c"), ("btn_edit", "e"), ("btn_remove", "r"))]])


def _picker_markup(meeting_id: int, position: int, lang: str) -> Any:
    """✏️ tapped: which field to change, or back to the normal buttons."""
    from telegram import InlineKeyboardButton as Button, InlineKeyboardMarkup
    labels = fmt.LABELS[lang]
    return InlineKeyboardMarkup([
        [Button(labels[label], callback_data=f"rec:f:{meeting_id}:{position}:{code}")
         for label, code in (("btn_name", "n"), ("btn_task", "t"), ("btn_deadline", "d"))],
        [Button(labels["btn_back"], callback_data=f"rec:b:{meeting_id}:{position}")]])


def _all_message(meeting_id: int, pending: int, lang: str) -> tuple:
    labels = fmt.LABELS[lang]
    if not pending:
        return labels["all_done"], None
    from telegram import InlineKeyboardButton as Button, InlineKeyboardMarkup
    markup = InlineKeyboardMarkup([[Button(labels["confirm_all"].format(n=pending),
                                           callback_data=f"rec:a:{meeting_id}:0")]])
    return f"{labels['all_prompt']}\n{labels['text_help']}", markup


async def _edit(bot: Any, chat_id: str, message_id: Any, text: str, markup: Any) -> bool:
    """Edit one of our messages in place; False when it can't be (no bot, no id, Telegram error)."""
    if bot is None or not message_id:
        return False
    try:
        # No reply_markup removes the buttons.
        await bot.edit_message_text(chat_id=_chat_arg(chat_id), message_id=int(message_id), text=text,
                                    reply_markup=markup)
        return True
    except Exception as exc:
        if "not modified" in str(exc).lower():  # already shows this text and buttons
            return True
        logger.warning("tact-recorder: editing a task message failed: %s", type(exc).__name__)
        return False


async def show_meeting(adapter: Any, platform: str, chat_id: str, meeting_id: int) -> None:
    """The brief; then any handled tasks in the summary layout; then the pending ones, one card each
    with buttons (or one combined message)."""
    meeting = store.get_meeting(meeting_id, platform, chat_id)
    tasks = store.tasks_for(meeting_id)
    pending = [t for t in tasks if t["status"] == "pending"]
    lang = fmt.lang_of(meeting["language"])
    bot = _bot(adapter, platform)
    cards = bot is not None and 0 < len(pending) <= MAX_CARD_TASKS
    several = cards or 0 < len(pending) < len(tasks)  # "in the following messages" vs "in the next message"
    await send(adapter, chat_id, fmt.brief_text(meeting_id, meeting["created_at"], lang,
                                                json.loads(meeting["brief"]), len(tasks), several))
    if len(pending) < len(tasks):
        await send(adapter, chat_id, fmt.final_text(meeting_id, lang, tasks))
    if not pending:
        return
    if cards:
        try:
            for task in pending:
                msg = await bot.send_message(chat_id=_chat_arg(chat_id), text=fmt.task_card(task, len(tasks), lang),
                                             reply_markup=_card_markup(meeting_id, task, lang))
                store.set_task_message(meeting_id, task["position"], msg.message_id)
            text, markup = _all_message(meeting_id, len(pending), lang)
            msg = await bot.send_message(chat_id=_chat_arg(chat_id), text=text, reply_markup=markup)
            store.update_meeting(meeting_id, confirm_message_id=str(msg.message_id))
            return
        except Exception as exc:
            logger.warning("tact-recorder: task cards failed, sending one list: %s", type(exc).__name__)
    await send(adapter, chat_id, fmt.combined_text(pending, len(tasks), lang))


async def show_changes(adapter: Any, platform: str, chat_id: str, outcome: Outcome,
                       pressed: Optional[Dict[int, Any]] = None) -> None:
    """Reflect *outcome*: edit the changed cards (and the "Confirm all" message) in place, falling back
    to a text message per task only when its card can't be edited; the final summary when done.
    ``pressed`` maps a task number (0 = "Confirm all") to the message whose button was tapped, in
    case that is an older copy than the stored card (the meeting was shown again with /brief)."""
    pressed = pressed or {}
    for text in outcome.messages:
        await send(adapter, chat_id, text)
    if not outcome.changed:
        return
    meeting = store.get_meeting(outcome.meeting_id, platform, chat_id)
    tasks = store.tasks_for(outcome.meeting_id)
    by_pos = {t["position"]: t for t in tasks}
    lang = fmt.lang_of(meeting["language"])
    bot = _bot(adapter, platform)
    for position in outcome.changed:
        task = by_pos[position]
        message_ids = list(dict.fromkeys(m for m in (task["message_id"], pressed.get(position)) if m))
        edited = False
        if bot is not None and message_ids:
            text = fmt.task_card(task, len(tasks), lang)
            markup = (_picker_markup(outcome.meeting_id, position, lang) if position == outcome.picker
                      else _card_markup(outcome.meeting_id, task, lang))
            for message_id in message_ids:
                edited = await _edit(bot, chat_id, message_id, text, markup) or edited
        if not edited and position in outcome.acks:
            await send(adapter, chat_id, outcome.acks[position])
    all_ids = list(dict.fromkeys(m for m in (meeting["confirm_message_id"], pressed.get(0)) if m))
    if bot is not None and all_ids:
        text, markup = _all_message(outcome.meeting_id, sum(t["status"] == "pending" for t in tasks), lang)
        for message_id in all_ids:
            await _edit(bot, chat_id, message_id, text, markup)
    if outcome.finished:
        await send(adapter, chat_id, fmt.final_text(outcome.meeting_id, lang, tasks))
