"""Sending each confirmed task to the person it is assigned to — only after the manager's ✅.

After a meeting is confirmed the manager gets [📤 Send tasks]. That shows a preview (``plan``):
per registered member the tasks they will get on Telegram, and every other task with why it is
skipped (unclear owner, "⏳ not joined yet", or a name that fits several members, which the manager
resolves with the preview's buttons). Only the [✅ Send] under that preview sends. Each member gets
one message with only their own confirmed tasks; a task with joint owners goes to each of them;
unclear and cancelled tasks never go out. Sent tasks get ``sent_at`` / ``sent_via``; nothing is
ever re-sent on its own, and a second [📤] warns that those tasks were already sent.

People are reached only as members who joined through /invite (``contacts.py``). The SMTP email
path below (``email_ready`` / ``_send_email``) is kept dormant: no route produces an email target
any more. Message contents, tokens and passwords are never logged.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any, Dict, List, Optional, Tuple

from . import brief as fmt
from . import contacts
from . import store

logger = logging.getLogger(__name__)

SMTP_VARS = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM")

TEXT = {
    "ar": {"channel_telegram": "تيليجرام", "channel_email": "البريد الإلكتروني",
           "unclear": "المسؤول غير واضح", "not_joined": "⏳ لم ينضم بعد", "self": "مهمتك — لا تُرسل",
           "resent": "📤 أُعيد إرسال المهمة {n} إلى {who}.", "resend_ask": "📤 إعادة إرسال المهمة {n} بعد التعديل؟",
           "btn_resend": "📤 إعادة إرسال لهذا الشخص",
           "ambiguous": "❓ أكثر من شخص بهذا الاسم: {choices} — اختر من الأزرار",
           "preview": "📤 معاينة الإرسال — اجتماع {id}", "skipped": "⏭️ لن تُرسل:",
           "resend": "⚠️ {n} من هذه المهام أُرسلت من قبل وسيُعاد إرسالها إذا ضغطت ✅.",
           "nothing": "لا توجد مهام يمكن إرسالها الآن.",
           "fix_hint": "أرسل /invite لمن لم ينضم بعد، أو اربط المهمة بعضو: ✏️ ثم 📞 التواصل.",
           "report": "📤 نتيجة الإرسال — اجتماع {id}", "sent_to": "✅ {name} — {channel}: {n} مهام",
           "failed": "⚠️ فشل الإرسال إلى {name} ({channel})", "cancelled": "لم يُرسل شيء.",
           "header": "📋 مهامك من اجتماع {date} مع {manager}:", "line": "{i}. {task} — الموعد: {deadline}",
           "manager": "المدير", "subject": "مهامك من اجتماع {date}",
           "ask": "إرسال كل مهمة مؤكدة إلى صاحبها؟ ستظهر معاينة أولاً.", "btn": "📤 إرسال المهام",
           "btn_send": "✅ إرسال", "btn_cancel": "❌ إلغاء"},
    "en": {"channel_telegram": "Telegram", "channel_email": "email",
           "unclear": "owner unclear", "not_joined": "⏳ not joined yet", "self": "your task — not sent",
           "resent": "📤 Task {n} was sent again to {who}.", "resend_ask": "📤 Send task {n} again after the edit?",
           "btn_resend": "📤 Send again to this person",
           "ambiguous": "❓ more than one person with this name: {choices} — choose with the buttons",
           "preview": "📤 Sending preview — Meeting {id}", "skipped": "⏭️ Will not be sent:",
           "resend": "⚠️ {n} of these tasks were sent before and will be sent again if you tap ✅.",
           "nothing": "There are no tasks that can be sent now.",
           "fix_hint": "Send /invite to anyone who hasn't joined, or link the task to a member: ✏️ then 📞 Contact.",
           "report": "📤 Sending result — Meeting {id}", "sent_to": "✅ {name} — {channel}: {n} task(s)",
           "failed": "⚠️ Sending to {name} ({channel}) failed", "cancelled": "Nothing was sent.",
           "header": "📋 Your tasks from the meeting on {date} with {manager}:",
           "line": "{i}. {task} — Deadline: {deadline}", "manager": "the manager",
           "subject": "Your tasks from the meeting on {date}",
           "ask": "Send each confirmed task to its owner? You'll see a preview first.", "btn": "📤 Send tasks",
           "btn_send": "✅ Send", "btn_cancel": "❌ Cancel"},
}


def _secret(name: str) -> str:
    from agent.secret_scope import get_secret
    return (get_secret(name, "") or "").strip()


def email_ready() -> bool:
    return all(_secret(name) for name in SMTP_VARS)


def plan(meeting_id: int, positions: Optional[List[int]] = None) -> Dict[str, Any]:
    """Who gets which confirmed tasks and how; which tasks are skipped and why (only *positions*,
    when given). Sends nothing. The manager's own tasks are never sent."""
    recipients: Dict[Tuple[str, str], Dict[str, Any]] = {}
    skipped: List[Tuple[Dict[str, Any], str, str]] = []
    choices: List[Tuple[Dict[str, Any], tuple]] = []
    tasks = [t for t in store.tasks_for(meeting_id) if t["status"] == "confirmed"
             and (positions is None or t["position"] in positions)]
    for task in tasks:
        names = fmt.split_owners(task["person"])
        if not names:
            skipped.append((task, "", "unclear"))
            continue
        for name in names:
            if fmt.is_manager(name):
                skipped.append((task, name, "self"))
                continue
            # The task's own contact belongs to its single owner; joint owners use the book.
            single = len(names) == 1
            route = contacts.route(name, task.get("contact_id") if single else None)
            if not route.channel:
                skipped.append((task, route.who or name, route.reason))
                if route.choices:  # a shared first name: the manager picks, nothing is guessed
                    choices.append((task, route.choices))
                continue
            entry = recipients.setdefault((route.channel, route.target),
                                          {"name": route.who or name, "channel": route.channel,
                                           "target": route.target, "tasks": []})
            entry["tasks"].append(task)
    already = sum(1 for t in tasks if t["sent_at"] and any(t in r["tasks"] for r in recipients.values()))
    return {"recipients": list(recipients.values()), "skipped": skipped, "already_sent": already,
            "choices": choices}


def _reason(text: Dict[str, str], p: Dict[str, Any], task: Dict[str, Any], reason: str) -> str:
    if reason != "ambiguous":
        return text[reason]
    rows = next((rows for t, rows in p.get("choices", []) if t["position"] == task["position"]), ())
    return text["ambiguous"].format(choices=" / ".join(contacts.label(r) for r in rows))


def preview_text(meeting_id: int, lang: str, p: Dict[str, Any]) -> str:
    text = TEXT[lang]
    lines = [text["preview"].format(id=meeting_id)]
    for r in p["recipients"]:
        where = text[f"channel_{r['channel']}"] + (f" ({r['target']})" if r["channel"] == "email" else "")
        lines.append(f"\n👤 {r['name']} — {where}")
        lines += [f"  • {t['position']}. {t['task']}" for t in r["tasks"]]
    if p["skipped"]:
        lines.append(f"\n{text['skipped']}")
        lines += [f"  • {t['position']}. {name or fmt.owner_text(t, lang, question=False)} — "
                  f"{_reason(text, p, t, reason)}" for t, name, reason in p["skipped"]]
        lines.append(text["fix_hint"])
    if not p["recipients"]:
        lines.append(f"\n{text['nothing']}")
    elif p["already_sent"]:
        lines.append(f"\n{text['resend'].format(n=p['already_sent'])}")
    return "\n".join(lines)


def message_text(meeting: Any, lang: str, tasks: List[Dict[str, Any]], who: str = "") -> str:
    """The member's message; *who* is how they are known ("full name (role)")."""
    text = TEXT[lang]
    header = text["header"].format(date=meeting["created_at"][:10], manager=meeting["manager_name"] or text["manager"])
    missing = fmt.LABELS[lang]["missing"]
    return (f"👤 {who}\n" if who else "") + header + "\n" + "\n".join(
        text["line"].format(i=i, task=t["task"], deadline=t["deadline"] or missing) for i, t in enumerate(tasks, 1))


def _send_email(to: str, subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = _secret("SMTP_FROM"), to, subject
    msg.set_content(body)
    with smtplib.SMTP(_secret("SMTP_HOST"), int(_secret("SMTP_PORT")), timeout=30) as smtp:
        smtp.starttls(context=ssl.create_default_context())
        smtp.login(_secret("SMTP_USER"), _secret("SMTP_PASSWORD"))
        smtp.send_message(msg)


async def execute(bot: Any, meeting: Any, lang: str, p: Dict[str, Any]) -> Tuple[str, List[int]]:
    """Send the plan; ``(report for the manager, task numbers that were sent)``."""
    text = TEXT[lang]
    report = [text["report"].format(id=meeting["id"])]
    channels: Dict[int, List[str]] = {}
    sent = 0
    for r in p["recipients"]:
        body = message_text(meeting, lang, r["tasks"], r["name"])
        channel_name = text[f"channel_{r['channel']}"]
        try:
            if r["channel"] == "telegram":
                chat = int(r["target"]) if str(r["target"]).lstrip("-").isdigit() else r["target"]
                await bot.send_message(chat_id=chat, text=body)
            else:
                await asyncio.to_thread(_send_email, r["target"],
                                        text["subject"].format(date=meeting["created_at"][:10]), body)
        except Exception as exc:
            logger.warning("tact-recorder: sending meeting #%s tasks by %s failed: %s",
                           meeting["id"], r["channel"], type(exc).__name__)
            report.append(text["failed"].format(name=r["name"], channel=channel_name))
            continue
        sent += 1
        report.append(text["sent_to"].format(name=r["name"], channel=channel_name, n=len(r["tasks"])))
        for task in r["tasks"]:
            channels.setdefault(task["position"], []).append(r["channel"])
    by_pos = {t["position"]: t for r in p["recipients"] for t in r["tasks"]}
    for position, via in channels.items():
        previous = [c for c in (by_pos[position]["sent_via"] or "").split(",") if c]
        store.mark_sent(meeting["id"], position, previous + via)
    if p["skipped"]:
        report.append(text["skipped"])
        report += [f"  • {t['position']}. {name or fmt.owner_text(t, lang, question=False)} — "
                   f"{_reason(text, p, t, reason)}" for t, name, reason in p["skipped"]]
    logger.info("tact-recorder: meeting #%s: %d of %d recipient(s) sent, %d task(s) skipped",
                meeting["id"], sent, len(p["recipients"]), len(p["skipped"]))
    return "\n".join(report), sorted(channels)


def markup(kind: str, meeting_id: int, lang: str, p: Optional[Dict[str, Any]] = None) -> Any:
    """[📤 Send tasks] ("ask"), or the preview's buttons ("confirm"): one per possible person for a
    name that fits several, then [✅ Send] [❌ Cancel] when anyone can be sent to."""
    from telegram import InlineKeyboardButton as Button, InlineKeyboardMarkup
    text = TEXT[lang]
    if kind == "ask":
        return InlineKeyboardMarkup([[Button(text["btn"], callback_data=f"rec:x:p:{meeting_id}")]])
    rows = [[Button(f"{task['position']} ← {contacts.label(row)}",
                    callback_data=f"rec:w:a:{meeting_id}:{task['position']}:{row['id']}")]
            for task, candidates in (p or {}).get("choices", []) for row in candidates]
    if not p or p["recipients"]:
        rows.append([Button(text["btn_send"], callback_data=f"rec:x:s:{meeting_id}"),
                     Button(text["btn_cancel"], callback_data=f"rec:x:c:{meeting_id}")])
    return InlineKeyboardMarkup(rows) if rows else None


def has_sendable(meeting_id: int) -> bool:
    return any(t["status"] == "confirmed" and any(not fmt.is_manager(n) for n in fmt.split_owners(t["person"]))
               for t in store.tasks_for(meeting_id))


def resend_markup(meeting_id: int, position: int, lang: str) -> Any:
    from telegram import InlineKeyboardButton as Button, InlineKeyboardMarkup
    return InlineKeyboardMarkup([[Button(TEXT[lang]["btn_resend"], callback_data=f"rec:x:r:{meeting_id}:{position}")]])


def resend_ask_text(position: int, lang: str) -> str:
    return TEXT[lang]["resend_ask"].format(n=position)


def cancelled_text(lang: str) -> str:
    return TEXT[lang]["cancelled"]


def ask_text(lang: str) -> str:
    return TEXT[lang]["ask"]


def resolve_meeting(meeting_id: int, platform: str, chat_id: str, user_id: str) -> Optional[Any]:
    """The meeting if this user (its manager) may send its tasks from this chat."""
    meeting = store.get_meeting(meeting_id, platform, chat_id)
    return meeting if meeting is not None and meeting["user_id"] == str(user_id) else None
