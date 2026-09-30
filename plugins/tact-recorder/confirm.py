"""Task confirmation: one ``apply`` for both the Telegram buttons and the text replies.

Text replies (English or Arabic, Arabic-Indic digits accepted) act on the chat's latest meeting
that still has tasks awaiting confirmation: ``confirm all``, ``confirm 2``, ``remove 3`` and
``edit 2: deadline: Thursday`` (explicit labels, see ``parse_labelled``). Tasks are always referred to by their number in the meeting. When no
task is left pending the meeting is marked confirmed and the final list, grouped by person, goes
back to the manager. Nothing is sent to anyone else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import brief as fmt
from . import store

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_CONFIRM = r"(?:confirm|approve|تأكيد|تاكيد|أكد|اكد|أكّد|اعتماد|اعتمد)"
_ALL_RE = re.compile(rf"^{_CONFIRM}\s+(?:all|everything|الكل|الجميع)$", re.IGNORECASE)
_ONE_RE = re.compile(rf"^{_CONFIRM}\s*#?(\d+)$", re.IGNORECASE)
# "حذف" (the old wording) stays accepted next to "إلغاء".
_CANCEL_RE = re.compile(r"^(?:cancel|remove|delete|إلغاء|الغاء|ألغ|الغ|حذف|احذف|إزالة|ازالة)\s*#?(\d+)$",
                        re.IGNORECASE)
_EDIT_RE = re.compile(r"^(?:edit|change|تعديل|عدل|عدّل)\s*#?(\d+)\s*(?:[:：\-–]\s*(.*))?$",
                      re.IGNORECASE | re.DOTALL)

FIELDS = ("person", "task", "deadline")
# Explicit labels only, each followed by ":" at the start of the text or after a separator. An
# Arabic label may carry the conjunction "و" ("والموعد:").
_LABELS = {
    "person": r"الاسم|الإسم|اسم|الشخص|المسؤول|name|owner|person",
    "task": r"المهمة|المهمه|مهمة|task",
    "deadline": r"الموعد|موعد|التاريخ|deadline|due|date",
}
_LABEL_RE = re.compile(
    r"(?:^|(?<=[,،;؛\n]))\s*(?:و\s*)?(?:(?P<person>" + _LABELS["person"] + r")|(?P<task>" + _LABELS["task"]
    + r")|(?P<deadline>" + _LABELS["deadline"] + r"))\s*[:：]", re.IGNORECASE)

ASK_FIELD = {
    "en": ('Which field of task {n}? Tap 👤 Name, 📌 Task or 📅 Deadline, or reply e.g. '
           '"edit {n}: deadline: Thursday" (fields: name, task, deadline).'),
    "ar": ('أي حقل تريد تعديله في المهمة {n}؟ اضغط 👤 الاسم أو 📌 المهمة أو 📅 الموعد، أو اكتب مثلاً: '
           '"تعديل {n}: الموعد: الخميس" (الحقول: الاسم، المهمة، الموعد).'),
}
FIELD_PROMPT = {
    "en": {"person": "Send the new name for task {n}", "task": "Send the new task text for task {n}",
           "deadline": "Send the new deadline for task {n}"},
    "ar": {"person": "أرسل الاسم الجديد للمهمة {n}", "task": "أرسل المهمة الجديدة للمهمة {n}",
           "deadline": "أرسل الموعد الجديد للمهمة {n}"},
}


def parse_command(text: str) -> Optional[Tuple[str, int, str]]:
    """``(action, number, edit_text)`` for a confirmation reply; action in all/confirm/remove/edit."""
    text = (text or "").translate(_DIGITS).strip()
    if _ALL_RE.match(text):
        return "all", 0, ""
    for action, regex in (("confirm", _ONE_RE), ("remove", _CANCEL_RE)):
        m = regex.match(text)
        if m:
            return action, int(m.group(1)), ""
    m = _EDIT_RE.match(text)
    if m:
        return "edit", int(m.group(1)), (m.group(2) or "").strip()
    return None


def parse_labelled(text: str) -> Dict[str, str]:
    """Fields named by an explicit label, any order: ``"الاسم: عمر، الموعد: الخميس"`` -> person +
    deadline. Each value is everything up to the next label, trimmed of separators. Text without a
    label gives ``{}``: the manager is then asked which field, never guessed for."""
    text = text or ""
    matches = list(_LABEL_RE.finditer(text))
    if not matches or text[:matches[0].start()].strip(" \t\n,،;؛"):
        return {}
    fields: Dict[str, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = text[m.end():end].strip().strip(",،;؛").strip()
        if value:
            fields[m.lastgroup] = value
    return fields


@dataclass
class Outcome:
    """What one confirmation action changed; ``present.show_changes`` turns it into Telegram edits
    (or text messages where a task has no card to edit)."""
    meeting_id: int
    messages: List[str] = field(default_factory=list)  # always sent: errors, edit instructions
    changed: List[int] = field(default_factory=list)  # task numbers whose card must be refreshed
    acks: Dict[int, str] = field(default_factory=dict)  # text used only when that card can't be edited
    finished: bool = False  # nothing pending any more: the final summary follows
    picker: int = 0  # task number whose card shows the Name / Task / Deadline choice instead


def _meeting_lang(meeting: Any) -> str:
    return fmt.lang_of(meeting["language"])


def _update_fields(meeting_id: int, position: int, fields: Dict[str, str]) -> None:
    """Store the new values exactly as given; a name makes an unclear task that person's."""
    values: Dict[str, Any] = dict(fields)
    if "person" in values:
        values["person_key"] = fmt.name_key(values["person"])
        values["candidates"] = "[]"
    store.update_task(meeting_id, position, **values)


def _edit_ack(meeting_id: int, position: int, lang: str) -> str:
    tasks = store.tasks_for(meeting_id)
    task = next(t for t in tasks if t["position"] == position)
    hint = ("أكّدها بـ ✅ أو اكتب \"تأكيد {}\"." if lang == "ar"
            else 'Confirm it with ✅ or reply "confirm {}".').format(position)
    return f"✏️ {fmt.task_card(task, len(tasks), lang)}\n\n{hint}"


def set_field(meeting: Any, position: int, field_name: str, value: str) -> Outcome:
    """The manager's reply after choosing a field: it replaces only that field, as typed."""
    lang = _meeting_lang(meeting)
    out = Outcome(meeting["id"])
    value = (value or "").strip()
    if meeting["status"] != "pending":
        out.messages.append(("الاجتماع #{} مؤكد بالفعل." if lang == "ar" else "Meeting #{} is already confirmed.")
                            .format(meeting["id"]))
        return out
    if not value:
        out.messages.append(FIELD_PROMPT[lang][field_name].format(n=position))
        return out
    _update_fields(meeting["id"], position, {field_name: value})
    out.changed = [position]
    out.acks[position] = _edit_ack(meeting["id"], position, lang)
    return out


def apply(meeting: Any, action: str, position: int = 0, edit_text: str = "") -> Outcome:
    """Apply one confirmation action to *meeting* (a store row)."""
    lang = _meeting_lang(meeting)
    meeting_id = meeting["id"]
    out = Outcome(meeting_id)
    if meeting["status"] != "pending":
        out.messages.append(("الاجتماع #{} مؤكد بالفعل." if lang == "ar" else "Meeting #{} is already confirmed.")
                            .format(meeting_id))
        return out
    tasks = store.tasks_for(meeting_id)
    by_pos = {t["position"]: t for t in tasks}
    if action != "all" and position not in by_pos:
        out.messages.append(("لا توجد مهمة رقم {} في الاجتماع #{}." if lang == "ar"
                             else "There is no task {} in meeting #{}.").format(position, meeting_id))
        return out

    if action == "all":
        out.changed = [t["position"] for t in tasks if t["status"] == "pending"]
        store.set_task_status(meeting_id, out.changed, "confirmed")
    elif action in ("confirm", "remove"):
        store.set_task_status(meeting_id, [position], "confirmed" if action == "confirm" else "removed")
        out.changed = [position]
        out.acks[position] = (("✅ تم تأكيد المهمة {}." if lang == "ar" else "✅ Task {} confirmed.")
                              if action == "confirm" else
                              ("❌ تم إلغاء المهمة {}." if lang == "ar" else "❌ Task {} cancelled.")).format(position)
    elif action == "edit":
        fields = parse_labelled(edit_text)
        if not fields:
            out.messages.append(ASK_FIELD[lang].format(n=position))
            out.changed, out.picker = [position], position
            return out
        _update_fields(meeting_id, position, fields)
        out.changed = [position]
        out.acks[position] = _edit_ack(meeting_id, position, lang)
    else:
        return out

    pending = [t for t in store.tasks_for(meeting_id) if t["status"] == "pending"]
    if pending:
        if action in ("confirm", "remove"):
            out.acks[position] += (f" (متبقٍ {len(pending)} للتأكيد)" if lang == "ar"
                                   else f" ({len(pending)} still to confirm)")
        return out
    store.update_meeting(meeting_id, status="confirmed")
    out.finished = True
    return out
