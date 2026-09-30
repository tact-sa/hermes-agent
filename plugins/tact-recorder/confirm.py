"""Task confirmation: one ``apply`` for both the Telegram buttons and the text replies.

Text replies (English or Arabic, Arabic-Indic digits accepted) act on the chat's latest meeting
that still has tasks awaiting confirmation: ``confirm all``, ``confirm 2``, ``remove 3`` and
``edit 2: Omar, due Sunday``. When no task is left pending the meeting is marked confirmed and the
final list, grouped by person, goes back to the manager. Nothing is sent to anyone else.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from . import brief as fmt
from . import store

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_CONFIRM = r"(?:confirm|approve|تأكيد|تاكيد|أكد|اكد|أكّد|اعتماد|اعتمد)"
_ALL_RE = re.compile(rf"^{_CONFIRM}\s+(?:all|everything|الكل|الجميع)$", re.IGNORECASE)
_ONE_RE = re.compile(rf"^{_CONFIRM}\s*#?(\d+)$", re.IGNORECASE)
_REMOVE_RE = re.compile(r"^(?:remove|delete|حذف|احذف|إزالة|ازالة|الغاء|إلغاء)\s*#?(\d+)$", re.IGNORECASE)
_EDIT_RE = re.compile(r"^(?:edit|change|تعديل|عدل|عدّل)\s*#?(\d+)\s*[:：\-–]\s*(.+)$", re.IGNORECASE | re.DOTALL)

_FIELD_RES = (
    ("deadline", re.compile(r"^(?:due|deadline|by|الموعد|موعد|بحلول)\s*[:：]?\s*(.+)$", re.IGNORECASE | re.DOTALL)),
    ("task", re.compile(r"^(?:task|المهمة|مهمة)\s*[:：]?\s*(.+)$", re.IGNORECASE | re.DOTALL)),
    ("person", re.compile(r"^(?:person|owner|assignee|who|المسؤول|الشخص|المكلف)\s*[:：]?\s*(.+)$",
                          re.IGNORECASE | re.DOTALL)),
)
_NO_DEADLINE = {"none", "no deadline", "not mentioned", "لا يوجد", "بدون", "غير مذكور"}

EDIT_HELP = {
    "en": ('Send the change for task {n}, e.g. "Omar, due Sunday" or "task: send the budget, due Thursday".'),
    "ar": ('أرسل التعديل للمهمة {n}، مثال: "عمر، الموعد الأحد" أو "المهمة: إرسال الميزانية، الموعد الخميس".'),
}


def parse_command(text: str) -> Optional[Tuple[str, int, str]]:
    """``(action, position, edit_text)`` for a confirmation reply; action in all/confirm/remove/edit."""
    text = (text or "").translate(_DIGITS).strip()
    if _ALL_RE.match(text):
        return "all", 0, ""
    for action, regex in (("confirm", _ONE_RE), ("remove", _REMOVE_RE)):
        m = regex.match(text)
        if m:
            return action, int(m.group(1)), ""
    m = _EDIT_RE.match(text)
    if m:
        return "edit", int(m.group(1)), m.group(2).strip()
    return None


def parse_edit(text: str) -> Dict[str, str]:
    """Fields the manager changed. ``"Omar, due Sunday"`` -> person + deadline; a short leading part
    without a keyword is the person, anything longer is the task."""
    fields: Dict[str, str] = {}
    last = ""
    parts = [p.strip() for p in re.split(r"[,،;؛\n]", text or "") if p.strip()]
    for i, part in enumerate(parts):
        for name, regex in _FIELD_RES:
            m = regex.match(part)
            if m:
                fields[name], last = fmt.clean(m.group(1)), name
                break
        else:
            if i == 0 and len(part.split()) <= 3:
                fields["person"], last = fmt.clean(part), "person"
            elif last == "task" or "task" in fields:
                fields["task"] = f"{fields['task']}, {fmt.clean(part)}"
            else:
                fields["task"], last = fmt.clean(part), "task"
    if fields.get("deadline", "").casefold() in _NO_DEADLINE:
        fields["deadline"] = ""
    return fields


def _meeting_lang(meeting: Any) -> str:
    return fmt.lang_of(meeting["language"])


def apply(meeting: Any, action: str, position: int = 0, edit_text: str = "") -> List[str]:
    """Apply one confirmation action to *meeting* (a store row); returns the replies to send."""
    lang = _meeting_lang(meeting)
    meeting_id = meeting["id"]
    if meeting["status"] != "pending":
        return [("الاجتماع #{} مؤكد بالفعل." if lang == "ar" else "Meeting #{} is already confirmed.").format(meeting_id)]
    tasks = store.tasks_for(meeting_id)
    by_pos = {t["position"]: t for t in tasks}
    if action != "all" and position not in by_pos:
        return [("لا توجد مهمة رقم {} في الاجتماع #{}." if lang == "ar" else "There is no task {} in meeting #{}.")
                .format(position, meeting_id)]

    if action == "all":
        store.set_task_status(meeting_id, [t["position"] for t in tasks if t["status"] == "pending"], "confirmed")
        replies: List[str] = []
    elif action == "confirm":
        store.set_task_status(meeting_id, [position], "confirmed")
        replies = [("✅ تم تأكيد المهمة {}." if lang == "ar" else "✅ Task {} confirmed.").format(position)]
    elif action == "remove":
        store.set_task_status(meeting_id, [position], "removed")
        replies = [("❌ تم حذف المهمة {}." if lang == "ar" else "❌ Task {} removed.").format(position)]
    elif action == "edit":
        fields = parse_edit(edit_text)
        if not fields:
            return [EDIT_HELP[lang].format(n=position)]
        if "person" in fields:
            fields["person_key"] = fmt.name_key(fields["person"])
            fields["candidates"] = "[]"
        store.update_task(meeting_id, position, **fields)
        task = next(t for t in store.tasks_for(meeting_id) if t["position"] == position)
        confirm_hint = ("أكّدها بـ ✅ أو اكتب \"تأكيد {}\"." if lang == "ar"
                        else 'Confirm it with ✅ or reply "confirm {}".').format(position)
        replies = [f"✏️ {fmt.task_line(task, lang)}\n{confirm_hint}"]
    else:
        return []

    tasks = store.tasks_for(meeting_id)
    pending = [t for t in tasks if t["status"] == "pending"]
    if pending:
        if action in ("confirm", "remove"):
            left = len(pending)
            replies[-1] += (f" (متبقٍ {left} للتأكيد)" if lang == "ar" else f" ({left} still to confirm)")
        return replies
    store.update_meeting(meeting_id, status="confirmed")
    return replies + [fmt.final_text(meeting_id, lang, tasks)]
