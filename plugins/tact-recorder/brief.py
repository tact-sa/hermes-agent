"""Transcript -> brief + tasks (one structured LLM call), and every text the manager sees.

Models drift from the requested shape (decisions as objects, "open_questions" for open_issues,
"due_date" for deadline, the deadline glued onto the task sentence), so ``normalize`` reads every
common variant and never silently drops a non-empty item. Two short follow-up calls cover what is
still missing: decisions/open issues when both came back empty from a long meeting, and deadlines
when none came back although the transcript names some.

The model only extracts; this module formats. A deadline the model leaves empty always reads
NOT MENTIONED, an owner it leaves empty always reads UNCLEAR (with the candidates it named), and
each person is shown under ONE spelling: the model lists every person with the spellings used
for them (أحمد / Ahmad), and owners are mapped through that alias table here.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# A long Arabic meeting's brief and task list runs to thousands of tokens; the provider default
# can cut it off mid-JSON.
MAX_TOKENS = 8000
STRICT_RETRY = "Return ONLY one valid JSON object, no prose, no code fences."
FOLLOWUP_MAX_TOKENS = 2000
FOLLOWUP_MIN_TRANSCRIPT = 1500  # characters: shorter meetings may genuinely decide nothing
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?|\n?```\s*$")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


class BriefError(Exception):
    """The model gave no usable brief (after the retry)."""

LABELS = {
    "en": {"summary": "📝 SUMMARY", "decisions": "✅ DECISIONS", "open": "❓ OPEN ISSUES",
           "missing": "NOT MENTIONED", "unclear": "❓ UNCLEAR", "or": "or", "none": "None",
           "meeting": "Meeting", "no_tasks": "📋 No tasks were stated in this meeting.",
           "tasks_next": "📋 Tasks ({n}) in the following messages",
           "tasks_below": "📋 Tasks ({n}) in the next message", "tasks_title": "📋 Tasks ({n})",
           "task_n": "📋 Task {i} of {n}", "confirmed": "✅ Confirmed", "removed": "❌ Cancelled",
           "btn_confirm": "✅ Confirm", "btn_edit": "✏️ Edit", "btn_remove": "❌ Cancel",
           "btn_name": "👤 Name", "btn_task": "📌 Task", "btn_deadline": "📅 Deadline", "btn_back": "↩️ Back",
           "name": "Name", "task": "Task", "deadline": "Deadline", "contact": "Contact", "btn_contact": "📞 Contact",
           "registered": "✅ {username} (registered)", "registered_plain": "✅ registered",
           "not_joined": "⏳ not joined yet — send /invite", "no_owner": "—",
           "manager": "Manager", "you": "you", "self_task": "— your task", "btn_undo": "↩️ Undo",
           "sent_telegram": "📤 Sent via Telegram", "sent_email": "📤 Sent by email",
           "confirm_all": "✅ Confirm all ({n})", "all_prompt": "Confirm all remaining tasks at once:",
           "all_done": "✅ All tasks handled.",
           "confirmed_title": "✅ Confirmed tasks — Meeting {id}", "cancelled_title": "❌ Cancelled tasks",
           "no_confirmed": "No tasks were confirmed.",
           "text_help": ('Or reply: "confirm all", "confirm 2", "edit 2: deadline: Thursday", "cancel 3".')},
    "ar": {"summary": "📝 الملخص", "decisions": "✅ القرارات", "open": "❓ قضايا مفتوحة",
           "missing": "غير مذكور", "unclear": "❓ غير واضح", "or": "أو", "none": "لا يوجد",
           "meeting": "اجتماع", "no_tasks": "📋 لم تُذكر أي مهام في هذا الاجتماع.",
           "tasks_next": "📋 المهام ({n}) في الرسائل التالية",
           "tasks_below": "📋 المهام ({n}) في الرسالة التالية", "tasks_title": "📋 المهام ({n})",
           "task_n": "📋 مهمة {i} من {n}", "confirmed": "✅ مؤكدة", "removed": "❌ ملغاة",
           "btn_confirm": "✅ تأكيد", "btn_edit": "✏️ تعديل", "btn_remove": "❌ إلغاء",
           "btn_name": "👤 الاسم", "btn_task": "📌 المهمة", "btn_deadline": "📅 الموعد", "btn_back": "↩️ رجوع",
           "name": "الاسم", "task": "المهمة", "deadline": "الموعد", "contact": "التواصل", "btn_contact": "📞 التواصل",
           "registered": "✅ {username} (مسجل)", "registered_plain": "✅ مسجل",
           "not_joined": "⏳ لم ينضم بعد — أرسل /invite", "no_owner": "—",
           "manager": "المدير", "you": "أنت", "self_task": "— مهمتك", "btn_undo": "↩️ تراجع",
           "sent_telegram": "📤 أُرسلت عبر تيليجرام", "sent_email": "📤 أُرسلت بالبريد الإلكتروني",
           "confirm_all": "✅ تأكيد الكل ({n})", "all_prompt": "تأكيد كل المهام المتبقية دفعة واحدة:",
           "all_done": "✅ تم التعامل مع كل المهام.",
           "confirmed_title": "✅ المهام المؤكدة — اجتماع {id}", "cancelled_title": "❌ المهام الملغاة",
           "no_confirmed": "لم يتم تأكيد أي مهمة.",
           "text_help": 'أو اكتب: "تأكيد الكل"، "تأكيد 2"، "تعديل 2: الموعد: الخميس"، "إلغاء 3".'},
}

INSTRUCTIONS = """\
You analyse the transcript of a work meeting recorded by a manager. The transcript comes from
speech recognition: it can be Arabic, English or both mixed, and may contain recognition errors.

Return JSON only, with these fields:
- language: "ar" if the meeting is mainly Arabic, otherwise "en". Write every text field in that language.
- summary: 5 to 8 short lines summarising the whole meeting.
- decisions: everything decided or agreed in the meeting ("قررنا", "اتفقنا", "موافق", "we agreed",
  "let's go with"), one per item (empty list only if nothing was decided).
- open_issues: everything postponed, left unresolved or still to be decided, one per item (empty
  list only if there is none).
summary, decisions and open_issues are lists of plain strings: one sentence per item, never objects.
- people: every person who is given a task, once each: {"name": one spelling, in the meeting's
  language, "aliases": every other spelling or script used for the same person, e.g. أحمد and Ahmad}.
- tasks: every task actually stated in the meeting, in the order stated:
  {"owner": the person responsible, exactly as in people.name; when several people are jointly
   responsible, all of their names separated by "، "; or "" if it is not clear who;
   "owner_candidates": when the owner is unclear, the people it could be (else []);
   "task": the action only, one short sentence, WITHOUT the deadline;
   "deadline": the deadline exactly as said, or "" if none was said}.
  The deadline goes ONLY in "deadline", never inside "task". Relative deadlines count and are copied
  exactly as said: "بكرة", "يوم الأحد القادم", "قبل نهاية الأسبوع", "tomorrow", "by Tuesday".
  Example: task "تجهيز العرض التقديمي لشركة النخبة وترتيب اجتماع معهم", deadline "قبل نهاية الأسبوع القادم".

Tasks the speaker (the manager who recorded the meeting) takes on himself are tasks too: "أنا بكلم
المدير المالي", "أنا بـ…", "عليّ…", "بتابع أنا…", "I'll…", "I will…", "leave it with me". Their owner
is exactly "المدير" (or "Manager" in an English meeting); never drop them.

Rules: never invent tasks, owners, deadlines or decisions. Only include what was actually said.
If you are not sure who owns a task, leave owner empty and list the candidates instead of guessing.
Treat Arabic and English spellings of one name as the same person."""

SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string", "enum": ["ar", "en"]},
        "summary": {"type": "array", "items": {"type": "string"}},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "open_issues": {"type": "array", "items": {"type": "string"}},
        "people": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "aliases": {"type": "array", "items": {"type": "string"}}},
            "required": ["name"]}},
        "tasks": {"type": "array", "items": {
            "type": "object",
            "properties": {"owner": {"type": "string"},
                           "owner_candidates": {"type": "array", "items": {"type": "string"}},
                           "task": {"type": "string"}, "deadline": {"type": "string"}},
            "required": ["task"]}},
    },
    "required": ["language", "summary", "tasks"],
}

_AR_DIACRITICS = re.compile(r"[ً-ْٰـ]")  # harakat, dagger alef, tatweel
_AR_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"})


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


# Keys whose value is an item's text when a list item comes back as an object ({"decision": "..."}).
_TEXT_KEYS = ("text", "decision", "issue", "question", "title", "description", "content", "value", "item",
              "point", "summary", "task", "القرار", "القضية", "المسألة", "النص", "العنوان", "الوصف", "المحتوى")


def _norm_key(key: Any) -> str:
    return str(key).strip().casefold().replace(" ", "_").replace("-", "_")


def _leaves(value: Any) -> List[str]:
    """Every non-empty scalar inside *value* (nested dicts and lists included), in order."""
    if isinstance(value, dict):
        return [leaf for v in value.values() for leaf in _leaves(v)]
    if isinstance(value, (list, tuple)):
        return [leaf for v in value for leaf in _leaves(v)]
    text = clean(value) if value is not None and not isinstance(value, bool) else ""
    return [text] if text else []


def _item_text(item: Any) -> str:
    """One list item as text: an object gives its text-like field, else all its values joined."""
    if isinstance(item, dict):
        by_key = {_norm_key(k): v for k, v in item.items()}
        for key in _TEXT_KEYS:
            text = " ".join(_leaves(by_key.get(_norm_key(key))))
            if text:
                return text
        return " — ".join(_leaves(item))
    return " ".join(_leaves(item))


def clean_list(values: Any) -> List[str]:
    """Plain strings from a string, a list, an object or nested lists; no non-empty item is dropped."""
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        values = [values]
    out: List[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            out += clean_list(value)
        else:
            text = _item_text(value)
            if text:
                out.append(text)
    return out


# Section and task-field names models use instead of the requested ones (compared via _norm_key).
_SECTION_KEYS = {
    "language": ("language", "lang", "اللغة"),
    "summary": ("summary", "الملخص", "ملخص"),
    "decisions": ("decisions", "decisions_made", "agreed", "القرارات", "قرارات"),
    "open_issues": ("open_issues", "open_questions", "issues", "open", "unresolved", "open_points",
                    "القضايا_المفتوحة", "قضايا_مفتوحة", "النقاط_المفتوحة", "المسائل_المفتوحة", "قضايا_معلقة"),
    "people": ("people", "participants", "persons", "الأشخاص", "الاشخاص"),
    "tasks": ("tasks", "action_items", "actions", "todos", "المهام"),
}
_TASK_KEYS = {
    "task": ("task", "description", "title", "action", "action_item", "what", "المهمة", "مهمة", "الوصف"),
    "owner": ("owner", "person", "assignee", "responsible", "name", "who", "assigned_to",
              "الاسم", "المسؤول", "الشخص", "المكلف"),
    "candidates": ("owner_candidates", "candidates", "possible_owners"),
    "deadline": ("deadline", "due", "due_date", "date", "when", "timeline", "by",
                 "الموعد", "الموعد_النهائي", "التاريخ", "موعد"),
}


# The manager's own tasks: owner "المدير" / "Manager" (and the speaker's "أنا" / "I" / "me", which a
# model sometimes writes instead). Shown as "(you)" and never sent to anyone.
_MANAGER_KEYS = frozenset({"المدير", "مدير", "المدير العام", "manager", "the manager", "أنا", "انا", "me", "i", "myself"})


def is_manager(name: str) -> bool:
    return name_key(name) in {name_key(k) for k in _MANAGER_KEYS}


def split_owners(person: str) -> List[str]:
    """The names in a (possibly joint) owner: "أحمد، عمر" / "Ahmad and Omar" -> both."""
    return [n for n in (clean(p) for p in re.split(r"\s*[،,&]\s*|\s+(?:and|و)\s+", person or "")) if n]
_ALL_SECTION_KEYS = frozenset(_norm_key(k) for keys in _SECTION_KEYS.values() for k in keys)


def _pick(data: Dict[str, Any], names: Tuple[str, ...]) -> Any:
    by_key = {_norm_key(k): v for k, v in data.items()}
    for name in names:
        value = by_key.get(_norm_key(name))
        if value not in (None, "", [], {}):
            return value
    return None


def _has_brief_keys(parsed: Any) -> bool:
    return isinstance(parsed, dict) and any(_norm_key(k) in _ALL_SECTION_KEYS for k in parsed)


# Deadline phrases (Arabic incl. Gulf "بكرة", and English), used to find a deadline glued onto the end
# of a task sentence and to tell whether a transcript names any deadline at all.
_AR_DAYS = "الأحد|الاحد|الإثنين|الاثنين|الثلاثاء|الأربعاء|الاربعاء|الخميس|الجمعة|الجمعه|السبت"
_AR_NEXT = r"(?:\s+(?:القادم|القادمة|الجاي|الجاية|المقبل|المقبلة|الحالي))?"
_AR_PERIOD = r"(?:الأسبوع|الاسبوع|الشهر|اليوم|السنة)"
_AR_CORE = (rf"(?:بعد\s+(?:بكرة|بكره|غد)|بكرة|بكره|غداً|غدا|الليلة"
            rf"|(?:يوم\s+)?(?:{_AR_DAYS}){_AR_NEXT}|(?:نهاية|آخر|اخر)\s+{_AR_PERIOD}{_AR_NEXT}"
            rf"|(?:الأسبوع|الاسبوع|الشهر)\s+(?:القادم|الجاي|المقبل))")
_EN_DAYS = "sunday|monday|tuesday|wednesday|thursday|friday|saturday"
_EN_CORE = (rf"(?:tomorrow|tonight|(?:next\s+|this\s+)?(?:{_EN_DAYS})|next\s+(?:week|month)"
            rf"|(?:the\s+)?end\s+of\s+(?:the\s+|this\s+|next\s+)?(?:week|month|day))")
_TRAILING_DEADLINE_RE = re.compile(
    rf"[\s،,:\-–—]+((?:(?:قبل|بحلول|حتى|خلال|في)\s+|ب)?{_AR_CORE}|(?:(?:by|before|until|on|within)\s+)?{_EN_CORE})"
    r"\s*[.!؟?]*$", re.IGNORECASE)
_DEADLINE_WORDS_RE = re.compile(
    rf"بعد\s+(?:بكرة|غد)|بكرة|بكره|غداً|غدا|(?:يوم\s+)?(?:{_AR_DAYS})|(?:الأسبوع|الاسبوع)\s+(?:القادم|الجاي|المقبل)"
    rf"|(?:نهاية|آخر|اخر)\s+(?:الأسبوع|الاسبوع|الشهر)|\b(?:tomorrow|{_EN_DAYS}|next\s+week|end\s+of\s+(?:the\s+)?(?:week|month))\b",
    re.IGNORECASE)


def split_trailing_deadline(task: str) -> Tuple[str, str]:
    """``(action, deadline)`` when *task* ends with a deadline phrase, else ``(task, "")``."""
    m = _TRAILING_DEADLINE_RE.search(task)
    if not m or not task[:m.start()].strip():
        return task, ""
    return task[:m.start()].strip(" ،,:-–—"), m.group(1).strip()


def name_key(name: str) -> str:
    """Spelling-insensitive key: case, Arabic diacritics and alef/ta-marbuta/ya variants folded."""
    text = unicodedata.normalize("NFKC", clean(name)).casefold()
    return _AR_DIACRITICS.sub("", text).translate(_AR_FOLD)


def lang_of(language: Any) -> str:
    return "ar" if str(language or "").lower().startswith("ar") else "en"


def guess_language(text: str) -> str:
    """Main script of *text*: "ar" when Arabic letters outnumber Latin ones."""
    return "ar" if len(_ARABIC_RE.findall(text or "")) > len(_LATIN_RE.findall(text or "")) else "en"


def extract_json(text: Any) -> Optional[Dict[str, Any]]:
    """The brief object from a model response that may wrap it in code fences or prose, or None.
    Accepted when it carries at least one of the expected keys, even if the strict schema failed."""
    raw = str(text or "").strip()
    candidates = [raw, _FENCE_RE.sub("", raw).strip()]
    start, end = raw.find("{"), raw.rfind("}")
    if 0 <= start < end:
        candidates.append(raw[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if _has_brief_keys(parsed):
            return parsed
    return None


# A decision that hands work to someone ("تكليف خالد بإعداد التقرير", "Omar will send the deck") is a
# task. An explicit assignment verb names the owner even if the model did not list that person;
# "<name> will / عليه / سيقوم ..." counts only for a person the meeting already knows.
_ASSIGN_VERB_RE = re.compile(r"^(?:تم\s+)?(?:تكليف|يكلف|يُكلف|يكلّف|كلف|كُلف|كلّف)\s+|^(?P<en>assign(?:ed)?)\s+",
                             re.IGNORECASE)
_ON_RE = re.compile(r"^على\s+")
_CONNECTOR_RE = re.compile(
    r"^\s*(?:[:：\-–—،,]\s*|(?:عليه|عليها|سيقوم|ستقوم|سوف\s+يقوم|سوف\s+تقوم|سيتولى|ستتولى|يتولى|تتولى|مسؤول\s+عن"
    r"|مسؤولة\s+عن|will|should|shall|must|needs\s+to|is\s+to|to|with)\s+)?", re.IGNORECASE)
_COMPOUND_NAME = frozenset({"عبد", "أبو", "ابو", "ابن", "بن", "آل", "ال"})


def _leading_name(text: str, known: Dict[str, str], allow_unknown: bool) -> Tuple[str, str]:
    """``(canonical name, rest)`` when *text* starts with a person's name, else ``("", text)``."""
    words = text.split()
    for n in (3, 2, 1):  # longest known spelling first ("عبد الله", "Abdul Rahman")
        if len(words) > n and name_key(" ".join(words[:n])) in known:
            return known[name_key(" ".join(words[:n]))], " ".join(words[n:])
    if allow_unknown and len(words) > 1:
        n = 2 if words[0] in _COMPOUND_NAME and len(words) > 2 else 1
        return " ".join(words[:n]).strip("،,:"), " ".join(words[n:])
    return "", text


def assignment(text: str, known: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """The task a sentence assigns to one person, or None when it is not phrased as an assignment."""
    text = re.sub(r"\s*([:：])", r" \1", clean(text))  # "خالد: ..." -> "خالد : ..."
    verb = _ASSIGN_VERB_RE.match(text) or _ON_RE.match(text)
    name, rest = _leading_name(text[verb.end():] if verb else text, known, allow_unknown=bool(verb))
    if not name:
        return None
    connector = _CONNECTOR_RE.match(rest)
    if (not verb or verb.groupdict().get("en")) and not connector.group(0).strip():
        return None  # "<name> <anything>" / "assign X" without "will / عليه / : / to" is not clearly one
    rest = rest[connector.end():].strip()
    if rest.startswith("بـ"):
        rest = rest[2:].strip()
    elif rest.startswith("ب") and (verb or connector.group(0).strip() in ("سيقوم", "ستقوم", "سوف يقوم", "سوف تقوم")):
        rest = rest[1:]
    task, deadline = split_trailing_deadline(rest)
    if not task:
        return None
    return {"person": name, "person_key": name_key(name), "candidates": [], "task": task, "deadline": deadline,
            "contact": ""}


def move_assignments(brief: Dict[str, Any], tasks: List[Dict[str, Any]], known: Dict[str, str]) -> None:
    """Decisions that assign work become tasks (unless already listed), and an unclear task whose
    text is itself an assignment to one known person gets that owner. In place."""
    kept, moved = [], 0
    for item in brief["decisions"]:
        task = assignment(item, known)
        if task is None:
            kept.append(item)
            continue
        key = name_key(task["task"])
        if not any(key in name_key(t["task"]) or name_key(t["task"]) in key for t in tasks):
            tasks.append(task)
        moved += 1
    brief["decisions"] = kept
    claimed = 0
    for task in tasks:
        if task["person"]:
            continue
        found = assignment(task["task"], known)
        if found and name_key(found["person"]) in known:
            task.update(person=found["person"], person_key=found["person_key"], candidates=[],
                        task=found["task"], deadline=task["deadline"] or found["deadline"])
            claimed += 1
    if moved or claimed:
        logger.info("tact-recorder: moved %d assignment(s) from decisions to tasks, gave %d unclear task(s) "
                    "their named owner", moved, claimed)


def _task_fields(raw: Any) -> Tuple[str, List[str], str]:
    """``(task, owner names, deadline)`` from one task item: an object with any of the known key
    spellings, or a plain string (then owner and deadline are unknown)."""
    if not isinstance(raw, dict):
        return _item_text(raw), [], ""
    task = _item_text(_pick(raw, _TASK_KEYS["task"]))
    owners = [n for o in clean_list(_pick(raw, _TASK_KEYS["owner"])) for n in split_owners(o)]
    return task, owners, " ".join(_leaves(_pick(raw, _TASK_KEYS["deadline"])))


def normalize(parsed: Dict[str, Any], fallback_language: str = "en") -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """``(language, brief, tasks)`` from the model's JSON, owners mapped to one spelling each.
    Missing or malformed fields become empty (shown as None / NOT MENTIONED / UNCLEAR)."""
    section = {name: _pick(parsed, keys) for name, keys in _SECTION_KEYS.items()}
    language = lang_of(" ".join(_leaves(section["language"])) or fallback_language)
    brief = {"summary": clean_list(section["summary"])[:8],
             "decisions": clean_list(section["decisions"]),
             "open_issues": clean_list(section["open_issues"])}
    canonical: Dict[str, str] = {}
    people = section["people"] if isinstance(section["people"], list) else []
    for person in people:
        name = _item_text(_pick(person, ("name", "الاسم")) if isinstance(person, dict) else person)
        if not name:
            continue
        aliases = clean_list(_pick(person, ("aliases", "spellings", "الأسماء"))) if isinstance(person, dict) else []
        for spelling in [name, *aliases]:
            canonical.setdefault(name_key(spelling), name)

    def canon(name: str) -> str:
        return canonical.get(name_key(name), name) if name else ""

    raw_tasks = section["tasks"]
    tasks = []
    for raw in raw_tasks if isinstance(raw_tasks, list) else [raw_tasks] if raw_tasks else []:
        task, owners, deadline = _task_fields(raw)
        if not task:
            continue
        if not deadline:
            task, deadline = split_trailing_deadline(task)
        cands = clean_list(_pick(raw, _TASK_KEYS["candidates"])) if isinstance(raw, dict) else []
        # Several owners named for one task are joint owners (it goes to each); candidates are the
        # model saying it is unclear WHO, and that is never guessed.
        person = "، ".join(dict.fromkeys(LABELS[language]["manager"] if is_manager(o) else canon(o) for o in owners))
        candidates = [] if person else list(dict.fromkeys(canon(c) for c in cands))
        # No contact is taken from the meeting: people are reached only as members who joined via
        # /invite, linked by name (contacts.py); any email/phone the model returns is ignored.
        tasks.append({"person": person, "person_key": name_key(person), "candidates": candidates,
                      "task": task, "deadline": deadline, "contact": ""})
    known = dict(canonical)
    known.update({name_key(n): n for t in tasks for n in [t["person"], *t["candidates"]] if n})
    move_assignments(brief, tasks, known)
    return language, brief, tasks


def _shape(parsed: Dict[str, Any]) -> str:
    """Key names and item types of the model's JSON (never its values), to spot format drift."""
    parts = []
    for key, value in list(parsed.items())[:20]:
        name = str(key)[:30]
        if isinstance(value, list):
            kinds: Dict[str, int] = {}
            for item in value:
                kinds[type(item).__name__] = kinds.get(type(item).__name__, 0) + 1
            item_keys = sorted({str(k)[:30] for item in value if isinstance(item, dict) for k in item})[:12]
            desc = ",".join(f"{k}x{n}" for k, n in kinds.items()) or "empty"
            parts.append(f"{name}=list[{desc}]" + (f"{{{','.join(item_keys)}}}" if item_keys else ""))
        else:
            parts.append(f"{name}={type(value).__name__}")
    return " ".join(parts)


class _Route:
    """Which model this brief's calls go to: ``tact_recorder.brief_model`` until a call to it fails,
    then the main chat model for the rest of the brief."""

    def __init__(self, model: Optional[str]):
        self.model = model

    async def call(self, llm: Any, **kw: Any) -> Any:
        if self.model:
            try:
                return await llm.acomplete_structured(**kw, model=self.model)
            except Exception as exc:  # unavailable, auth, trust gate, ...: the brief must still be written
                logger.warning("tact-recorder: brief model %s failed (%s); falling back to the main chat model",
                               self.model, type(exc).__name__)
                self.model = None
        return await llm.acomplete_structured(**kw)


async def analyze(llm: Any, transcript: str, note: str = "",
                  model: Optional[str] = None) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """One structured call, retried once with a stricter instruction when the response holds no
    usable JSON. ``model`` overrides the main model (``tact_recorder.brief_model``)."""
    text = f"Manager's note sent with the recording: {note}\n\n" if note else ""
    blocks = [{"type": "text", "text": f"{text}TRANSCRIPT:\n{transcript}"}]
    route = _Route(model)
    for attempt in (1, 2):
        instructions = INSTRUCTIONS if attempt == 1 else f"{INSTRUCTIONS}\n\n{STRICT_RETRY}"
        result = await route.call(
            llm, instructions=instructions, input=blocks, json_schema=SCHEMA, schema_name="meeting_brief",
            purpose="meeting brief", timeout=300, max_tokens=MAX_TOKENS)
        parsed = result.parsed
        if not _has_brief_keys(parsed):
            parsed = extract_json(result.text)
        if parsed is not None:
            logger.info("tact-recorder: brief shape: %s", _shape(parsed))
            language, brief, tasks = normalize(parsed, guess_language(transcript))
            followed = await _fill_gaps(llm, transcript, language, brief, tasks, route)
            logger.info("tact-recorder: brief counts: decisions=%d open_issues=%d tasks=%d with_deadline=%d"
                        " follow-ups=%s", len(brief["decisions"]), len(brief["open_issues"]), len(tasks),
                        sum(bool(t["deadline"]) for t in tasks), ",".join(followed) or "none")
            return language, brief, tasks
        # Length and stop reason only: the response itself is meeting content and never logged.
        logger.warning("tact-recorder: brief response is not valid JSON (attempt %d/2, %d chars, finish_reason=%s)",
                       attempt, len(result.text or ""), getattr(result, "finish_reason", "") or "unknown")
    raise BriefError("the model returned no valid JSON")


async def _followup(llm: Any, instructions: str, text: str, schema: Dict[str, Any], name: str,
                    route: _Route) -> Optional[Dict[str, Any]]:
    """One short structured call; None (logged by type only) when it fails or returns no JSON."""
    try:
        result = await route.call(
            llm, instructions=instructions, input=[{"type": "text", "text": text}], json_schema=schema,
            schema_name=name, purpose=name.replace("_", " "), timeout=120, max_tokens=FOLLOWUP_MAX_TOKENS)
    except Exception as exc:
        logger.warning("tact-recorder: %s follow-up failed: %s", name, type(exc).__name__)
        return None
    parsed = result.parsed if isinstance(result.parsed, dict) else None
    if parsed is None:
        raw = str(result.text or "")
        start, end = raw.find("{"), raw.rfind("}")
        try:
            parsed = json.loads(raw[start:end + 1]) if 0 <= start < end else None
        except ValueError:
            parsed = None
    if not isinstance(parsed, dict):
        logger.warning("tact-recorder: %s follow-up returned no JSON (%d chars)", name, len(result.text or ""))
        return None
    logger.info("tact-recorder: %s follow-up shape: %s", name, _shape(parsed))
    return parsed


DECISIONS_INSTRUCTIONS = """From this meeting transcript, list only the decisions and the open issues, in {language}.
Decisions: everything decided or agreed ("قررنا", "اتفقنا", "موافق", "we agreed").
Open issues: everything postponed, unresolved or still to be decided.
Return ONLY this JSON: {{"decisions": ["..."], "open_issues": ["..."]}} with plain strings, one per
item. Never invent anything; use empty lists if there is none."""
DECISIONS_SCHEMA = {"type": "object", "properties": {
    "decisions": {"type": "array", "items": {"type": "string"}},
    "open_issues": {"type": "array", "items": {"type": "string"}}}, "required": ["decisions", "open_issues"]}
DEADLINES_INSTRUCTIONS = """Below are a meeting transcript and the numbered tasks taken from it. For each task, give the
deadline exactly as it was said in the meeting, including relative ones ("بكرة", "يوم الأحد القادم",
"قبل نهاية الأسبوع", "tomorrow", "by Tuesday"), or "" if no deadline was said for it.
Return ONLY this JSON: {"deadlines": [{"n": task number, "deadline": "..."}]}. Never invent a deadline."""
DEADLINES_SCHEMA = {"type": "object", "properties": {"deadlines": {"type": "array", "items": {
    "type": "object", "properties": {"n": {"type": "integer"}, "deadline": {"type": "string"}},
    "required": ["n", "deadline"]}}}, "required": ["deadlines"]}


def _deadline_pairs(parsed: Dict[str, Any]) -> List[Tuple[int, str]]:
    """``(task number, deadline)`` pairs from a list of objects or a {"1": "..."} mapping."""
    raw = _pick(parsed, ("deadlines", "tasks", "المواعيد"))
    pairs = []
    if isinstance(raw, dict):
        items = [{"n": k, "deadline": v} for k, v in raw.items()]
    else:
        items = raw if isinstance(raw, list) else []
    for i, item in enumerate(items, 1):
        if isinstance(item, dict):
            number = _pick(item, ("n", "number", "task_number", "index", "id", "رقم"))
            deadline = " ".join(_leaves(_pick(item, _TASK_KEYS["deadline"])))
        else:
            number, deadline = i, " ".join(_leaves(item))
        try:
            pairs.append((int(str(number).strip()), deadline))
        except (TypeError, ValueError):
            continue
    return pairs


async def _fill_gaps(llm: Any, transcript: str, language: str, brief: Dict[str, Any],
                     tasks: List[Dict[str, Any]], route: _Route) -> List[str]:
    """Follow-up calls for what the main call lost; returns which ones ran."""
    ran = []
    if not brief["decisions"] and not brief["open_issues"] and len(transcript) > FOLLOWUP_MIN_TRANSCRIPT:
        ran.append("decisions")
        parsed = await _followup(llm, DECISIONS_INSTRUCTIONS.format(language="Arabic" if language == "ar" else "English"),
                                 f"TRANSCRIPT:\n{transcript}", DECISIONS_SCHEMA, "meeting_decisions", route)
        if parsed:
            brief["decisions"] = clean_list(_pick(parsed, _SECTION_KEYS["decisions"]))
            brief["open_issues"] = clean_list(_pick(parsed, _SECTION_KEYS["open_issues"]))
            known = {name_key(n): n for t in tasks for n in [t["person"], *t["candidates"]] if n}
            move_assignments(brief, tasks, known)
    if tasks and not any(t["deadline"] for t in tasks) and _DEADLINE_WORDS_RE.search(transcript):
        ran.append("deadlines")
        listing = "\n".join(f"{i}. {t['person'] or ' / '.join(t['candidates']) or '?'} — {t['task']}"
                             for i, t in enumerate(tasks, 1))
        parsed = await _followup(llm, DEADLINES_INSTRUCTIONS, f"TRANSCRIPT:\n{transcript}\n\nTASKS:\n{listing}",
                                 DEADLINES_SCHEMA, "meeting_deadlines", route)
        for number, deadline in _deadline_pairs(parsed or {}):
            if 1 <= number <= len(tasks) and deadline and not tasks[number - 1]["deadline"]:
                tasks[number - 1]["deadline"] = deadline
    return ran


# -- formatting --------------------------------------------------------------------------------
# One numbering everywhere: a task's number is its position in the meeting. No brackets: "[#3]"
# renders scrambled inside right-to-left Arabic text.

def owner_text(task: Dict[str, Any], lang: str, question: bool = True) -> str:
    if task.get("owner_display"):  # set by contacts.annotate: members as "full name (role)"
        return task["owner_display"]
    if task["person"]:
        names = split_owners(task["person"])
        if not any(is_manager(n) for n in names):
            return task["person"]
        me = f"{LABELS[lang]['manager']} ({LABELS[lang]['you']})"
        return "، ".join(me if is_manager(n) else n for n in names)
    labels = LABELS[lang]
    cands = task.get("candidates") or []
    if not cands:
        return labels["unclear"]
    joiner = f" {labels['or']} "
    mark = ("؟" if lang == "ar" else "?") if question else ""
    return f"{labels['unclear']}: {joiner.join(cands)}{mark}"


def task_card(task: Dict[str, Any], total: int, lang: str) -> str:
    """One task as its own message; a handled task shows its status under it."""
    labels = LABELS[lang]
    text = (f"{labels['task_n'].format(i=task['position'], n=total)}\n"
            f"👤 {owner_text(task, lang)}\n"
            f"📌 {task['task']}\n"
            f"📅 {task['deadline'] or labels['missing']}\n"
            f"📞 {labels['contact']}: {contact_text(task, lang)}")
    status = [labels[task["status"]]] if task["status"] in ("confirmed", "removed") else []
    status += sent_lines(task, lang)
    return f"{text}\n\n" + "\n".join(status) if status else text


def contact_text(task: Dict[str, Any], lang: str) -> str:
    """The member a task reaches: ``contact_display`` (set by ``contacts.annotate``: registered /
    not joined yet), else "not joined yet" for an owner and "—" for an unclear one."""
    return task.get("contact_display") or LABELS[lang]["not_joined" if task.get("person") else "no_owner"]


def sent_lines(task: Dict[str, Any], lang: str) -> List[str]:
    via = [c for c in (task.get("sent_via") or "").split(",") if c]
    return [LABELS[lang][f"sent_{c}"] for c in via if f"sent_{c}" in LABELS[lang]]


def combined_text(tasks: List[Dict[str, Any]], total: int, lang: str) -> str:
    """The pending task cards in one message: many tasks, or no inline buttons on this platform."""
    labels = LABELS[lang]
    return (labels["tasks_title"].format(n=len(tasks)) + "\n\n"
            + "\n\n".join(task_card(t, total, lang) for t in tasks) + f"\n\n{labels['text_help']}")


def _bullets(items: List[str], lang: str) -> str:
    return "\n".join(f"- {item}" for item in items) if items else f"- {LABELS[lang]['none']}"


def brief_text(meeting_id: int, created_at: str, lang: str, brief: Dict[str, Any],
               n_tasks: int, cards: bool = True) -> str:
    """Summary, decisions and open issues; the tasks follow in their own message(s)."""
    labels = LABELS[lang]
    if not n_tasks:
        tasks_line = labels["no_tasks"]
    else:
        tasks_line = labels["tasks_next" if cards else "tasks_below"].format(n=n_tasks)
    return "\n\n".join([
        f"🎙️ {labels['meeting']} #{meeting_id} — {created_at[:16].replace('T', ' ')}",
        f"{labels['summary']}\n" + "\n".join(brief.get("summary") or [labels["none"]]),
        f"{labels['decisions']}\n{_bullets(brief.get('decisions') or [], lang)}",
        f"{labels['open']}\n{_bullets(brief.get('open_issues') or [], lang)}",
        tasks_line])


def task_block(task: Dict[str, Any], lang: str) -> str:
    """A handled task in the summary: its number on its own line, then labelled fields."""
    labels = LABELS[lang]
    return (f"{task['position']}.\n"
            f"👤 {labels['name']}: {owner_text(task, lang, question=False)}\n"
            f"📌 {labels['task']}: {task['task']}\n"
            f"📅 {labels['deadline']}: {task['deadline'] or labels['missing']}\n"
            f"📞 {labels['contact']}: {contact_text(task, lang)}"
            + "".join(f"\n{line}" for line in sent_lines(task, lang)))


def final_text(meeting_id: int, lang: str, tasks: List[Dict[str, Any]]) -> str:
    """Confirmed tasks, then (only if any) cancelled ones, each in task-number order; pending tasks
    are not part of it."""
    labels = LABELS[lang]
    confirmed = [t for t in tasks if t["status"] == "confirmed"]
    cancelled = [t for t in tasks if t["status"] == "removed"]
    parts = [labels["confirmed_title"].format(id=meeting_id)]
    parts += [task_block(t, lang) for t in confirmed] or [labels["no_confirmed"]]
    if cancelled:
        parts.append(labels["cancelled_title"])
        parts += [task_block(t, lang) for t in cancelled]
    return "\n\n".join(parts)
