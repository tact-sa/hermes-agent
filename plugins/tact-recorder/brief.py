"""Transcript -> brief + tasks (one structured LLM call), and every text the manager sees.

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
_EXPECTED_KEYS = frozenset({"language", "summary", "decisions", "open_issues", "people", "tasks"})
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
           "name": "Name", "task": "Task", "deadline": "Deadline",
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
           "name": "الاسم", "task": "المهمة", "deadline": "الموعد",
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
- decisions: decisions actually taken in the meeting (empty list if none).
- open_issues: questions or problems left unresolved (empty list if none).
- people: every person who is given a task, once each: {"name": one spelling, in the meeting's
  language, "aliases": every other spelling or script used for the same person, e.g. أحمد and Ahmad}.
- tasks: every task actually stated in the meeting, in the order stated:
  {"owner": the person responsible, exactly as in people.name, or "" if it is not clear who;
   "owner_candidates": when the owner is unclear, the people it could be (else []);
   "task": what must be done, one short sentence;
   "deadline": the deadline exactly as said, or "" if none was said}.

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


def clean_list(values: Any) -> List[str]:
    if isinstance(values, str):
        values = [values]
    return [c for c in (clean(v) for v in (values or []) if not isinstance(v, (dict, list))) if c]


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
        if isinstance(parsed, dict) and _EXPECTED_KEYS & parsed.keys():
            return parsed
    return None


def normalize(parsed: Dict[str, Any], fallback_language: str = "en") -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """``(language, brief, tasks)`` from the model's JSON, owners mapped to one spelling each.
    Missing or malformed fields become empty (shown as None / NOT MENTIONED / UNCLEAR)."""
    language = lang_of(parsed.get("language") or fallback_language)
    brief = {"summary": clean_list(parsed.get("summary"))[:8],
             "decisions": clean_list(parsed.get("decisions")),
             "open_issues": clean_list(parsed.get("open_issues"))}
    canonical: Dict[str, str] = {}
    for person in parsed.get("people") or []:
        if not isinstance(person, dict) or not clean(person.get("name")):
            continue
        name = clean(person["name"])
        for spelling in [name, *clean_list(person.get("aliases"))]:
            canonical.setdefault(name_key(spelling), name)

    def canon(name: str) -> str:
        return canonical.get(name_key(name), name) if name else ""

    tasks = []
    for raw in parsed.get("tasks") or []:
        if not isinstance(raw, dict) or not clean(raw.get("task")):
            continue
        person = canon(clean(raw.get("owner")))
        candidates = [] if person else list(dict.fromkeys(canon(c) for c in clean_list(raw.get("owner_candidates"))))
        tasks.append({"person": person, "person_key": name_key(person), "candidates": candidates,
                      "task": clean(raw["task"]), "deadline": clean(raw.get("deadline"))})
    return language, brief, tasks


async def analyze(llm: Any, transcript: str, note: str = "",
                  model: Optional[str] = None) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """One structured call, retried once with a stricter instruction when the response holds no
    usable JSON. ``model`` overrides the main model (``tact_recorder.brief_model``)."""
    text = f"Manager's note sent with the recording: {note}\n\n" if note else ""
    blocks = [{"type": "text", "text": f"{text}TRANSCRIPT:\n{transcript}"}]
    overrides = {"model": model} if model else {}
    for attempt in (1, 2):
        instructions = INSTRUCTIONS if attempt == 1 else f"{INSTRUCTIONS}\n\n{STRICT_RETRY}"
        result = await llm.acomplete_structured(
            instructions=instructions, input=blocks, json_schema=SCHEMA, schema_name="meeting_brief",
            purpose="meeting brief", timeout=300, max_tokens=MAX_TOKENS, **overrides)
        parsed = result.parsed
        if not (isinstance(parsed, dict) and _EXPECTED_KEYS & parsed.keys()):
            parsed = extract_json(result.text)
        if parsed is not None:
            return normalize(parsed, guess_language(transcript))
        # Length and stop reason only: the response itself is meeting content and never logged.
        logger.warning("tact-recorder: brief response is not valid JSON (attempt %d/2, %d chars, finish_reason=%s)",
                       attempt, len(result.text or ""), getattr(result, "finish_reason", "") or "unknown")
    raise BriefError("the model returned no valid JSON")


# -- formatting --------------------------------------------------------------------------------
# One numbering everywhere: a task's number is its position in the meeting. No brackets: "[#3]"
# renders scrambled inside right-to-left Arabic text.

def owner_text(task: Dict[str, Any], lang: str, question: bool = True) -> str:
    if task["person"]:
        return task["person"]
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
            f"📅 {task['deadline'] or labels['missing']}")
    status = labels.get(task["status"]) if task["status"] in ("confirmed", "removed") else ""
    return f"{text}\n\n{status}" if status else text


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
            f"📅 {labels['deadline']}: {task['deadline'] or labels['missing']}")


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
