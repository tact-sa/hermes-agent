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
           "tasks": "📋 TASKS", "by_person": "👥 BY PERSON", "missing": "NOT MENTIONED",
           "unclear": "❓ UNCLEAR", "or": "or", "none": "None", "meeting": "Meeting",
           "no_tasks": "No tasks were stated in this meeting.",
           "confirmed_title": "✅ Confirmed tasks", "no_confirmed": "No tasks were confirmed.",
           "legend": "Confirm the tasks: ✅ confirm · ✏️ edit · ❌ remove",
           "confirm_all": "✅ Confirm all",
           "text_help": ('Or reply: "confirm all", "confirm 2", "edit 2: Omar, due Sunday", "remove 3".')},
    "ar": {"summary": "📝 الملخص", "decisions": "✅ القرارات", "open": "❓ قضايا مفتوحة",
           "tasks": "📋 المهام", "by_person": "👥 حسب الشخص", "missing": "غير مذكور",
           "unclear": "❓ غير واضح", "or": "أو", "none": "لا يوجد", "meeting": "اجتماع",
           "no_tasks": "لم تُذكر أي مهام في هذا الاجتماع.",
           "confirmed_title": "✅ المهام المؤكدة", "no_confirmed": "لم يتم تأكيد أي مهمة.",
           "legend": "أكّد المهام: ✅ تأكيد · ✏️ تعديل · ❌ حذف",
           "confirm_all": "✅ تأكيد الكل",
           "text_help": 'أو اكتب: "تأكيد الكل"، "تأكيد 2"، "تعديل 2: عمر، الموعد الأحد"، "حذف 3".'},
}
STATUS_MARK = {"pending": "⏳", "confirmed": "✅", "removed": "❌"}

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

def owner_text(task: Dict[str, Any], lang: str) -> str:
    if task["person"]:
        return task["person"]
    labels = LABELS[lang]
    cands = task.get("candidates") or []
    if not cands:
        return labels["unclear"]
    joiner = f" {labels['or']} "
    return f"{labels['unclear']}: {joiner.join(cands)}{'؟' if lang == 'ar' else '?'}"


def task_line(task: Dict[str, Any], lang: str, mark: bool = False) -> str:
    prefix = f"{STATUS_MARK.get(task['status'], '')} " if mark else ""
    deadline = task["deadline"] or LABELS[lang]["missing"]
    return f"{prefix}{task['position']}. {owner_text(task, lang)} → {task['task']} → {deadline}"


def by_person(tasks: List[Dict[str, Any]], lang: str) -> List[str]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        groups.setdefault(task["person_key"] if task["person"] else "", []).append(task)
    lines = []
    # Named people first (in order of first task), unclear owners last.
    for key in sorted(groups, key=lambda k: k == ""):
        items = groups[key]
        who = items[0]["person"] if key else LABELS[lang]["unclear"]
        missing = LABELS[lang]["missing"]
        entries = " ".join(f"{i}) {t['task']} ({t['deadline'] or missing}) [#{t['position']}]"
                           for i, t in enumerate(items, 1))
        lines.append(f"{who}: {entries}")
    return lines


def _bullets(items: List[str], lang: str) -> str:
    return "\n".join(f"- {item}" for item in items) if items else f"- {LABELS[lang]['none']}"


def brief_text(meeting_id: int, created_at: str, lang: str, brief: Dict[str, Any],
               tasks: List[Dict[str, Any]], mark: bool = False) -> str:
    labels = LABELS[lang]
    shown = [t for t in tasks if mark or t["status"] != "removed"]
    parts = [f"🎙️ {labels['meeting']} #{meeting_id} — {created_at[:16].replace('T', ' ')}",
             f"{labels['summary']}\n" + "\n".join(brief.get("summary") or [labels["none"]]),
             f"{labels['decisions']}\n{_bullets(brief.get('decisions') or [], lang)}",
             f"{labels['open']}\n{_bullets(brief.get('open_issues') or [], lang)}"]
    if shown:
        parts.append(f"{labels['tasks']}\n" + "\n".join(task_line(t, lang, mark) for t in shown))
        active = [t for t in shown if t["status"] != "removed"]
        if active:
            parts.append(f"{labels['by_person']}\n" + "\n".join(by_person(active, lang)))
    else:
        parts.append(f"{labels['tasks']}\n{labels['no_tasks']}")
    return "\n\n".join(parts)


def final_text(meeting_id: int, lang: str, tasks: List[Dict[str, Any]]) -> str:
    labels = LABELS[lang]
    confirmed = [t for t in tasks if t["status"] == "confirmed"]
    if not confirmed:
        return f"{labels['meeting']} #{meeting_id}: {labels['no_confirmed']}"
    return (f"{labels['confirmed_title']} — {labels['meeting']} #{meeting_id}\n"
            + "\n".join(by_person(confirmed, lang)))
