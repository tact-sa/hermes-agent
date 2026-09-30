"""The contacts book: how to reach each person tasks are assigned to.

One row per person (name + other spellings) with any of: a Telegram chat id (only ever set when the
manager approves an invite join, see ``invite.py``), a Telegram @username, an email, a phone. Names
match the way owners do (``brief.name_key``: Arabic/English variants, case, diacritics).

``route`` decides how a task reaches one owner: the task's own contact (said in the meeting or set
by the manager) first, else the book. Telegram needs a chat id: a bot cannot message someone who
never started it, so an @username alone is "not registered yet". Email needs SMTP configured.
Phones are stored but cannot be messaged yet.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from . import brief as fmt
from . import store


def _connect():
    con = store.connect()
    con.execute(
        "CREATE TABLE IF NOT EXISTS contacts ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " name TEXT NOT NULL, name_key TEXT NOT NULL UNIQUE, aliases TEXT NOT NULL DEFAULT '[]',"
        " telegram_chat_id TEXT, telegram_username TEXT, email TEXT, phone TEXT,"
        " updated_at TEXT NOT NULL)")
    # Names the manager removed (/contacts delete) that live on in past tasks: kept out of pickers.
    con.execute("CREATE TABLE IF NOT EXISTS hidden_names (name_key TEXT PRIMARY KEY)")
    return con


# A person's name, not a placeholder the model or the manager wrote while unsure ("احمد او عمر
# (يحدد لاحقا)", "فهد او نورة", "Sara / Omar") and not a phrase.
_NOT_A_NAME_RE = re.compile(r"(?:^|\s)(?:او|أو|or)(?:\s|$)|[/\\()\[\]{}<>؟?]", re.IGNORECASE)
MAX_NAME_WORDS = 3


def is_person_name(name: str) -> bool:
    name = fmt.clean(name)
    return bool(name) and len(name.split()) <= MAX_NAME_WORDS and not _NOT_A_NAME_RE.search(name)


def _hidden() -> set:
    with _connect() as con:
        return {r["name_key"] for r in con.execute("SELECT name_key FROM hidden_names").fetchall()}


def _set_hidden(name: str, hidden: bool) -> None:
    with _connect() as con:
        if hidden:
            con.execute("INSERT OR IGNORE INTO hidden_names (name_key) VALUES (?)", (fmt.name_key(name),))
        else:
            con.execute("DELETE FROM hidden_names WHERE name_key = ?", (fmt.name_key(name),))


def picker_names(platform: str, chat_id: str, limit: int = 12) -> List[str]:
    """Names to offer on buttons (e.g. join approval): the contacts book first, then owners of
    confirmed tasks from recent meetings; placeholders, phrases and removed names left out; one
    button per person (``brief.name_key``)."""
    hidden = _hidden()
    names = [r["name"] for r in all_contacts()]
    names += [n for person in store.recent_people(platform, chat_id) for n in fmt.split_owners(person)]
    unique: Dict[str, str] = {}
    for name in names:
        key = fmt.name_key(name)
        if is_person_name(name) and key not in hidden:
            unique.setdefault(key, fmt.clean(name))
    return list(unique.values())[:limit]


def all_contacts() -> List[Dict[str, Any]]:
    with _connect() as con:
        rows = con.execute("SELECT * FROM contacts ORDER BY name").fetchall()
    return [{**dict(r), "aliases": json.loads(r["aliases"] or "[]")} for r in rows]


def find(name: str) -> Optional[Dict[str, Any]]:
    """The person's row by name or any saved spelling."""
    key = fmt.name_key(name)
    if not key:
        return None
    for row in all_contacts():
        if key == row["name_key"] or key in {fmt.name_key(a) for a in row["aliases"]}:
            return row
    return None


def by_username(username: str) -> Optional[Dict[str, Any]]:
    wanted = username.lstrip("@").casefold()
    return next((r for r in all_contacts() if (r["telegram_username"] or "").lstrip("@").casefold() == wanted), None)


def by_chat_id(chat_id: str) -> Optional[Dict[str, Any]]:
    return next((r for r in all_contacts() if r["telegram_chat_id"] == str(chat_id)), None)


def _upsert(name: str, **fields: Any) -> None:
    fields["updated_at"] = store.now().isoformat(timespec="seconds")
    _set_hidden(name, False)  # a person the manager gives a contact to is back in the pickers
    row = find(name)
    with _connect() as con:
        if row is None:
            fields.update(name=fmt.clean(name), name_key=fmt.name_key(name))
            cols = ", ".join(fields)
            con.execute(f"INSERT INTO contacts ({cols}) VALUES ({', '.join('?' * len(fields))})",
                        tuple(fields.values()))
        else:
            cols = ", ".join(f"{k} = ?" for k in fields)
            con.execute(f"UPDATE contacts SET {cols} WHERE id = ?", (*fields.values(), row["id"]))


def save_contact(name: str, contact: str) -> bool:
    """Store an email / @username / phone for *name*; False when *contact* is none of those."""
    kind, value = fmt.classify_contact(contact)
    if not kind:
        return False
    _upsert(name, **{{"email": "email", "username": "telegram_username", "phone": "phone"}[kind]: value})
    return True


def link_telegram(name: str, chat_id: str, username: str = "") -> None:
    """The manager approved this Telegram chat for *name*; a chat belongs to one person only."""
    with _connect() as con:
        con.execute("UPDATE contacts SET telegram_chat_id = NULL WHERE telegram_chat_id = ?", (str(chat_id),))
    fields: Dict[str, Any] = {"telegram_chat_id": str(chat_id)}
    if username:
        fields["telegram_username"] = "@" + username.lstrip("@")
    _upsert(name, **fields)


def _in_past_tasks(name: str) -> bool:
    key = fmt.name_key(name)
    with store.connect() as con:
        people = [r["person"] for r in con.execute("SELECT DISTINCT person FROM tasks WHERE person != ''")]
    return any(fmt.name_key(n) == key for person in people for n in fmt.split_owners(person))


def delete(name: str) -> bool:
    """Remove *name* from the book and from every name picker (past tasks keep their text);
    False when the name is neither in the book nor in any task."""
    row = find(name)
    known = row is not None or _in_past_tasks(name)
    if row is not None:
        with _connect() as con:
            con.execute("DELETE FROM contacts WHERE id = ?", (row["id"],))
    if known:
        _set_hidden(name, True)
    return known


def rename(old: str, new: str) -> Tuple[bool, str]:
    """Fix a person's name in the book and in past tasks: ``(done, message)``."""
    old, new = fmt.clean(old), fmt.clean(new)
    if not is_person_name(new):
        return False, f"«{new}» ليس اسماً صالحاً."
    row, target = find(old), find(new)
    if row is not None and target is not None and target["id"] != row["id"]:
        return False, f"{new} موجود بالفعل في جهات الاتصال."
    tasks = store.rename_person(fmt.name_key(old), new, fmt.name_key, fmt.split_owners)
    if row is None and not tasks:
        return False, f"لا يوجد اسم {old}."
    if row is not None:
        with _connect() as con:
            con.execute("UPDATE contacts SET name = ?, name_key = ?, updated_at = ? WHERE id = ?",
                        (new, fmt.name_key(new), store.now().isoformat(timespec="seconds"), row["id"]))
    _set_hidden(old, True)
    _set_hidden(new, False)
    return True, f"✏️ أُعيدت تسمية {old} إلى {new}" + (f" (في {tasks} مهمة)" if tasks else "") + "."


def has_value(name: str, contact: str) -> bool:
    """Whether the book already holds exactly this contact for *name* (no need to ask again)."""
    kind, value = fmt.classify_contact(contact)
    row = find(name)
    column = {"email": "email", "username": "telegram_username", "phone": "phone"}.get(kind)
    return bool(row and column and (row[column] or "").casefold() == value.casefold())


# -- routing and display -------------------------------------------------------------------------

class Route(NamedTuple):
    channel: str  # "telegram" / "email", or "" when the task can't be sent to this person
    target: str  # chat id / email address
    reason: str  # why not: unknown / not_registered / email_off / phone


def route(name: str, explicit: str, email_ready: bool) -> Route:
    kind, value = fmt.classify_contact(explicit)
    row = None
    if kind == "username":
        row = by_username(value)
        return Route("telegram", row["telegram_chat_id"], "") if row and row["telegram_chat_id"] else \
            Route("", "", "not_registered")
    if kind == "email":
        return Route("email", value, "") if email_ready else Route("", "", "email_off")
    if kind == "phone":
        return Route("", "", "phone")
    row = find(name)
    if row and row["telegram_chat_id"]:
        return Route("telegram", row["telegram_chat_id"], "")
    if row and row["email"]:
        return Route("email", row["email"], "") if email_ready else Route("", "", "email_off")
    if row and row["telegram_username"]:
        return Route("", "", "not_registered")
    if row and row["phone"]:
        return Route("", "", "phone")
    return Route("", "", "unknown")


def row_text(row: Dict[str, Any], lang: str) -> str:
    """How a book entry reads on a card or in /contacts."""
    labels = fmt.LABELS[lang]
    telegram = "تيليجرام" if lang == "ar" else "Telegram"
    parts = []
    if row["telegram_chat_id"]:
        parts.append(f"✅ {telegram}" + (f" ({row['telegram_username']})" if row["telegram_username"] else ""))
    elif row["telegram_username"]:
        parts.append(f"{row['telegram_username']} ({labels['not_registered']})")
    parts += [v for v in (row["email"], row["phone"]) if v]
    return " · ".join(parts)


def display(task: Dict[str, Any], lang: str) -> str:
    labels = fmt.LABELS[lang]
    kind, value = fmt.classify_contact(task.get("contact") or "")
    if kind == "username":
        row = by_username(value)
        return value if row and row["telegram_chat_id"] else f"{value} ({labels['not_registered']})"
    if kind:
        return value
    names = fmt.split_owners(task["person"])
    parts = []
    for name in names:
        row = find(name)
        text = row_text(row, lang) if row else ""
        if text:
            parts.append(f"{name}: {text}" if len(names) > 1 else text)
    return "، ".join(parts) or labels["unknown"]


def annotate(tasks: List[Dict[str, Any]], lang: str) -> List[Dict[str, Any]]:
    """Add each task's ``contact_display`` (what cards, summaries and /brief show). In place."""
    for task in tasks:
        task["contact_display"] = display(task, lang)
    return tasks


def list_text() -> str:
    rows = all_contacts()
    if not rows:
        return "👥 لا توجد جهات اتصال بعد. أرسل /invite لدعوة شخص عبر تيليجرام."
    lines = [f"👥 جهات الاتصال ({len(rows)}):"]
    for row in rows:
        lines.append(f"• {row['name']} — {row_text(row, 'ar') or 'غير معروف'}")
    lines.append("لحذف جهة: /contacts delete <الاسم> · لتصحيح اسم: /contacts rename <القديم> -> <الجديد>")
    return "\n".join(lines)
