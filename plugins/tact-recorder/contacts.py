"""The people tasks go to: ONLY members who joined through a manager-approved /invite link.

One row per person: a short name (unique; what meetings usually say), other spellings (aliases),
an optional full name and role (shown as "full name (role)" wherever the manager picks or reads a
person), and the Telegram chat id + @username set when the manager approves their join
(``invite.py``). There is no manual @username, email or phone entry: a person is reachable exactly
when they are a registered member (``telegram_chat_id``), and ``/team`` is the panel that manages them.

A task reaches its owner's member, matched by short name, aliases, full name or first name
(``brief.name_key``), or the member the manager picked for that task (``tasks.contact_id``). A name
that fits more than one member is never guessed: the manager chooses in the send preview.
Older email / phone / username values in the table are no longer read.
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
        " updated_at TEXT NOT NULL, full_name TEXT, role TEXT, joined_at TEXT)")
    for column in ("full_name", "role", "joined_at"):  # added after the first release
        if column not in {row[1] for row in con.execute("PRAGMA table_info(contacts)")}:
            con.execute(f"ALTER TABLE contacts ADD COLUMN {column} TEXT")
    # Names the manager removed (removed in the /team panel) that live on in past tasks: kept out of pickers.
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
        if is_person_name(name) and key not in hidden and not fmt.is_manager(name):
            unique.setdefault(key, fmt.clean(name))
    return list(unique.values())[:limit]


def all_contacts() -> List[Dict[str, Any]]:
    with _connect() as con:
        rows = con.execute("SELECT * FROM contacts ORDER BY name").fetchall()
    return [{**dict(r), "aliases": json.loads(r["aliases"] or "[]")} for r in rows]


def label(row: Dict[str, Any]) -> str:
    """How the manager sees a person: "full name (role)", falling back to the short name."""
    who = row.get("full_name") or row["name"]
    return f"{who} ({row['role']})" if row.get("role") else who


def label_for(name: str) -> str:
    row = find(name)
    return label(row) if row else name


def by_id(contact_id: Any) -> Optional[Dict[str, Any]]:
    return next((r for r in all_contacts() if str(r["id"]) == str(contact_id)), None)


def exact(name: str) -> Optional[Dict[str, Any]]:
    """The row whose short name, spelling or full name is exactly *name* (for writes)."""
    key = fmt.name_key(name)
    for row in all_contacts() if key else []:
        if key in {row["name_key"], fmt.name_key(row["full_name"] or "")} | {fmt.name_key(a) for a in row["aliases"]}:
            return row
    return None


def matches(name: str) -> List[Dict[str, Any]]:
    """Every person *name* may mean: an exact full name is one person; otherwise the short name,
    spellings and first names all count, so a shared first name gives several rows."""
    key = fmt.name_key(name)
    if not key:
        return []
    rows = all_contacts()
    full = [r for r in rows if r["full_name"] and fmt.name_key(r["full_name"]) == key]
    if len(full) == 1:
        return full

    def first(text: str) -> str:
        return fmt.name_key(text).split(" ")[0] if text else ""
    return [r for r in rows if key == r["name_key"] or key in {fmt.name_key(a) for a in r["aliases"]}
            or (" " not in key and key in {first(r["name"]), first(r["full_name"] or "")})]


def find(name: str) -> Optional[Dict[str, Any]]:
    """The one person *name* means, or None when it is nobody or more than one person."""
    found = matches(name)
    return found[0] if len(found) == 1 else None


def by_username(username: str) -> Optional[Dict[str, Any]]:
    wanted = username.lstrip("@").casefold()
    return next((r for r in all_contacts() if (r["telegram_username"] or "").lstrip("@").casefold() == wanted), None)


def by_chat_id(chat_id: str) -> Optional[Dict[str, Any]]:
    return next((r for r in all_contacts() if r["telegram_chat_id"] == str(chat_id)), None)


def _upsert(name: str, **fields: Any) -> None:
    fields["updated_at"] = store.now().isoformat(timespec="seconds")
    _set_hidden(name, False)  # a person the manager gives a contact to is back in the pickers
    row = exact(name)
    with _connect() as con:
        if row is None:
            fields.update(name=fmt.clean(name), name_key=fmt.name_key(name))
            cols = ", ".join(fields)
            con.execute(f"INSERT INTO contacts ({cols}) VALUES ({', '.join('?' * len(fields))})",
                        tuple(fields.values()))
        else:
            cols = ", ".join(f"{k} = ?" for k in fields)
            con.execute(f"UPDATE contacts SET {cols} WHERE id = ?", (*fields.values(), row["id"]))


def is_member(row: Optional[Dict[str, Any]]) -> bool:
    return bool(row and row["telegram_chat_id"])


def members() -> List[Dict[str, Any]]:
    """Registered members, sorted by the name the manager sees."""
    return sorted((r for r in all_contacts() if is_member(r)), key=lambda r: fmt.name_key(r["full_name"] or r["name"]))


def members_for(name: str) -> List[Dict[str, Any]]:
    return [r for r in matches(name) if is_member(r)]


def knows(row: Dict[str, Any], name: str) -> bool:
    """Whether *name* already leads to this member (short name, alias or full name)."""
    key = fmt.name_key(name)
    return key in {row["name_key"], fmt.name_key(row["full_name"] or "")} | {fmt.name_key(a) for a in row["aliases"]}


def add_alias(contact_id: Any, alias: str) -> None:
    """Remember that meetings call this member *alias* (the manager confirmed it)."""
    row = by_id(contact_id)
    if row is None or knows(row, alias):
        return
    with _connect() as con:
        con.execute("UPDATE contacts SET aliases = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(row["aliases"] + [fmt.clean(alias)], ensure_ascii=False),
                     store.now().isoformat(timespec="seconds"), row["id"]))


def link_telegram(name: str, chat_id: str, username: str = "") -> None:
    """The manager approved this Telegram chat for *name*; a chat belongs to one person only."""
    with _connect() as con:
        con.execute("UPDATE contacts SET telegram_chat_id = NULL WHERE telegram_chat_id = ?", (str(chat_id),))
    row = exact(name)
    fields: Dict[str, Any] = {"telegram_chat_id": str(chat_id)}
    if username:
        fields["telegram_username"] = "@" + username.lstrip("@")
    if row is None or not row["joined_at"] or row["telegram_chat_id"] != str(chat_id):
        fields["joined_at"] = store.now().isoformat(timespec="seconds")
    _upsert(name, **fields)


def registered_elsewhere(name: str, chat_id: str) -> Optional[Dict[str, Any]]:
    """The contact already registered under exactly this name for ANOTHER Telegram chat."""
    row = exact(name)
    return row if row and row["telegram_chat_id"] and row["telegram_chat_id"] != str(chat_id) else None


def set_details(name: str, **fields: str) -> None:
    """Full name / role for an existing person (empty values are ignored)."""
    fields = {k: fmt.clean(v) for k, v in fields.items() if k in ("full_name", "role") and fmt.clean(v)}
    if fields:
        _upsert(name, **fields)


def _in_past_tasks(name: str) -> bool:
    key = fmt.name_key(name)
    with store.connect() as con:
        people = [r["person"] for r in con.execute("SELECT DISTINCT person FROM tasks WHERE person != ''")]
    return any(fmt.name_key(n) == key for person in people for n in fmt.split_owners(person))


def delete(name: str) -> bool:
    """Remove *name* from the book and from every name picker (past tasks keep their text);
    False when the name is neither in the book nor in any task."""
    row = exact(name)
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
    row, target = exact(old), exact(new)
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


# -- routing and display -------------------------------------------------------------------------

class Route(NamedTuple):
    channel: str  # "telegram", or "" when the task can't be sent to this person
    target: str  # the member's chat id
    reason: str  # why not: not_joined / ambiguous
    who: str = ""  # the member's label, when known
    choices: tuple = ()  # the members an ambiguous name may mean


def route(name: str, contact_id: Any = None) -> Route:
    """How a task reaches one owner: their registered member, or why not."""
    chosen = by_id(contact_id) if contact_id else None
    found = [chosen] if is_member(chosen) else members_for(name)
    if len(found) > 1:
        return Route("", "", "ambiguous", choices=tuple(found))
    if not found:
        return Route("", "", "not_joined")
    return Route("telegram", found[0]["telegram_chat_id"], "", label(found[0]))


def member_text(row: Dict[str, Any], lang: str) -> str:
    labels = fmt.LABELS[lang]
    username = row["telegram_username"]
    return labels["registered"].format(username=username) if username else labels["registered_plain"]


def display(task: Dict[str, Any], lang: str) -> str:
    """"✅ @username (مسجل)" / "⏳ لم ينضم بعد — أرسل /invite" per owner ("—" when unclear)."""
    labels = fmt.LABELS[lang]
    names = fmt.split_owners(task["person"])
    if not names:
        return labels["no_owner"]
    chosen = by_id(task["contact_id"]) if len(names) == 1 and task.get("contact_id") else None
    parts = []
    for name in names:
        if fmt.is_manager(name):  # the manager's own task: nobody to reach
            parts.append(f"{name}: {labels['self_task']}" if len(names) > 1 else labels["self_task"])
            continue
        found = [chosen] if is_member(chosen) else members_for(name)
        if len(found) > 1:  # shared first name: the manager chooses in the send preview
            text = "❓ " + " / ".join(label(r) for r in found)
        else:
            text = member_text(found[0], lang) if found else labels["not_joined"]
        parts.append(f"{name}: {text}" if len(names) > 1 else text)
    return "، ".join(parts)


def owner_display(task: Dict[str, Any], lang: str) -> str:
    """Each owner as the manager should read them: a linked registered member as "full name
    (role)", the manager as "(you)", anyone else (not joined, or a name fitting several members)
    by the short name the meeting used. "" for an unclear owner (``brief.owner_text`` handles it)."""
    names = fmt.split_owners(task["person"])
    chosen = by_id(task["contact_id"]) if len(names) == 1 and task.get("contact_id") else None
    shown = []
    for name in names:
        if fmt.is_manager(name):
            shown.append(f"{fmt.LABELS[lang]['manager']} ({fmt.LABELS[lang]['you']})")
            continue
        found = [chosen] if is_member(chosen) else members_for(name)
        shown.append(label(found[0]) if len(found) == 1 else name)
    return "، ".join(shown)


def annotate(tasks: List[Dict[str, Any]], lang: str) -> List[Dict[str, Any]]:
    """Add each task's ``contact_display`` and ``owner_display`` (what cards, summaries and /brief
    show). In place."""
    for task in tasks:
        task["contact_display"] = display(task, lang)
        task["owner_display"] = owner_display(task, lang)
    return tasks


def not_joined(platform: str, chat_id: str) -> List[str]:
    """Owners of confirmed tasks from recent meetings (and book entries) with no registered member."""
    hidden = _hidden()
    names = [r["name"] for r in all_contacts() if not is_member(r)]
    names += [n for person in store.recent_people(platform, chat_id) for n in fmt.split_owners(person)]
    unique: Dict[str, str] = {}
    for name in names:
        key = fmt.name_key(name)
        if is_person_name(name) and key not in hidden and not members_for(name) and not fmt.is_manager(name):
            unique.setdefault(key, fmt.clean(name))
    return list(unique.values())


def team_text() -> str:
    """/team: only people the manager approved and who registered in the bot."""
    rows = members()
    if not rows:
        return "لا يوجد أحد مسجل بعد — استخدم /invite"
    blocks = []
    for row in rows:
        lines = [f"👤 {row['full_name'] or row['name']}"]
        if row["role"]:
            lines.append(f"💼 {row['role']}")
        if row["telegram_username"]:
            lines.append(f"🔗 {row['telegram_username']}")
        lines.append(f"📅 انضم: {(row['joined_at'] or row['updated_at'])[:10]}")
        blocks.append("\n".join(lines))
    return f"👥 الفريق ({len(rows)})\n\n" + "\n\n".join(blocks)
