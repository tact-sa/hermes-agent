"""/team: the people panel. Every person is a button; one message is edited in place as the manager
moves between the list, a person's details, a field prompt and the remove question.

Joined members come first, then people seen in meetings who have not joined. The logic is the
contacts book's own (``contacts.rename`` / ``set_details`` / ``delete``); this module only maps
buttons to it. Buttons go through ``ui`` (rows of ``(label, callback_id)``), never a messaging
library. Only the manager can use it: a tap from anyone else does nothing, and typed values are
read only in the chat and from the user who asked for them, within ``INPUT_TTL_SECONDS``.

Callback ids: ``rec:t:<action>:<ref>`` where ref is ``c<contact id>``, ``n<hash of the name>`` (seen
in meetings only, no book row yet) or ``0`` (no person).
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from . import brief as fmt
from . import contacts
from . import invite
from . import ui

INPUT_TTL_SECONDS = 5 * 60
PERSON_LABEL_MAX = 30  # people sit one per row, so they get more room than action buttons
CALLBACK_RE = re.compile(r"^rec:t:([a-z]):(0|c\d+|n[0-9a-f]{10})$")

# field button -> (what the prompt calls it, key passed to contacts); the short name is a rename
FIELDS = {"s": ("name", "الاسم المختصر"), "f": ("full_name", "الاسم الكامل"), "r": ("role", "الوظيفة")}
# (platform, chat) -> (ref, field, panel message id, manager user id, expiry)
_awaiting: Dict[Tuple[str, str], Tuple[str, str, Any, str, float]] = {}

LIST_TITLE = "👥 الفريق ({n})\nاختر شخصاً:"
EMPTY = "👥 لا يوجد أحد في الفريق بعد."
GONE = "لم يعد هذا الشخص موجوداً."
INVITE_BUTTON = "➕ دعوة شخص جديد"
BACK = "↩️ رجوع"


class Person(NamedTuple):
    ref: str
    name: str
    row: Optional[Dict[str, Any]]
    joined: bool


def _ref(name: str, row: Optional[Dict[str, Any]]) -> str:
    return f"c{row['id']}" if row else "n" + hashlib.sha1(fmt.name_key(name).encode("utf-8")).hexdigest()[:10]


def people(platform: str, chat_id: str) -> List[Person]:
    """Joined members, then the people from meetings who have not joined."""
    found = [Person(f"c{r['id']}", r["name"], r, True) for r in contacts.members()]
    for name in contacts.not_joined(platform, chat_id):
        row = contacts.exact(name)
        found.append(Person(_ref(name, row), row["name"] if row else name, row, False))
    return found


def _lookup(ref: str, platform: str, chat_id: str, name: str = "") -> Optional[Person]:
    key = fmt.name_key(name) if name else ""
    return next((p for p in people(platform, chat_id) if p.ref == ref or (key and fmt.name_key(p.name) == key)), None)


def person_label(person: Person) -> str:
    if person.joined:
        who = person.row["full_name"] or person.name
        return f"👤 {who} — {person.row['role']}" if person.row["role"] else f"👤 {who}"
    return f"⏳ {ui.short(person.name, 16)} — لم ينضم"


def _cb(action: str, ref: str = "0") -> str:
    return f"rec:t:{action}:{ref}"


# -- views: (text, button rows) ------------------------------------------------------------------

def list_view(platform: str, chat_id: str, note: str = "") -> Tuple[str, List[List[Tuple[str, str]]]]:
    found = people(platform, chat_id)
    rows = [[(person_label(p), _cb("o", p.ref))] for p in found]
    rows.append([(INVITE_BUTTON, _cb("i"))])
    text = LIST_TITLE.format(n=len(found)) if found else EMPTY
    return (f"{note}\n\n{text}" if note else text), rows


def details_view(person: Person, note: str = "") -> Tuple[str, List[List[Tuple[str, str]]]]:
    if person.joined:
        row = person.row
        joined = (row["joined_at"] or row["updated_at"])[:10]
        text = (f"👤 {row['full_name'] or person.name}\n"
                f"الاسم المختصر: {person.name}\n"
                f"💼 الوظيفة: {row['role'] or '—'}\n"
                f"🔗 {row['telegram_username'] or '—'}\n"
                f"📅 انضم: {joined}")
        rows = [[("✏️ الاسم المختصر", _cb("s", person.ref)), ("✏️ الاسم الكامل", _cb("f", person.ref)),
                 ("💼 الوظيفة", _cb("r", person.ref))],
                [("🗑️ إزالة", _cb("d", person.ref)), (BACK, _cb("l"))]]
    else:
        text = f"⏳ {person.name}\nلم ينضم بعد — ظهر في الاجتماعات فقط."
        rows = [[("✏️ الاسم المختصر", _cb("s", person.ref)), ("🗑️ إزالة", _cb("d", person.ref)),
                 (BACK, _cb("l"))],
                [("➕ دعوة", _cb("i", person.ref))]]
    return (f"{note}\n\n{text}" if note else text), rows


def _prompt_view(person: Person, field: str, note: str = ""):
    title = next(t for key, t in FIELDS.values() if key == field)
    text = f"أرسل {title} الجديد لـ {person.name} (خلال 5 دقائق)"
    return (f"{note}\n\n{text}" if note else text), [[("❌ إلغاء", _cb("x", person.ref))]]


def _remove_view(person: Person):
    return (f"إزالة {person.name}؟ لن يستقبل مهام بعد الآن",
            [[("✅ نعم", _cb("y", person.ref)), ("❌ لا", _cb("n", person.ref))]])


# -- sending / buttons / typed values --------------------------------------------------------------

async def _show(adapter: Any, chat_id: str, message_id: Any, view: Tuple[str, list]) -> Optional[str]:
    """Edit the panel message in place; a new one only when it can't be edited."""
    text, rows = view
    if message_id and await ui.edit(adapter, chat_id, message_id, text, rows, PERSON_LABEL_MAX):
        return str(message_id)
    return await ui.send(adapter, chat_id, text, rows, PERSON_LABEL_MAX)


async def show_list(adapter: Any, platform: str, chat_id: str) -> None:
    """/team: one message with a button per person."""
    _awaiting.pop((platform, chat_id), None)
    text, rows = list_view(platform, chat_id)
    await ui.send(adapter, chat_id, text, rows, PERSON_LABEL_MAX)


async def handle_button(adapter: Any, data: str, platform: str, chat_id: str, user_id: str,
                        message_id: Any) -> None:
    """A ``rec:t:`` tap. Anyone but the manager gets nothing."""
    m = CALLBACK_RE.match(data or "")
    if not m or not ui.is_manager(adapter, user_id, chat_id):
        return
    action, ref = m.groups()
    key = (platform, str(chat_id))
    if action == "i":  # a new invite link needs no person
        await ui.send(adapter, chat_id, invite.link_message(adapter, platform, chat_id, user_id))
        return
    if action == "l":
        _awaiting.pop(key, None)
        await _show(adapter, chat_id, message_id, list_view(platform, chat_id))
        return
    person = _lookup(ref, platform, chat_id) if ref != "0" else None
    if person is None:
        await _show(adapter, chat_id, message_id, list_view(platform, chat_id, GONE))
        return
    if action in FIELDS:
        field = FIELDS[action][0]
        if field != "name" and not person.joined:  # only a short name for someone who has not joined
            await _show(adapter, chat_id, message_id, details_view(person))
            return
        _awaiting[key] = (person.ref, field, message_id, str(user_id), time.monotonic() + INPUT_TTL_SECONDS)
        await _show(adapter, chat_id, message_id, _prompt_view(person, field))
    elif action == "d":
        await _show(adapter, chat_id, message_id, _remove_view(person))
    elif action == "y":
        contacts.delete(person.name)
        await _show(adapter, chat_id, message_id, list_view(platform, chat_id, f"🗑️ أُزيل {person.name}."))
    else:  # "x" cancel an input, "n" keep, "o" open
        _awaiting.pop(key, None)
        await _show(adapter, chat_id, message_id, details_view(person))


async def take_input(adapter: Any, platform: str, chat_id: str, user_id: str, text: str) -> bool:
    """The manager's typed value after a field button: saved, then the details card is shown again.
    False when nothing is awaited from this user in this chat (or the wait expired)."""
    key = (platform, str(chat_id))
    waiting = _awaiting.pop(key, None)
    if waiting is None or waiting[4] <= time.monotonic() or waiting[3] != str(user_id):
        return False
    ref, field, message_id = waiting[0], waiting[1], waiting[2]
    person = _lookup(ref, platform, chat_id)
    if person is None:
        await _show(adapter, chat_id, message_id, list_view(platform, chat_id, GONE))
        return True
    if field == "name":
        done, note = contacts.rename(person.name, text)
    else:
        contacts.set_details(person.name, **{field: text})
        done, note = True, "✅ حُفظ."
    if not done:  # duplicate or invalid short name: say why and keep waiting
        _awaiting[key] = (ref, field, message_id, str(user_id), time.monotonic() + INPUT_TTL_SECONDS)
        await _show(adapter, chat_id, message_id, _prompt_view(person, field, note))
        return True
    person = _lookup(ref, platform, chat_id, text if field == "name" else "")
    await _show(adapter, chat_id, message_id,
                details_view(person, note) if person else list_view(platform, chat_id, note))
    return True
