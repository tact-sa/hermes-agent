"""SQLite storage for recorded meetings: ``<home>/recordings/recordings.db``.

One row per meeting (who sent it, when, transcript path, brief) and one row per extracted task.
``tasks.contact`` is a contact said in the meeting or set by the manager (the contacts book in
``contacts.py`` fills the rest); ``tasks.sent_at`` / ``sent_via`` record when and how a confirmed
task was sent to its owner.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Riyadh")

# Meeting status: transcribing -> analyzing -> pending (tasks await confirmation) -> confirmed; or failed.
# Task status: pending -> confirmed | removed.


def recordings_dir() -> Path:
    from hermes_constants import get_hermes_home
    path = get_hermes_home() / "recordings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def meeting_dir(meeting_id: int) -> Path:
    path = recordings_dir() / str(meeting_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(recordings_dir() / "recordings.db")
    con.row_factory = sqlite3.Row
    con.execute(
        "CREATE TABLE IF NOT EXISTS meetings ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " platform TEXT NOT NULL, chat_id TEXT NOT NULL, user_id TEXT NOT NULL,"
        " created_at TEXT NOT NULL,"
        " status TEXT NOT NULL DEFAULT 'transcribing',"
        " language TEXT, transcript_path TEXT, brief TEXT, error TEXT, note TEXT,"
        " confirm_message_id TEXT)")
    con.execute(
        "CREATE TABLE IF NOT EXISTS tasks ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " meeting_id INTEGER NOT NULL REFERENCES meetings(id),"
        " position INTEGER NOT NULL,"
        " person TEXT NOT NULL DEFAULT '', person_key TEXT NOT NULL DEFAULT '',"
        " candidates TEXT NOT NULL DEFAULT '[]',"
        " task TEXT NOT NULL, deadline TEXT NOT NULL DEFAULT '',"
        " status TEXT NOT NULL DEFAULT 'pending',"
        " confirmed_at TEXT, sent_at TEXT, message_id TEXT,"
        " UNIQUE (meeting_id, position))")
    # Columns added after the first release, added in place to existing databases: the caption note
    # (feeds "/brief <id> retry") and the Telegram message ids of each task card and of the
    # "Confirm all" message (edited in place when a task changes).
    for table, column in (("meetings", "note"), ("meetings", "confirm_message_id"), ("meetings", "manager_name"),
                          ("tasks", "message_id"), ("tasks", "contact"), ("tasks", "sent_via"),
                          ("tasks", "contact_id")):
        if column not in {row[1] for row in con.execute(f"PRAGMA table_info({table})")}:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
    return con


def now() -> datetime:
    return datetime.now(TZ)


def create_meeting(platform: str, chat_id: str, user_id: str, note: str = "", manager_name: str = "") -> int:
    with connect() as con:
        return con.execute(
            "INSERT INTO meetings (platform, chat_id, user_id, created_at, note, manager_name)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (platform, chat_id, user_id, now().isoformat(timespec="seconds"), note, manager_name)).lastrowid


def update_meeting(meeting_id: int, **fields: Any) -> None:
    cols = ", ".join(f"{k} = ?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE meetings SET {cols} WHERE id = ?", (*fields.values(), meeting_id))


def save_analysis(meeting_id: int, language: str, brief: Dict[str, Any], tasks: List[Dict[str, Any]]) -> None:
    with connect() as con:
        con.execute("UPDATE meetings SET language = ?, brief = ?, status = ? WHERE id = ?",
                    (language, json.dumps(brief, ensure_ascii=False),
                     "pending" if tasks else "confirmed", meeting_id))
        con.executemany(
            "INSERT INTO tasks (meeting_id, position, person, person_key, candidates, task, deadline, contact)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(meeting_id, i, t["person"], t["person_key"], json.dumps(t["candidates"], ensure_ascii=False),
              t["task"], t["deadline"], t.get("contact") or "") for i, t in enumerate(tasks, 1)])


def get_meeting(meeting_id: int, platform: str, chat_id: str) -> Optional[sqlite3.Row]:
    with connect() as con:
        return con.execute("SELECT * FROM meetings WHERE id = ? AND platform = ? AND chat_id = ?",
                           (meeting_id, platform, chat_id)).fetchone()


def latest_pending(platform: str, chat_id: str) -> Optional[sqlite3.Row]:
    with connect() as con:
        return con.execute(
            "SELECT * FROM meetings WHERE platform = ? AND chat_id = ? AND status = 'pending'"
            " ORDER BY id DESC LIMIT 1", (platform, chat_id)).fetchone()


def latest_briefed(platform: str, chat_id: str) -> Optional[sqlite3.Row]:
    """The chat's latest meeting with a brief, confirmed or not (contact edits after confirmation)."""
    with connect() as con:
        return con.execute(
            "SELECT * FROM meetings WHERE platform = ? AND chat_id = ? AND brief IS NOT NULL"
            " ORDER BY id DESC LIMIT 1", (platform, chat_id)).fetchone()


def recent_people(platform: str, chat_id: str, meetings: int = 10) -> List[str]:
    """Owners of CONFIRMED tasks in the chat's latest meetings, most recent first. Unclear owners
    (candidates), pending and cancelled tasks are left out; joint owners stay joined ("أحمد، عمر")."""
    with connect() as con:
        rows = con.execute(
            "SELECT t.person FROM tasks t JOIN meetings m ON m.id = t.meeting_id"
            " WHERE m.platform = ? AND m.chat_id = ? AND t.status = 'confirmed' AND t.person != ''"
            " AND m.id IN (SELECT id FROM meetings WHERE platform = ? AND chat_id = ? ORDER BY id DESC LIMIT ?)"
            " ORDER BY m.id DESC, t.position", (platform, chat_id, platform, chat_id, meetings)).fetchall()
    return [r["person"] for r in rows]


def delete_meeting(meeting_id: int) -> None:
    """The meeting, its tasks and its folder (transcript, any leftover audio)."""
    with connect() as con:
        con.execute("DELETE FROM tasks WHERE meeting_id = ?", (meeting_id,))
        con.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    shutil.rmtree(recordings_dir() / str(meeting_id), ignore_errors=True)


def meeting_ids(platform: str, chat_id: str, user_id: str) -> List[int]:
    with connect() as con:
        return [r["id"] for r in con.execute(
            "SELECT id FROM meetings WHERE platform = ? AND chat_id = ? AND user_id = ? ORDER BY id",
            (platform, chat_id, str(user_id))).fetchall()]


def rename_person(old_key: str, new_name: str, key_of, split) -> int:
    """Rename one person in every task's owner (joint owners included); returns tasks changed.
    ``key_of`` / ``split`` are ``brief.name_key`` / ``brief.split_owners``."""
    changed = 0
    with connect() as con:
        for row in con.execute("SELECT id, person FROM tasks WHERE person != ''").fetchall():
            names = split(row["person"])
            if not any(key_of(n) == old_key for n in names):
                continue
            person = "، ".join(dict.fromkeys(new_name if key_of(n) == old_key else n for n in names))
            con.execute("UPDATE tasks SET person = ?, person_key = ? WHERE id = ?", (person, key_of(person), row["id"]))
            changed += 1
    return changed


def recent_meetings(platform: str, chat_id: str, limit: int = 10) -> List[sqlite3.Row]:
    with connect() as con:
        return con.execute(
            "SELECT m.*,"
            " (SELECT COUNT(*) FROM tasks t WHERE t.meeting_id = m.id) AS n_tasks,"
            " (SELECT COUNT(*) FROM tasks t WHERE t.meeting_id = m.id AND t.status = 'confirmed') AS n_confirmed,"
            " (SELECT COUNT(*) FROM tasks t WHERE t.meeting_id = m.id AND t.status = 'pending') AS n_pending"
            " FROM meetings m WHERE platform = ? AND chat_id = ? ORDER BY id DESC LIMIT ?",
            (platform, chat_id, limit)).fetchall()


def tasks_for(meeting_id: int) -> List[Dict[str, Any]]:
    with connect() as con:
        rows = con.execute("SELECT * FROM tasks WHERE meeting_id = ? ORDER BY position",
                           (meeting_id,)).fetchall()
    return [{**dict(r), "candidates": json.loads(r["candidates"] or "[]")} for r in rows]


def set_task_status(meeting_id: int, positions: List[int], status: str) -> None:
    stamp = now().isoformat(timespec="seconds") if status == "confirmed" else None
    with connect() as con:
        con.executemany("UPDATE tasks SET status = ?, confirmed_at = ? WHERE meeting_id = ? AND position = ?",
                        [(status, stamp, meeting_id, p) for p in positions])


def set_task_message(meeting_id: int, position: int, message_id: Any) -> None:
    with connect() as con:
        con.execute("UPDATE tasks SET message_id = ? WHERE meeting_id = ? AND position = ?",
                    (str(message_id), meeting_id, position))


def set_task_contact_id(meeting_id: int, position: int, contact_id: int) -> None:
    """The manager's choice of which contact a shared first name means for this task."""
    with connect() as con:
        con.execute("UPDATE tasks SET contact_id = ? WHERE meeting_id = ? AND position = ?",
                    (str(contact_id), meeting_id, position))


def mark_sent(meeting_id: int, position: int, channels: List[str]) -> None:
    stamp = now().isoformat(timespec="seconds")
    with connect() as con:
        con.execute("UPDATE tasks SET sent_at = ?, sent_via = ? WHERE meeting_id = ? AND position = ?",
                    (stamp, ",".join(dict.fromkeys(channels)), meeting_id, position))


def update_task(meeting_id: int, position: int, **fields: Any) -> None:
    """Edit a task's fields; its status stays as it is (a confirmed task edited stays confirmed)."""
    cols = ", ".join(f"{k} = ?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE tasks SET {cols} WHERE meeting_id = ? AND position = ?",
                    (*fields.values(), meeting_id, position))
