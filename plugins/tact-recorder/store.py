"""SQLite storage for recorded meetings: ``<home>/recordings/recordings.db``.

One row per meeting (who sent it, when, transcript path, brief) and one row per extracted task.
``tasks.person_key`` (normalised name) and ``tasks.sent_at`` are there so "send each person their
tasks" can be added later without a migration; nothing reads them yet.
"""

from __future__ import annotations

import json
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
        " language TEXT, transcript_path TEXT, brief TEXT, error TEXT, note TEXT)")
    # Databases created before the caption note was kept (it feeds "/brief <id> retry").
    if "note" not in {row[1] for row in con.execute("PRAGMA table_info(meetings)")}:
        con.execute("ALTER TABLE meetings ADD COLUMN note TEXT")
    con.execute(
        "CREATE TABLE IF NOT EXISTS tasks ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " meeting_id INTEGER NOT NULL REFERENCES meetings(id),"
        " position INTEGER NOT NULL,"
        " person TEXT NOT NULL DEFAULT '', person_key TEXT NOT NULL DEFAULT '',"
        " candidates TEXT NOT NULL DEFAULT '[]',"
        " task TEXT NOT NULL, deadline TEXT NOT NULL DEFAULT '',"
        " status TEXT NOT NULL DEFAULT 'pending',"
        " confirmed_at TEXT, sent_at TEXT,"
        " UNIQUE (meeting_id, position))")
    return con


def now() -> datetime:
    return datetime.now(TZ)


def create_meeting(platform: str, chat_id: str, user_id: str, note: str = "") -> int:
    with connect() as con:
        return con.execute(
            "INSERT INTO meetings (platform, chat_id, user_id, created_at, note) VALUES (?, ?, ?, ?, ?)",
            (platform, chat_id, user_id, now().isoformat(timespec="seconds"), note)).lastrowid


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
            "INSERT INTO tasks (meeting_id, position, person, person_key, candidates, task, deadline)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(meeting_id, i, t["person"], t["person_key"], json.dumps(t["candidates"], ensure_ascii=False),
              t["task"], t["deadline"]) for i, t in enumerate(tasks, 1)])


def get_meeting(meeting_id: int, platform: str, chat_id: str) -> Optional[sqlite3.Row]:
    with connect() as con:
        return con.execute("SELECT * FROM meetings WHERE id = ? AND platform = ? AND chat_id = ?",
                           (meeting_id, platform, chat_id)).fetchone()


def latest_pending(platform: str, chat_id: str) -> Optional[sqlite3.Row]:
    with connect() as con:
        return con.execute(
            "SELECT * FROM meetings WHERE platform = ? AND chat_id = ? AND status = 'pending'"
            " ORDER BY id DESC LIMIT 1", (platform, chat_id)).fetchone()


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


def update_task(meeting_id: int, position: int, **fields: Any) -> None:
    fields["status"] = "pending"
    fields["confirmed_at"] = None
    cols = ", ".join(f"{k} = ?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE tasks SET {cols} WHERE meeting_id = ? AND position = ?",
                    (*fields.values(), meeting_id, position))
