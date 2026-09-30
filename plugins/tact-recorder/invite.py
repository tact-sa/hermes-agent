"""Telegram invite links so team members can receive their tasks — only with the manager's approval.

``/invite`` (manager) creates ``t.me/<bot>?start=join_<token>``: the token is
``secrets.token_urlsafe(18)`` (18 random bytes), only its SHA-256 is stored, and it expires after
``INVITE_TTL_SECONDS``. ``/invite revoke`` cancels every active link of that manager.

Someone opening the link sends ``/start join_<token>``. That message is caught by a Telegram handler
scoped to exactly that text in private chats (``JOIN_RE``), ahead of the adapter's allowlist
prefilter, which would otherwise drop it before any plugin hook sees it. Nothing is registered at
this point: the person is told to wait and the manager gets their display name, @username and id
with one button per known person, "another name" and "reject" (plus their profile photo, if any).
Only the manager's pick saves the chat id to that person in the contacts book; a Telegram display
name is never matched on its own, and a short name already registered to someone else is refused.
The manager is then asked the person's full name and role (each skippable).
Invalid, expired or revoked tokens get one fixed reply and nothing else; join attempts are limited
per Telegram user. Every other message from a non-allowlisted user takes the unchanged gateway
path and is dropped exactly as before: joining grants no access to the agent, tools or commands.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

from . import brief as fmt
from . import contacts
from . import store

logger = logging.getLogger(__name__)

INVITE_TTL_SECONDS = 30 * 60
JOIN_LIMIT, JOIN_WINDOW_SECONDS = 5, 3600
MAX_NAME_BUTTONS = 12
NAME_TTL_SECONDS = 15 * 60
JOIN_RE = re.compile(r"^/start(?:@\w+)?\s+join_([A-Za-z0-9_-]{16,64})\s*$")

WAITING = "⏳ طلبك بانتظار موافقة المدير."
REGISTERED = "✅ تم تسجيلك لاستقبال المهام."
ALREADY = "✅ أنت مسجّل بالفعل لاستقبال المهام."
REJECTED = "❌ لم تتم الموافقة على طلبك."
INVALID = "⛔ الرابط منتهي أو غير صالح. اطلب رابطاً جديداً من المدير."

_attempts: Dict[str, Deque[float]] = defaultdict(deque)
# (platform, manager chat) -> (join request id, expiry) after "➕ اسم آخر"
awaiting_name: Dict[Tuple[str, str], Tuple[int, float]] = {}
# (platform, manager chat) -> (join request id, "full_name" | "role", expiry) after an approval
awaiting_detail: Dict[Tuple[str, str], Tuple[int, str, float]] = {}
DETAIL_QUESTIONS = {"full_name": "الاسم الكامل؟", "role": "الوظيفة / القسم؟"}


def _now() -> float:
    return time.time()


def chat_arg(chat_id: Any) -> Any:
    return int(chat_id) if str(chat_id).lstrip("-").isdigit() else chat_id


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _connect():
    con = store.connect()
    con.execute(
        "CREATE TABLE IF NOT EXISTS invites ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, token_hash TEXT NOT NULL UNIQUE,"
        " creator_platform TEXT NOT NULL, creator_chat_id TEXT NOT NULL, creator_user_id TEXT NOT NULL,"
        " created_at REAL NOT NULL, expires_at REAL NOT NULL, revoked INTEGER NOT NULL DEFAULT 0)")
    con.execute(
        "CREATE TABLE IF NOT EXISTS join_requests ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, invite_id INTEGER NOT NULL,"
        " user_chat_id TEXT NOT NULL, user_id TEXT NOT NULL, username TEXT, display_name TEXT,"
        " options TEXT NOT NULL DEFAULT '[]', status TEXT NOT NULL DEFAULT 'pending',"
        " manager_message_id TEXT, created_at REAL NOT NULL, linked_name TEXT)")
    if "linked_name" not in {row[1] for row in con.execute("PRAGMA table_info(join_requests)")}:
        con.execute("ALTER TABLE join_requests ADD COLUMN linked_name TEXT")
    return con


# -- manager side --------------------------------------------------------------------------------

def create_invite(platform: str, chat_id: str, user_id: str) -> Tuple[str, float]:
    token = secrets.token_urlsafe(18)
    now = _now()
    with _connect() as con:
        con.execute("INSERT INTO invites (token_hash, creator_platform, creator_chat_id, creator_user_id,"
                    " created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (_hash(token), platform, str(chat_id), str(user_id), now, now + INVITE_TTL_SECONDS))
    logger.info("tact-recorder: invite link created (expires in %d min)", INVITE_TTL_SECONDS // 60)
    return token, now + INVITE_TTL_SECONDS


def revoke_invites(user_id: str) -> int:
    with _connect() as con:
        count = con.execute("UPDATE invites SET revoked = 1 WHERE creator_user_id = ? AND revoked = 0"
                            " AND expires_at > ?", (str(user_id), _now())).rowcount
    logger.info("tact-recorder: %d invite link(s) revoked", count)
    return count


def invite_text(bot_username: str, token: str, expires_at: float) -> str:
    until = datetime.fromtimestamp(expires_at, store.TZ).strftime("%H:%M")
    return (f"🔗 رابط الدعوة (صالح 30 دقيقة، حتى {until}):\n"
            f"https://t.me/{bot_username}?start=join_{token}\n"
            "أرسله للشخص. لن يُسجَّل إلا بعد موافقتك، ولن يستطيع استخدام البوت إلا لاستقبال مهامه.\n"
            "لإلغاء كل الروابط: /invite revoke")


# -- joining ---------------------------------------------------------------------------------------

def _rate_limited(user_id: str) -> bool:
    now = _now()
    attempts = _attempts[str(user_id)]
    while attempts and attempts[0] <= now - JOIN_WINDOW_SECONDS:
        attempts.popleft()
    if len(attempts) >= JOIN_LIMIT:
        return True
    attempts.append(now)
    return False


def _valid_invite(token: str) -> Optional[Any]:
    with _connect() as con:
        return con.execute("SELECT * FROM invites WHERE token_hash = ? AND revoked = 0 AND expires_at > ?",
                           (_hash(token), _now())).fetchone()


def _name_options(invite: Any) -> List[str]:
    return contacts.picker_names(invite["creator_platform"], invite["creator_chat_id"], MAX_NAME_BUTTONS)


def _request_markup(request_id: int, options: List[str]) -> Any:
    from telegram import InlineKeyboardButton as Button, InlineKeyboardMarkup
    rows = [[Button(contacts.label_for(name), callback_data=f"rec:j:a:{request_id}:{i}")
             for i, name in enumerate(options[k:k + 2], k)] for k in range(0, len(options), 2)]
    rows.append([Button("➕ اسم آخر", callback_data=f"rec:j:o:{request_id}"),
                 Button("❌ رفض", callback_data=f"rec:j:r:{request_id}")])
    return InlineKeyboardMarkup(rows)


def _clock(epoch: float, with_date: bool = False) -> str:
    return datetime.fromtimestamp(epoch, store.TZ).strftime("%Y-%m-%d %H:%M" if with_date else "%H:%M")


def request_text(request: Any, invite_created: float) -> str:
    """Everything Telegram tells us about the person asking to join."""
    username = f"@{request['username']}" if request["username"] else "بدون اسم مستخدم"
    return ("📥 طلب انضمام\n"
            f"👤 الاسم في تيليجرام: {request['display_name'] or '—'}\n"
            f"🔗 {username}\n"
            f"🆔 {request['user_id']}\n"
            f"🕒 {_clock(request['created_at'], True)} — عبر رابط أنشأته الساعة {_clock(invite_created)}\n"
            "اربطه بـ:")


async def _send_profile_photo(bot: Any, manager_chat: str, user_id: str) -> None:
    """The person's Telegram profile photo, if they have one; any failure is skipped silently."""
    try:
        photos = await bot.get_user_profile_photos(user_id=int(user_id), limit=1)
        sizes = photos.photos[0] if photos and photos.photos else []
        if sizes:
            await bot.send_photo(chat_id=chat_arg(manager_chat), photo=sizes[-1].file_id)
    except Exception as exc:
        logger.debug("tact-recorder: no profile photo for a join request: %s", type(exc).__name__)


def _who(request: Any) -> str:
    username = f"@{request['username']}" if request["username"] else "بدون اسم مستخدم"
    return f"{request['display_name'] or '—'} ({username}, {request['user_id']})"


async def handle_join(bot: Any, chat_id: str, user_id: str, username: str, display_name: str, text: str) -> None:
    """A private ``/start join_<token>`` message; nothing is registered here."""
    m = JOIN_RE.match(text or "")
    if not m or _rate_limited(user_id):
        return
    invite = _valid_invite(m.group(1))
    if invite is None:
        logger.info("tact-recorder: join attempt with an invalid, expired or revoked link")
        await bot.send_message(chat_id=chat_arg(chat_id), text=INVALID)
        return
    if contacts.by_chat_id(chat_id):
        await bot.send_message(chat_id=chat_arg(chat_id), text=ALREADY)
        return
    with _connect() as con:
        pending = con.execute("SELECT id FROM join_requests WHERE user_id = ? AND status = 'pending'",
                              (str(user_id),)).fetchone()
        if pending is None:
            options = _name_options(invite)
            request_id = con.execute(
                "INSERT INTO join_requests (invite_id, user_chat_id, user_id, username, display_name, options,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (invite["id"], str(chat_id), str(user_id), username or "", display_name or "",
                 json.dumps(options, ensure_ascii=False), _now())).lastrowid
    await bot.send_message(chat_id=chat_arg(chat_id), text=WAITING)
    if pending is not None:
        return  # the manager already has this request
    request = get_request(request_id)
    await _send_profile_photo(bot, invite["creator_chat_id"], user_id)
    msg = await bot.send_message(
        chat_id=chat_arg(invite["creator_chat_id"]), text=request_text(request, invite["created_at"]),
        reply_markup=_request_markup(request_id, options))
    with _connect() as con:
        con.execute("UPDATE join_requests SET manager_message_id = ? WHERE id = ?", (str(msg.message_id), request_id))
    logger.info("tact-recorder: join request #%s sent to the manager for approval", request_id)


def get_request(request_id: int) -> Optional[Any]:
    with _connect() as con:
        return con.execute(
            "SELECT r.*, i.creator_platform, i.creator_chat_id, i.creator_user_id FROM join_requests r"
            " JOIN invites i ON i.id = r.invite_id WHERE r.id = ?", (request_id,)).fetchone()


def _may_decide(request: Any, chat_id: str, user_id: str) -> bool:
    return (request is not None and request["status"] == "pending"
            and request["creator_user_id"] == str(user_id) and request["creator_chat_id"] == str(chat_id))


async def _close(bot: Any, request: Any, status: str, text: str) -> None:
    with _connect() as con:
        con.execute("UPDATE join_requests SET status = ? WHERE id = ?", (status, request["id"]))
    if request["manager_message_id"]:
        try:
            await bot.edit_message_text(chat_id=chat_arg(request["creator_chat_id"]),
                                        message_id=int(request["manager_message_id"]),
                                        text=text, reply_markup=None)
        except Exception as exc:
            logger.warning("tact-recorder: updating the join request message failed: %s", type(exc).__name__)


async def approve(bot: Any, request: Any, name: str, platform: str = "telegram") -> List[str]:
    """Link the request to *name*, then ask its full name and role; messages for the manager."""
    taken = contacts.registered_elsewhere(name, request["user_chat_id"])
    if taken is not None:  # never move another person's Telegram link onto this request
        awaiting_name[(platform, request["creator_chat_id"])] = (request["id"], _now() + NAME_TTL_SECONDS)
        return [f"«{name}» مسجّل بالفعل لـ {contacts.label(taken)}. اكتب اسماً مختصراً آخر لهذا الشخص "
                "(مثلاً الاسم الأول وحرف من العائلة)."]
    contacts.link_telegram(name, request["user_chat_id"], request["username"] or "")
    with _connect() as con:
        con.execute("UPDATE join_requests SET linked_name = ? WHERE id = ?", (name, request["id"]))
    await _close(bot, request, "approved", f"✅ تم ربط {_who(request)} بـ {name}.")
    await bot.send_message(chat_id=chat_arg(request["user_chat_id"]), text=REGISTERED)
    logger.info("tact-recorder: join request #%s approved", request["id"])
    await _ask_detail(bot, platform, request, "full_name")
    return []


async def _ask_detail(bot: Any, platform: str, request: Any, field: str) -> None:
    """Ask the next missing detail (full name, then role) with [تخطي]; nothing when both are known."""
    from telegram import InlineKeyboardButton as Button, InlineKeyboardMarkup
    fields = ("full_name", "role")
    name = request["linked_name"] or get_request(request["id"])["linked_name"]
    row = contacts.exact(name) or {}
    for current in fields[fields.index(field):]:
        if not row.get(current):
            awaiting_detail[(platform, request["creator_chat_id"])] = (request["id"], current,
                                                                        _now() + NAME_TTL_SECONDS)
            await bot.send_message(chat_id=chat_arg(request["creator_chat_id"]),
                                   text=f"{DETAIL_QUESTIONS[current]} ({name})",
                                   reply_markup=InlineKeyboardMarkup([[Button(
                                       "تخطي", callback_data=f"rec:j:k:{request['id']}")]]))
            return
    awaiting_detail.pop((platform, request["creator_chat_id"]), None)
    await bot.send_message(chat_id=chat_arg(request["creator_chat_id"]),
                           text=f"✅ اكتمل تسجيل {contacts.label_for(name)}.")


async def _next_detail(bot: Any, platform: str, request: Any, answered: str) -> None:
    if answered == "full_name":
        await _ask_detail(bot, platform, request, "role")
    else:
        awaiting_detail.pop((platform, request["creator_chat_id"]), None)
        await bot.send_message(chat_id=chat_arg(request["creator_chat_id"]),
                               text=f"✅ اكتمل تسجيل {contacts.label_for(request['linked_name'])}.")


async def handle_button(bot: Any, action: str, request_id: int, option: int, chat_id: str, user_id: str,
                        platform: str = "telegram") -> Tuple[List[str], str]:
    """``(messages to the manager, toast)`` for a ``rec:j:*`` button; only the invite's creator decides."""
    request = get_request(request_id)
    if action == "k":  # [تخطي] a detail question after an approval
        waiting = awaiting_detail.get((platform, str(chat_id)))
        if request is None or request["creator_user_id"] != str(user_id) or not waiting or waiting[0] != request_id:
            return [], "✔️"
        await _next_detail(bot, platform, request, waiting[1])
        return [], ""
    if not _may_decide(request, chat_id, user_id):
        return [], "Not allowed." if request is None or request["creator_user_id"] != str(user_id) else "✔️"
    if action == "a":
        options = json.loads(request["options"] or "[]")
        if not 0 <= option < len(options):
            return [], "✔️"
        return await approve(bot, request, options[option], platform), ""
    if action == "o":
        awaiting_detail.pop((platform, str(chat_id)), None)  # the typed text is this name, not a detail
        awaiting_name[(platform, str(chat_id))] = (request_id, _now() + NAME_TTL_SECONDS)
        return [f"اكتب الاسم الذي تريد ربط {_who(request)} به."], ""
    await _close(bot, request, "rejected", f"❌ تم رفض طلب {_who(request)}.")
    await bot.send_message(chat_id=chat_arg(request["user_chat_id"]), text=REJECTED)
    logger.info("tact-recorder: join request #%s rejected", request_id)
    return [], ""


async def take_typed_name(bot: Any, platform: str, chat_id: str, user_id: str, text: str) -> Optional[List[str]]:
    """The manager's typed name after "➕ اسم آخر", or a typed full name / role after an approval;
    None when nothing is awaited in this chat."""
    detail = awaiting_detail.get((platform, str(chat_id)))
    if detail is not None and detail[2] > _now():
        request = get_request(detail[0])
        if request is not None and request["creator_user_id"] == str(user_id) and fmt.clean(text):
            contacts.set_details(request["linked_name"], **{detail[1]: text})
            await _next_detail(bot, platform, request, detail[1])
            return []
    waiting = awaiting_name.pop((platform, str(chat_id)), None)
    if waiting is None or waiting[1] <= _now():
        return None
    request = get_request(waiting[0])
    name = fmt.clean(text)
    if not _may_decide(request, chat_id, user_id):
        return ["لم يتم الربط."]
    if not contacts.is_person_name(name):  # a placeholder like "احمد او عمر" is not a person
        awaiting_name[(platform, str(chat_id))] = waiting
        return [f"«{name}» ليس اسم شخص. اكتب اسماً واحداً (ثلاث كلمات على الأكثر)."]
    return await approve(bot, request, name, platform)
