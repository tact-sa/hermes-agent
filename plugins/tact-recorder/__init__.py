"""tact-recorder — meeting recordings -> transcript, brief and a task list the manager confirms.

Without this plugin a recording reaches the agent the way ``gateway/run_inbound.py`` routes media:
an audio file or audio document (``MessageType.AUDIO`` / ``DOCUMENT``) becomes a "saved at: <path>"
note that nothing transcribes, and a voice note is transcribed into a normal chat turn.
``pre_gateway_dispatch`` sees the event before that routing and takes it when:

- the chat is private and the sender passes the gateway's own allowlist
  (``_is_user_authorized_for_source``, the check the gateway runs right after this hook), and
- it carries a recording: an audio attachment of any kind, a video file sent as a document
  (mp4), or a voice note of at least ``VOICE_MIN_SECONDS`` (shorter ones stay ordinary messages).

``/minutes`` is the explicit trigger: after a bare ``/minutes`` the next audio or voice message in
that chat within ``MINUTES_TTL_SECONDS`` is a recording whatever its length, and so is a file sent
with the caption ``/minutes``. The arming is dropped silently when it expires or when the manager
sends an ordinary text message instead.

The event is then dropped from normal dispatch and processed in the background: the audio moves to
``<home>/recordings/<id>/``, is split and transcribed (``audio.py``), the transcript is saved and
the audio deleted, one structured LLM call writes the brief and extracts the tasks (``brief.py``),
and the tasks go back to the manager for confirmation (``confirm.py``, shown by ``present.py``):
on Telegram one message per task with its own buttons (``rec:`` callbacks, scoped so the core
button flows keep working), edited in place as tasks are handled; otherwise one list and text replies.
Nothing goes to anyone but the manager until the manager, after confirming, taps [📤 Send tasks]
and then ✅ under the preview (``sending.py``): each person then gets only their own confirmed
tasks on Telegram. Only members who joined through a manager-approved ``/invite`` link
(``invite.py``) can be reached; a task reaches its owner's member (``contacts.py``, ``/team``).

The brief is retried once inside ``brief.analyze`` when the model returns no usable JSON; if it
still fails the transcript stays saved and ``/brief <id> retry`` writes the brief again from it
(no audio needed). ``tact_recorder.brief_model`` in config.yaml optionally names another model on
the main provider for the brief (it needs ``plugins.entries.tact-recorder.llm.allow_model_override``).

Telegram bots cannot download files over 20 MB: the adapter passes such a message on without its
media, and the manager is asked for a compressed audio-only file instead.

Transcripts and API keys are never logged; log lines carry meeting ids and counts only.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import audio
from . import brief as fmt
from . import confirm
from . import contacts
from . import invite
from . import present
from . import sending
from . import store

logger = logging.getLogger(__name__)

VOICE_MIN_SECONDS = 180
TELEGRAM_MAX_BYTES = 20 * 1024 * 1024
EDIT_TTL_SECONDS = 15 * 60
MINUTES_TTL_SECONDS = 10 * 60

TOO_LARGE = ("The file is too large for Telegram. Please send an audio-only compressed recording "
             "(e.g. m4a), or split it.")
STARTED = "🎙️ Transcribing your meeting, this may take a few minutes…"
MINUTES_READY = "🎙️ Send the meeting recording now."
BRIEF_FAILED = "⚠️ Meeting #{id}: the brief could not be written. Send /brief {id} retry to try again."

_LLM: Any = None  # ctx.llm, bound in register()
_RUNNING: set = set()  # background jobs (kept referenced until done)
# (platform, chat_id) -> (meeting_id, task number, field, expiry) after ✏️ and a field button
_awaiting_edit: Dict[Tuple[str, str], Tuple[int, int, str, float]] = {}
# (platform, chat_id) -> expiry after a bare /minutes: the next audio/voice is a recording
_minutes_armed: Dict[Tuple[str, str], float] = {}
_BUTTON_RE = re.compile(r"^rec:([acerfbu]):(\d+):(\d+)(?::([ntdk]))?$")
# rec:x:<p|s|c>:<meeting> sending, rec:s:<y|n>:<meeting>:<task> save contact, rec:j:<a|o|r>:<request>[:<i>] join
# rec:d:<y|n>:<meeting, 0 = all> delete recordings; rec:w:a:<meeting>:<task>:<contact> which person a
# shared first name means; rec:p:<e|f|r|k|d|n>:<contact> edit / delete a contact
# rec:m:<a|b>:<meeting>:<task>[:<member>] link a task to a registered member; rec:s:<y|n>:<meeting>:<task>
# keep the task's owner name as that member's alias
_EXTRA_RE = re.compile(r"^rec:([xsjdwpm]):([a-z]):(\d+)(?::(\d+))?(?::(\d+))?$")
_CONTACTS_RE = re.compile(r"^(edit|تعديل|delete|remove|حذف|احذف)\s+(.+?)$", re.IGNORECASE)
PERSON_ACTION_TTL = 15 * 60
# (chat_id, contact id) -> (manager user id, expiry): contact buttons shown to that manager, so a
# crafted callback from anyone else (or later) does nothing
_person_actions: Dict[Tuple[str, str], Tuple[str, float]] = {}
# (platform, chat_id) -> (contact id, field, expiry) after a contact field button
_awaiting_person: Dict[Tuple[str, str], Tuple[str, str, float]] = {}
_PERSON_FIELDS = {"f": ("full_name", "الاسم الكامل"), "r": ("role", "الوظيفة / القسم")}
_DELETE_RE = re.compile(r"^(?:delete|remove|حذف)\s+#?(\d+)(\s+confirm)?$|^(clear)(\s+confirm)?$", re.IGNORECASE)
_BUTTON_ACTIONS = {"a": "all", "c": "confirm", "e": "edit", "r": "remove", "f": "field", "b": "back", "u": "undo"}
_BUTTON_FIELDS = {"n": "person", "t": "task", "d": "deadline", "k": "contact"}
_INVITE_RE = re.compile(r"^(?:revoke|cancel|إلغاء|الغاء)?$", re.IGNORECASE)
_RETRY_RE = re.compile(r"^#?(\d+)\s+retry$", re.IGNORECASE)
_SHOW_RE = re.compile(r"^#?(\d+)$")


# -- recognising a recording ---------------------------------------------------------------------

def _attachment(raw: Any) -> Tuple[str, Any]:
    for kind in ("voice", "audio", "document", "video"):
        att = getattr(raw, kind, None)
        if att:
            return kind, att
    return "", None


def _is_recording_document(doc: Any) -> bool:
    from tools.transcription_common import SUPPORTED_FORMATS
    mime = str(getattr(doc, "mime_type", "") or "").lower()
    name = str(getattr(doc, "file_name", "") or "").lower()
    return mime.startswith(("audio/", "video/")) or Path(name).suffix in SUPPORTED_FORMATS


def classify(event: Any, max_bytes: int = TELEGRAM_MAX_BYTES, force: bool = False) -> Tuple[str, Optional[str]]:
    """``("recording", path)``, ``("too_large", None)`` or ``("", None)`` for anything else.
    ``force`` (the manager asked with /minutes) makes a voice note a recording whatever its length."""
    from gateway.platforms.event import MessageType

    kind, att = _attachment(getattr(event, "raw_message", None))
    media = list(getattr(event, "media_urls", None) or [])
    if not media:
        # The adapter drops media it refused to download (over the bot download limit).
        if kind in ("voice", "audio") or (kind == "document" and _is_recording_document(att)):
            size = int(getattr(att, "file_size", 0) or 0)
            if size > max_bytes or (kind == "document" and size <= 0):
                return "too_large", None
        return "", None
    types = list(getattr(event, "media_types", None) or [])
    for i, path in enumerate(media):
        mtype = (types[i] if i < len(types) else "").lower()
        is_audio = mtype.startswith("audio/") or (
            not mtype and event.message_type in (MessageType.VOICE, MessageType.AUDIO))
        if is_audio:
            if event.message_type == MessageType.VOICE:
                duration = int(getattr(att, "duration", 0) or 0) if kind == "voice" else 0
                return ("recording", path) if force or duration >= VOICE_MIN_SECONDS else ("", None)
            return "recording", path
        if mtype.startswith("video/") and kind == "document":
            return "recording", path
    return "", None


# -- gateway plumbing ----------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def _profile_scope(gateway: Any, source: Any):
    """The hook runs before the gateway binds the routed profile; bind it like run_inbound does."""
    if getattr(getattr(gateway, "config", None), "multiplex_profiles", False):
        from gateway.run import _async_profile_runtime_scope
        async with _async_profile_runtime_scope(gateway._resolve_profile_home_for_source(source)):
            yield
    else:
        yield


def _adapter(gateway: Any, source: Any) -> Any:
    finder = getattr(gateway, "_delivery_adapter_for", None)
    adapter = finder(source) if callable(finder) else None
    return adapter or (getattr(gateway, "adapters", None) or {}).get(source.platform)


_send = present.send


def _platform(source: Any) -> str:
    return getattr(source.platform, "value", str(source.platform))


# -- the pipeline --------------------------------------------------------------------------------

async def process_recording(gateway: Any, source: Any, meeting_id: int, audio_path: Path, note: str) -> None:
    adapter = _adapter(gateway, source)
    async with _profile_scope(gateway, source):
        try:
            transcript = await asyncio.to_thread(audio.transcribe_recording, audio_path)
        except audio.TranscriptionError as exc:
            store.update_meeting(meeting_id, status="failed", error=str(exc))
            await _send(adapter, source.chat_id, f"⚠️ Meeting #{meeting_id}: {exc} Please send the recording again.")
            return
        except Exception as exc:
            logger.exception("tact-recorder: meeting #%s transcription crashed", meeting_id)
            store.update_meeting(meeting_id, status="failed", error=type(exc).__name__)
            await _send(adapter, source.chat_id,
                        f"⚠️ Meeting #{meeting_id}: the transcription failed. Please send the recording again.")
            return
        transcript_path = store.meeting_dir(meeting_id) / "transcript.txt"
        transcript_path.write_text(transcript + "\n", encoding="utf-8")
        store.update_meeting(meeting_id, status="analyzing", transcript_path=str(transcript_path))
        audio_path.unlink(missing_ok=True)
        logger.info("tact-recorder: meeting #%s transcribed (%d chars)", meeting_id, len(transcript))
        await write_brief(adapter, source, meeting_id, transcript, note)


def _brief_model() -> Optional[str]:
    """``tact_recorder.brief_model``: a model on the main provider for the brief; unset = main model."""
    from hermes_cli.config import load_config_readonly
    section = (load_config_readonly() or {}).get("tact_recorder")
    model = section.get("brief_model") if isinstance(section, dict) else None
    return str(model or "").strip() or None


async def write_brief(adapter: Any, source: Any, meeting_id: int, transcript: str, note: str) -> None:
    """Brief + tasks from a saved transcript, sent to the manager; on failure the meeting stays
    retryable with ``/brief <id> retry``."""
    try:
        lang, brief, tasks = await fmt.analyze(_LLM, transcript, note, model=_brief_model())
    except Exception as exc:
        from agent.plugin_llm import PluginLlmTrustError
        # A trust error names only config keys (brief_model set without allow_model_override);
        # any other error text could echo model output, so only its type is logged.
        reason = str(exc) if isinstance(exc, PluginLlmTrustError) else type(exc).__name__
        logger.warning("tact-recorder: meeting #%s brief failed: %s", meeting_id, reason)
        store.update_meeting(meeting_id, status="failed", error=f"brief: {type(exc).__name__}")
        await _send(adapter, source.chat_id, BRIEF_FAILED.format(id=meeting_id))
        return
    store.save_analysis(meeting_id, lang, brief, tasks)
    logger.info("tact-recorder: meeting #%s briefed, %d task(s)", meeting_id, len(tasks))
    await present.show_meeting(adapter, _platform(source), str(source.chat_id), meeting_id)


def _start_job(coro: Any) -> None:
    job = asyncio.create_task(coro)
    _RUNNING.add(job)
    job.add_done_callback(_RUNNING.discard)


def _start_recording(gateway: Any, source: Any, path: str, note: str) -> int:
    meeting_id = store.create_meeting(_platform(source), str(source.chat_id), str(source.user_id), note,
                                      fmt.clean(getattr(source, "user_name", "") or ""))
    src = Path(path)
    dest = store.meeting_dir(meeting_id) / f"audio{src.suffix.lower() or '.bin'}"
    shutil.move(str(src), dest)
    _start_job(process_recording(gateway, source, meeting_id, dest, note))
    return meeting_id


async def _retry_brief(gateway: Any, source: Any, meeting_id: int) -> str:
    """Reply to ``/brief <id> retry``; starts the brief again from the saved transcript."""
    meeting = store.get_meeting(meeting_id, _platform(source), str(source.chat_id))
    if meeting is None:
        return f"No recorded meeting #{meeting_id}. See /recordings."
    if meeting["brief"]:
        return f"Meeting #{meeting_id} already has a brief. Show it with /brief {meeting_id}."
    if meeting["status"] == "analyzing":
        return f"📝 The brief for meeting #{meeting_id} is already being written."
    path = Path(meeting["transcript_path"] or "")
    if meeting["status"] != "failed" or not meeting["transcript_path"] or not path.is_file():
        return f"Meeting #{meeting_id} has no saved transcript. Please send the recording again."
    store.update_meeting(meeting_id, status="analyzing", error=None)

    async def run() -> None:
        async with _profile_scope(gateway, source):
            await write_brief(_adapter(gateway, source), source, meeting_id,
                              path.read_text(encoding="utf-8").strip(), meeting["note"] or "")
    _start_job(run())
    logger.info("tact-recorder: meeting #%s brief retry started", meeting_id)
    return f"📝 Writing the brief for meeting #{meeting_id} again…"


def _text_replies(platform: str, chat_id: str, text: str) -> Optional[confirm.Outcome]:
    """The outcome when *text* answers a pending confirmation in this chat, else None (not ours)."""
    command = confirm.parse_command(text)
    awaiting = _awaiting_edit.pop((platform, chat_id), None)
    if command is None and awaiting and awaiting[3] > time.monotonic():
        meeting = store.get_meeting(awaiting[0], platform, chat_id)
        if meeting is not None:
            # The whole message is the new value of the field the manager chose; never parsed.
            return confirm.set_field(meeting, awaiting[1], awaiting[2], text)
    if command is None:
        return None
    meeting = store.latest_pending(platform, chat_id)
    if meeting is None and command[0] in ("edit", "undo"):
        meeting = store.latest_briefed(platform, chat_id)  # a confirmed meeting can still be corrected
    if meeting is None:
        return None
    return confirm.apply(meeting, *command)


async def _on_pre_gateway_dispatch(event: Any = None, gateway: Any = None, **_: Any) -> Optional[Dict[str, str]]:
    source = getattr(event, "source", None)
    if (source is None or getattr(event, "internal", False) or source.chat_type != "dm"
            or not callable(getattr(gateway, "_is_user_authorized_for_source", None))):
        return None
    adapter = _adapter(gateway, source)
    key = (_platform(source), str(source.chat_id))
    text, command = event.text or "", event.get_command()
    has_attachment = bool(event.media_urls) or _attachment(getattr(event, "raw_message", None))[1] is not None
    force = has_attachment and (command == "minutes" or _minutes_armed.get(key, 0.0) > time.monotonic())
    kind, path = classify(event, int(getattr(adapter, "_max_doc_bytes", 0) or TELEGRAM_MAX_BYTES), force)
    brief_args = event.get_command_args().strip() if command == "brief" and not has_attachment else ""
    retry, show = _RETRY_RE.match(brief_args), _SHOW_RE.match(brief_args)
    invite_cmd = (command == "invite" and not has_attachment and key[0] == "telegram"
                  and _INVITE_RE.match(event.get_command_args().strip()))
    # "/recordings delete <id>" / "/recordings clear" ask with buttons (the "... confirm" form, for
    # chats without buttons, goes to the command itself).
    delete_cmd = (command == "recordings" and not has_attachment and getattr(adapter, "_bot", None) is not None
                  and key[0] == "telegram" and _DELETE_RE.match(event.get_command_args().strip()))
    if delete_cmd and (delete_cmd.group(2) or delete_cmd.group(4)):
        delete_cmd = None
    contacts_cmd = (command == "contacts" and not has_attachment and getattr(adapter, "_bot", None) is not None
                    and key[0] == "telegram" and _CONTACTS_RE.match(event.get_command_args().strip()))
    if contacts_cmd and contacts_cmd.group(2).lower().endswith(" confirm"):
        contacts_cmd = None
    # Other commands (a bare /minutes included) go on to the gateway's command dispatch.
    if (not kind and not retry and not show and not invite_cmd and not delete_cmd and not contacts_cmd
            and (has_attachment or not text.strip() or command)):
        return None
    # Only the manager: anyone else continues to the gateway's normal refusal / pairing path.
    if not gateway._is_user_authorized_for_source(source):
        return None
    # A recording uses up a /minutes arming; an ordinary text message cancels it silently.
    _minutes_armed.pop(key, None)
    if command == "minutes":
        text = event.get_command_args()
    async with _profile_scope(gateway, source):
        if delete_cmd:
            await _ask_delete(adapter, source, int(delete_cmd.group(1) or 0))
            return {"action": "skip", "reason": "tact-recorder: delete recordings"}
        if contacts_cmd:
            await _contacts_action(adapter, source, contacts_cmd.group(1).lower(), fmt.clean(contacts_cmd.group(2)))
            return {"action": "skip", "reason": "tact-recorder: contacts"}
        if invite_cmd:
            await _send(adapter, source.chat_id, _invite_reply(adapter, source, invite_cmd.group(0)))
            return {"action": "skip", "reason": "tact-recorder: invite"}
        if not command and not has_attachment:
            person_reply = await _take_person_value(adapter, key, str(source.user_id), text)
            if person_reply is not None:  # a contact's full name / role / contact after its field button
                await _send(adapter, source.chat_id, person_reply)
                return {"action": "skip", "reason": "tact-recorder: contact edit"}
            typed = await invite.take_typed_name(getattr(adapter, "_bot", None), *key, str(source.user_id), text)
            if typed is not None:  # the name for a join request after "➕ اسم آخر"
                for reply in typed:
                    await _send(adapter, source.chat_id, reply)
                return {"action": "skip", "reason": "tact-recorder: join name"}
        if retry:
            # Handled here, not in the /brief command, because the retry replies later from the
            # background through the adapter, which only this hook can reach.
            await _send(adapter, source.chat_id, await _retry_brief(gateway, source, int(retry.group(1))))
            return {"action": "skip", "reason": "tact-recorder: brief retry"}
        if show:
            # Here too: the brief and its task cards are several messages with buttons.
            meeting_id = int(show.group(1))
            reply = _brief_status(meeting_id, *key)
            if reply is None:
                await present.show_meeting(adapter, key[0], key[1], meeting_id)
            else:
                await _send(adapter, source.chat_id, reply)
            return {"action": "skip", "reason": "tact-recorder: show brief"}
        if kind == "too_large":
            await _send(adapter, source.chat_id, TOO_LARGE)
            return {"action": "skip", "reason": "tact-recorder: recording too large"}
        if kind == "recording":
            meeting_id = _start_recording(gateway, source, path, fmt.clean(text))
            logger.info("tact-recorder: meeting #%s received", meeting_id)
            await _send(adapter, source.chat_id, STARTED)
            return {"action": "skip", "reason": "tact-recorder: meeting recording"}
        outcome = _text_replies(*key, text)
        if outcome is None:
            return None
        await present.show_changes(adapter, key[0], key[1], outcome)
    return {"action": "skip", "reason": "tact-recorder: task confirmation"}


# -- Telegram buttons ----------------------------------------------------------------------------

def handle_button(data: str, platform: str, chat_id: str, user_id: str) -> Tuple[Optional[confirm.Outcome], str]:
    """``(outcome, toast)`` for a ``rec:`` button press. Only the manager who sent the recording, in
    the chat it was sent from, can act on its tasks."""
    m = _BUTTON_RE.match(data or "")
    if not m:
        return None, ""
    action, meeting_id, position = _BUTTON_ACTIONS[m.group(1)], int(m.group(2)), int(m.group(3))
    meeting = store.get_meeting(meeting_id, platform, chat_id)
    if meeting is None or meeting["user_id"] != user_id:
        return None, "Not allowed."
    if action in ("edit", "field", "back"):  # at any time: a confirmed task can still be corrected
        if action == "edit":  # the card offers Name / Task / Deadline / Back
            lang = fmt.lang_of(meeting["language"])
            warning = [confirm.SENT_WARNING[lang]] if confirm.was_sent(meeting_id, position) else []
            return confirm.Outcome(meeting_id, messages=warning, changed=[position], picker=position), ""
        if action == "back":  # normal buttons again
            _awaiting_edit.pop((platform, chat_id), None)
            return confirm.Outcome(meeting_id, changed=[position]), ""
        field_name = _BUTTON_FIELDS[m.group(4) or "n"]
        if field_name == "contact":  # no typed contacts: the registered members are offered instead
            return confirm.Outcome(meeting_id, changed=[position], member_picker=position), ""
        _awaiting_edit[(platform, chat_id)] = (meeting_id, position, field_name,
                                               time.monotonic() + EDIT_TTL_SECONDS)
        lang = fmt.lang_of(meeting["language"])
        return confirm.Outcome(meeting_id, messages=[confirm.FIELD_PROMPT[lang][field_name].format(n=position)]), ""
    return confirm.apply(meeting, action, position), ""


def _delete_question(meeting_id: int, platform: str, chat_id: str, user_id: str) -> Optional[str]:
    """What deleting would remove, or None when there is nothing of this manager's to delete."""
    if meeting_id:
        meeting = sending.resolve_meeting(meeting_id, platform, chat_id, user_id)
        return None if meeting is None else f"🗑️ حذف الاجتماع #{meeting_id} ومهامه ونصّه نهائياً؟"
    count = len(store.meeting_ids(platform, chat_id, user_id))
    return (f"🗑️ حذف كل الاجتماعات المسجّلة ({count}) ومهامها ونصوصها نهائياً؟ دفتر جهات الاتصال يبقى كما هو."
            if count else None)


def _delete(meeting_id: int, platform: str, chat_id: str, user_id: str) -> str:
    ids = [meeting_id] if meeting_id else store.meeting_ids(platform, chat_id, user_id)
    for number in ids:
        store.delete_meeting(number)
    logger.info("tact-recorder: %d meeting(s) deleted by the manager", len(ids))
    return f"🗑️ حُذف الاجتماع #{meeting_id}." if meeting_id else f"🗑️ حُذفت كل الاجتماعات ({len(ids)})."


async def _ask_delete(adapter: Any, source: Any, meeting_id: int) -> None:
    from telegram import InlineKeyboardButton as Button, InlineKeyboardMarkup
    question = _delete_question(meeting_id, _platform(source), str(source.chat_id), str(source.user_id))
    if question is None:
        await _send(adapter, source.chat_id, f"لا يوجد اجتماع #{meeting_id}." if meeting_id else "لا توجد اجتماعات.")
        return
    await adapter._bot.send_message(chat_id=present._chat_arg(source.chat_id), text=question, reply_markup=(
        InlineKeyboardMarkup([[Button("✅ حذف", callback_data=f"rec:d:y:{meeting_id}"),
                               Button("❌ إلغاء", callback_data=f"rec:d:n:{meeting_id}")]])))


def _allow_person_action(chat_id: str, contact_id: Any, user_id: str) -> None:
    _person_actions[(str(chat_id), str(contact_id))] = (str(user_id), time.monotonic() + PERSON_ACTION_TTL)


def _may_act_on_person(chat_id: str, contact_id: Any, user_id: str) -> bool:
    allowed = _person_actions.get((str(chat_id), str(contact_id)))
    return bool(allowed and allowed[0] == str(user_id) and allowed[1] > time.monotonic())


def _person_markup(row: Dict[str, Any]) -> Any:
    from telegram import InlineKeyboardButton as Button, InlineKeyboardMarkup
    return InlineKeyboardMarkup([
        [Button("👤 الاسم الكامل", callback_data=f"rec:p:f:{row['id']}"),
         Button("💼 الوظيفة", callback_data=f"rec:p:r:{row['id']}")]])


async def _contacts_action(adapter: Any, source: Any, verb: str, name: str) -> None:
    """``/contacts edit|delete <name>`` with buttons; a name fitting several people asks which."""
    from telegram import InlineKeyboardButton as Button, InlineKeyboardMarkup
    chat, user = str(source.chat_id), str(source.user_id)
    deleting = verb not in ("edit", "تعديل")
    exact = contacts.exact(name)
    rows = [exact] if exact else contacts.matches(name)
    if not rows:
        reply = _cmd_contacts(f"delete {name}") if deleting else f"لا توجد جهة اتصال باسم {name}."
        await _send(adapter, chat, reply)  # a name only in past tasks is hidden right away
        return
    for row in rows:
        _allow_person_action(chat, row["id"], user)
    if deleting:
        text = (f"🗑️ حذف {contacts.label(rows[0])} من جهات الاتصال؟"
                + (" هو مسجّل في البوت وسيتوقف عن استقبال المهام." if rows[0]["telegram_chat_id"] else "")
                if len(rows) == 1 else f"أكثر من شخص باسم {name}. من تريد حذفه؟")
        buttons = [[Button("✅ حذف" if len(rows) == 1 else f"🗑️ {contacts.label(r)}",
                           callback_data=f"rec:p:d:{r['id']}")] for r in rows]
        buttons.append([Button("❌ إلغاء", callback_data=f"rec:p:n:{rows[0]['id']}")])
        markup = InlineKeyboardMarkup(buttons)
    elif len(rows) == 1:
        text, markup = contacts.edit_text(rows[0]), _person_markup(rows[0])
    else:
        text = f"أكثر من شخص باسم {name}. من تريد تعديله؟"
        markup = InlineKeyboardMarkup([[Button(contacts.label(r), callback_data=f"rec:p:e:{r['id']}")] for r in rows])
    await adapter._bot.send_message(chat_id=present._chat_arg(chat), text=text, reply_markup=markup)


async def _person_button(adapter: Any, action: str, contact_id: int, chat_id: str, user_id: str,
                         message_id: Any) -> str:
    if not _may_act_on_person(chat_id, contact_id, user_id):
        return "Not allowed."
    bot = adapter._bot
    row = contacts.by_id(contact_id)
    if row is None:
        await present._edit(bot, chat_id, message_id, "لم تعد جهة الاتصال موجودة.", None)
        return ""
    if action == "e":
        await present._edit(bot, chat_id, message_id, contacts.edit_text(row), _person_markup(row))
    elif action in _PERSON_FIELDS:
        field, title = _PERSON_FIELDS[action]
        _awaiting_person[("telegram", str(chat_id))] = (str(contact_id), field, time.monotonic() + EDIT_TTL_SECONDS)
        await _send(adapter, chat_id, f"أرسل {title} لـ {row['name']}")
    elif action == "d":
        contacts.delete(row["name"])
        for key in [k for k in _person_actions if k[1] == str(contact_id)]:
            _person_actions.pop(key, None)
        await present._edit(bot, chat_id, message_id,
                            f"🗑️ حُذف {contacts.label(row)}. لن يستقبل مهاماً بعد الآن.", None)
    else:  # "n": keep
        await present._edit(bot, chat_id, message_id, "لم يُحذف شيء.", None)
    return ""


async def _take_person_value(adapter: Any, key: Tuple[str, str], user_id: str, text: str) -> Optional[str]:
    waiting = _awaiting_person.pop(key, None)
    if waiting is None or waiting[2] <= time.monotonic() or not _may_act_on_person(key[1], waiting[0], user_id):
        return None
    row = contacts.by_id(waiting[0])
    if row is None:
        return "لم تعد جهة الاتصال موجودة."
    contacts.set_details(row["name"], **{waiting[1]: text})
    row = contacts.by_id(waiting[0])
    await adapter._bot.send_message(chat_id=present._chat_arg(key[1]), text="✅ حُفظ.\n" + contacts.edit_text(row),
                                    reply_markup=_person_markup(row))
    return ""


def _invite_reply(adapter: Any, source: Any, args: str) -> str:
    if args:
        return f"🚫 أُلغيت روابط الدعوة النشطة ({invite.revoke_invites(str(source.user_id))})."
    finder = getattr(adapter, "_current_bot_username", None)
    username = finder() if callable(finder) else (getattr(getattr(adapter, "_bot", None), "username", "") or "")
    if not username:
        return "تعذّر معرفة اسم البوت؛ حاول مرة أخرى بعد قليل."
    token, expires_at = invite.create_invite(_platform(source), str(source.chat_id), str(source.user_id))
    return invite.invite_text(username.lstrip("@"), token, expires_at)


async def handle_extra_button(adapter: Any, data: str, chat_id: str, user_id: str, message_id: Any) -> str:
    """Sending, save-contact and join buttons; returns the toast. Each checks the presser is the
    manager the meeting (or the invite) belongs to."""
    m = _EXTRA_RE.match(data or "")
    if not m:
        return ""
    kind, action, number, extra = m.group(1), m.group(2), int(m.group(3)), int(m.group(4) or 0)
    bot = getattr(adapter, "_bot", None)
    if kind == "p":
        return await _person_button(adapter, action, number, chat_id, user_id, message_id)
    if kind == "d":  # only the manager whose meetings they are; nothing is deleted without ✅
        if _delete_question(number, "telegram", chat_id, user_id) is None:
            return "Not allowed."
        text = _delete(number, "telegram", chat_id, user_id) if action == "y" else "لم يُحذف شيء."
        await present._edit(bot, chat_id, message_id, text, None)
        return ""
    if kind == "j":
        messages, toast = await invite.handle_button(bot, action, number, extra, chat_id, user_id)
        for text in messages:
            await _send(adapter, chat_id, text)
        return toast
    meeting = sending.resolve_meeting(number, "telegram", chat_id, user_id)
    if meeting is None:
        return "Not allowed."
    lang = fmt.lang_of(meeting["language"])
    task = next((t for t in store.tasks_for(number) if t["position"] == extra), None) if extra else None
    if kind == "m":  # link the task to a registered member picked from /team
        if task is None:
            return "✔️"
        if action == "b":
            await present._edit(bot, chat_id, message_id, "لم يتغير شيء." if lang == "ar" else "Nothing changed.", None)
            return ""
        member = contacts.by_id(m.group(5) or 0)
        if not contacts.is_member(member):
            return "✔️"
        store.set_task_contact_id(number, extra, member["id"])
        await present._edit(bot, chat_id, message_id,
                            (f"✅ رُبطت المهمة {extra} بـ {contacts.label(member)}." if lang == "ar"
                             else f"✅ Task {extra} is linked to {contacts.label(member)}."), None)
        await present.refresh_cards(adapter, "telegram", chat_id, number, [extra])
        owner = task["person"]
        if len(fmt.split_owners(owner)) == 1 and not contacts.knows(member, owner):
            await present.ask_save_alias(bot, chat_id, number, extra, owner, member, lang)
        return ""
    if kind == "s":  # keep the owner's name as an alias of the member linked to the task
        member = contacts.by_id(task["contact_id"]) if task and task["contact_id"] else None
        if member is None:
            return "✔️"
        if action == "y":
            contacts.add_alias(member["id"], task["person"])
            text = (f"✅ «{task['person']}» محفوظ كاسم آخر لـ {contacts.label(member)}." if lang == "ar"
                    else f"✅ “{task['person']}” saved for {contacts.label(member)}.")
        else:
            text = "لم يُحفظ." if lang == "ar" else "Not saved."
        await present._edit(bot, chat_id, message_id, text, None)
        return ""
    if action == "p":  # preview only: nothing is sent from here
        plan = sending.plan(number)
        await bot.send_message(chat_id=present._chat_arg(chat_id), text=sending.preview_text(number, lang, plan),
                               reply_markup=sending.markup("confirm", number, lang, plan))
        return ""
    if kind == "w":  # the manager says which person a shared first name means for this task
        contact = contacts.by_id(m.group(5) or 0)
        if contact is None or not any(t["position"] == extra for t in store.tasks_for(number)):
            return "✔️"
        store.set_task_contact_id(number, extra, contact["id"])
        plan = sending.plan(number)
        await present._edit(bot, chat_id, message_id, sending.preview_text(number, lang, plan),
                            sending.markup("confirm", number, lang, plan))
        return ""
    if action == "c":
        await present._edit(bot, chat_id, message_id, sending.cancelled_text(lang), None)
        return ""
    if action == "r":  # [📤 re-send] after editing a sent task: this task only, to its own member(s)
        plan = sending.plan(number, [extra])
        await present._edit(bot, chat_id, message_id, "📤 …", None)
        report, sent = await sending.execute(bot, meeting, lang, plan)
        await present.refresh_cards(adapter, "telegram", chat_id, number, sent)
        await _send(adapter, chat_id, report)
        return ""
    # action == "s": the manager's explicit ✅ under the preview
    await present._edit(bot, chat_id, message_id, "📤 …", None)  # no second tap on the same preview
    report, sent = await sending.execute(bot, meeting, lang, sending.plan(number))
    await present.refresh_cards(adapter, "telegram", chat_id, number, sent)
    await _send(adapter, chat_id, report)
    return ""


async def on_join_message(adapter: Any, message: Any) -> None:
    """A private ``/start join_<token>``: handled here, before the allowlist prefilter would drop it."""
    user = message.from_user
    await invite.handle_join(adapter._bot, str(message.chat.id), str(user.id), user.username or "",
                             getattr(user, "full_name", "") or "", message.text or "")


def _telegram_handlers(app: Any, adapter: Any) -> None:
    from telegram.ext import CallbackQueryHandler, MessageHandler, filters

    async def on_button(update: Any, _context: Any) -> None:
        accept = getattr(adapter, "_accept_update", None)
        if callable(accept):
            accept()
        query = update.callback_query
        chat_id = str(query.message.chat.id)
        if _EXTRA_RE.match(query.data or ""):
            toast = await handle_extra_button(adapter, query.data, chat_id, str(query.from_user.id),
                                              query.message.message_id)
            await query.answer(toast or None)
            return
        outcome, toast = handle_button(query.data, "telegram", chat_id, str(query.from_user.id))
        await query.answer(toast or None)
        if outcome is not None:
            position = int(query.data.split(":")[3])
            await present.show_changes(adapter, "telegram", chat_id, outcome,
                                       pressed={position: query.message.message_id})

    async def on_join(update: Any, _context: Any) -> None:
        accept = getattr(adapter, "_accept_update", None)
        if callable(accept):
            accept()
        await on_join_message(adapter, update.effective_message)

    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^rec:"))
    # Exactly "/start join_<token>" in a private chat; every other message keeps the core path.
    app.add_handler(MessageHandler(filters.Regex(invite.JOIN_RE) & filters.ChatType.PRIVATE, on_join))


# -- slash commands ------------------------------------------------------------------------------

def _chat() -> Tuple[str, str]:
    from gateway.session_context import get_session_env
    return get_session_env("HERMES_SESSION_PLATFORM"), get_session_env("HERMES_SESSION_CHAT_ID")


_MEETING_STATUS = {"transcribing": "🎙️ transcribing", "analyzing": "📝 writing the brief",
                   "pending": "⏳ awaiting confirmation", "confirmed": "✅ confirmed", "failed": "⚠️ failed"}


def _cmd_invite(_raw_args: str) -> str:
    """Only reached where the hook did not answer (not a private Telegram chat)."""
    return "Use /invite in your private chat with the bot on Telegram."


def _cmd_team(_raw_args: str) -> str:
    return contacts.team_text()


def _cmd_contacts(raw_args: str) -> str:
    """Only reached where the hook did not answer with buttons (no private Telegram chat), and for
    names that exist only in past tasks."""
    args = str(raw_args or "").strip()
    if re.match(r"^(?:edit|تعديل)\s+", args, re.IGNORECASE):
        return "استخدم /contacts edit في محادثتك الخاصة مع البوت على تيليجرام."
    renamed = re.match(r"^(?:rename|تسمية|إعادة تسمية)\s+(.+?)\s*(?:->|→|=>|>)\s*(.+)$", args, re.IGNORECASE)
    if renamed:
        return contacts.rename(renamed.group(1), renamed.group(2))[1]
    m = re.match(r"^(?:delete|remove|حذف|احذف)\s+(.+?)(\s+confirm)?$", args, re.IGNORECASE)
    if m:
        name = fmt.clean(m.group(1))
        if contacts.exact(name) is not None and not m.group(2):  # a book entry: confirm first
            return f"للتأكيد أرسل: /contacts delete {name} confirm"
        return f"🗑️ حُذف {name}." if contacts.delete(name) else f"لا توجد جهة اتصال باسم {name}."
    return contacts.list_text(*_chat())


def _cmd_minutes(_raw_args: str) -> str:
    _minutes_armed[_chat()] = time.monotonic() + MINUTES_TTL_SECONDS
    return MINUTES_READY


def _cmd_recordings(raw_args: str) -> str:
    platform, chat_id = _chat()
    m = _DELETE_RE.match(str(raw_args or "").strip())
    if m:
        from gateway.session_context import get_session_env
        user_id = get_session_env("HERMES_SESSION_USER_ID")
        meeting_id = int(m.group(1) or 0)
        if _delete_question(meeting_id, platform, chat_id, user_id) is None:
            return f"لا يوجد اجتماع #{meeting_id}." if meeting_id else "لا توجد اجتماعات."
        if not (m.group(2) or m.group(4)):  # no buttons here: an explicit "confirm" is required
            return f"للتأكيد أرسل: /recordings {'delete ' + str(meeting_id) if meeting_id else 'clear'} confirm"
        return _delete(meeting_id, platform, chat_id, user_id)
    rows = store.recent_meetings(platform, chat_id)
    if not rows:
        return "🎙️ No recorded meetings yet. Send a meeting recording to start."
    lines = ["🎙️ Recent meetings:"]
    for r in rows:
        tasks = f"{r['n_tasks']} task(s), {r['n_confirmed']} confirmed" if r["n_tasks"] else "no tasks"
        lines.append(f"#{r['id']} — {r['created_at'][:16].replace('T', ' ')} — {tasks} — "
                     f"{_MEETING_STATUS.get(r['status'], r['status'])}")
    lines.append("Show one with /brief <id>.")
    return "\n".join(lines)


def _brief_status(meeting_id: int, platform: str, chat_id: str) -> Optional[str]:
    """The reply for a meeting that has no brief to show (unknown, in progress, failed), else None."""
    meeting = store.get_meeting(meeting_id, platform, chat_id)
    if meeting is None:
        return f"No recorded meeting #{meeting_id}. See /recordings."
    if meeting["brief"]:
        return None
    if meeting["status"] == "failed" and meeting["transcript_path"]:
        return BRIEF_FAILED.format(id=meeting_id)
    return f"Meeting #{meeting_id}: {_MEETING_STATUS.get(meeting['status'], meeting['status'])}."


def _cmd_brief(raw_args: str) -> str:
    """Only reached where the hook did not show the meeting itself (no private chat): one text."""
    try:
        meeting_id = int(str(raw_args).strip().lstrip("#"))
    except ValueError:
        return "Usage: /brief <id> (see /recordings for ids), or /brief <id> retry in a private chat."
    platform, chat_id = _chat()
    status = _brief_status(meeting_id, platform, chat_id)
    if status is not None:
        return status
    meeting = store.get_meeting(meeting_id, platform, chat_id)
    lang = fmt.lang_of(meeting["language"])
    tasks = contacts.annotate(store.tasks_for(meeting_id), lang)
    pending = [t for t in tasks if t["status"] == "pending"]
    text = fmt.brief_text(meeting_id, meeting["created_at"], lang, json.loads(meeting["brief"]), len(tasks), False)
    if len(pending) < len(tasks):
        text += "\n\n" + fmt.final_text(meeting_id, lang, tasks)
    if pending:
        text += "\n\n" + fmt.combined_text(pending, len(tasks), lang)
    return text


def register(ctx) -> None:
    global _LLM
    _LLM = ctx.llm
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
    ctx.register_telegram_handler(_telegram_handlers)
    ctx.register_command("minutes", _cmd_minutes,
                         description="Take meeting minutes from next audio")
    ctx.register_command("invite", _cmd_invite,
                         description="Invite link so a team member can receive tasks")
    ctx.register_command("contacts", _cmd_contacts, description="Who can receive tasks, and how")
    ctx.register_command("team", _cmd_team, description="Team members registered to receive tasks")
    ctx.register_command("recordings", _cmd_recordings,
                         description="List recorded meetings (delete <id> / clear)")
    ctx.register_command("brief", _cmd_brief, description="Show a recorded meeting's brief and tasks",
                         args_hint="<id>")
