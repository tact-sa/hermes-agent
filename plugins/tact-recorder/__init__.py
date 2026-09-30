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
Nothing is ever sent to anyone but the manager who sent the recording.

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
from . import present
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
# (platform, chat_id) -> (meeting_id, task position, expiry) after the manager taps ✏️
_awaiting_edit: Dict[Tuple[str, str], Tuple[int, int, float]] = {}
# (platform, chat_id) -> expiry after a bare /minutes: the next audio/voice is a recording
_minutes_armed: Dict[Tuple[str, str], float] = {}
_BUTTON_RE = re.compile(r"^rec:([acer]):(\d+):(\d+)$")
_BUTTON_ACTIONS = {"a": "all", "c": "confirm", "e": "edit", "r": "remove"}
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
    meeting_id = store.create_meeting(_platform(source), str(source.chat_id), str(source.user_id), note)
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
    if command is None and awaiting and awaiting[2] > time.monotonic():
        meeting = store.get_meeting(awaiting[0], platform, chat_id)
        if meeting is not None:
            return confirm.apply(meeting, "edit", awaiting[1], text)
    if command is None:
        return None
    meeting = store.latest_pending(platform, chat_id)
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
    # Other commands (a bare /minutes included) go on to the gateway's command dispatch.
    if not kind and not retry and not show and (has_attachment or not text.strip() or command):
        return None
    # Only the manager: anyone else continues to the gateway's normal refusal / pairing path.
    if not gateway._is_user_authorized_for_source(source):
        return None
    # A recording uses up a /minutes arming; an ordinary text message cancels it silently.
    _minutes_armed.pop(key, None)
    if command == "minutes":
        text = event.get_command_args()
    async with _profile_scope(gateway, source):
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
    if action == "edit" and meeting["status"] == "pending":
        _awaiting_edit[(platform, chat_id)] = (meeting_id, position, time.monotonic() + EDIT_TTL_SECONDS)
        lang = fmt.lang_of(meeting["language"])
        return confirm.Outcome(meeting_id, messages=[confirm.EDIT_HELP[lang].format(n=position)]), ""
    return confirm.apply(meeting, action, position), ""


def _telegram_handlers(app: Any, adapter: Any) -> None:
    from telegram.ext import CallbackQueryHandler

    async def on_button(update: Any, _context: Any) -> None:
        accept = getattr(adapter, "_accept_update", None)
        if callable(accept):
            accept()
        query = update.callback_query
        chat_id = str(query.message.chat.id)
        outcome, toast = handle_button(query.data, "telegram", chat_id, str(query.from_user.id))
        await query.answer(toast or None)
        if outcome is not None:
            position = int(query.data.rsplit(":", 1)[1])
            await present.show_changes(adapter, "telegram", chat_id, outcome,
                                       pressed={position: query.message.message_id})

    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^rec:"))


# -- slash commands ------------------------------------------------------------------------------

def _chat() -> Tuple[str, str]:
    from gateway.session_context import get_session_env
    return get_session_env("HERMES_SESSION_PLATFORM"), get_session_env("HERMES_SESSION_CHAT_ID")


_MEETING_STATUS = {"transcribing": "🎙️ transcribing", "analyzing": "📝 writing the brief",
                   "pending": "⏳ awaiting confirmation", "confirmed": "✅ confirmed", "failed": "⚠️ failed"}


def _cmd_minutes(_raw_args: str) -> str:
    _minutes_armed[_chat()] = time.monotonic() + MINUTES_TTL_SECONDS
    return MINUTES_READY


def _cmd_recordings(_raw_args: str) -> str:
    platform, chat_id = _chat()
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
    lang, tasks = fmt.lang_of(meeting["language"]), store.tasks_for(meeting_id)
    text = fmt.brief_text(meeting_id, meeting["created_at"], lang, json.loads(meeting["brief"]), len(tasks), False)
    if tasks:
        text += "\n\n" + fmt.combined_text(tasks, lang, with_help=meeting["status"] == "pending")
    return text


def register(ctx) -> None:
    global _LLM
    _LLM = ctx.llm
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
    ctx.register_telegram_handler(_telegram_handlers)
    ctx.register_command("minutes", _cmd_minutes,
                         description="Take meeting minutes from next audio")
    ctx.register_command("recordings", _cmd_recordings, description="List recent recorded meetings")
    ctx.register_command("brief", _cmd_brief, description="Show a recorded meeting's brief and tasks",
                         args_hint="<id>")
