"""tact-recorder: a meeting recording becomes a transcript, a brief and tasks the manager confirms.

Loads the bundled plugin through the real ``PluginManager`` against a temp HERMES_HOME and drives
the gateway's own ``pre_gateway_dispatch`` step with a stand-in gateway/adapter. STT
(``tools.transcription_tools.transcribe_audio``) and the LLM (``ctx.llm``) are mocked; the SQLite
store, the file handling and (when installed) ffmpeg's chunking are the real ones.
"""

import asyncio
import json
import logging
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAT, MANAGER = "4242", "7"
SECRET_LINE = "Ahmad will send the Q3 budget by Sunday"


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    import hermes_yaml as yaml
    from hermes_cli import plugins as pmod

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", CHAT)
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["tact-recorder"]}}))
    mgr = pmod.PluginManager()
    mgr.discover_and_load()
    loaded = mgr._plugins["tact-recorder"]
    assert loaded.enabled, loaded.error
    monkeypatch.setattr(pmod, "_plugin_manager", mgr)
    return loaded.module


class FakeAdapter:
    _max_doc_bytes = 20 * 1024 * 1024

    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content, **_kw):
        self.sent.append(content)
        return SimpleNamespace(success=True)


class FakeGateway:
    def __init__(self, authorized=True):
        self.adapter = FakeAdapter()
        self.authorized = authorized

    def _is_user_authorized_for_source(self, _source):
        return self.authorized

    def _delivery_adapter_for(self, _source):
        return self.adapter


class FakeLlm:
    def __init__(self, parsed):
        self.parsed, self.calls = parsed, []

    async def acomplete_structured(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(parsed=self.parsed, text=json.dumps(self.parsed))


def _wav(path, seconds):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * 16000 * seconds)
    return path


def _event(text="", message_type="AUDIO", media=(), mime="audio/wav", raw=None, user=MANAGER):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session import SessionSource
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=CHAT, user_id=user, chat_type="dm")
    return MessageEvent(text=text, message_type=MessageType[message_type], source=source,
                        media_urls=[str(m) for m in media], media_types=[mime] * len(media), raw_message=raw)


def _dispatch(mod, gateway, event):
    """The gateway's pre_gateway_dispatch step plus any background job it started; None = taken."""
    from gateway.run_inbound import GatewayInboundMixin

    async def run():
        out = await GatewayInboundMixin._hm_pre_gateway_dispatch_hook(gateway, event, event.source)
        while mod._RUNNING:
            await asyncio.gather(*list(mod._RUNNING))
        return out
    return asyncio.run(run())


def _fake_stt(monkeypatch, texts):
    from tools import transcription_tools
    calls = []

    def fake(path, model=None, source=None):
        calls.append(path)
        return {"success": True, "transcript": texts[len(calls) - 1]}
    monkeypatch.setattr(transcription_tools, "transcribe_audio", fake)
    return calls


ANALYSIS = {
    "language": "en",
    "summary": ["The team reviewed Q3.", "Budget is tight.", "Hiring is paused.", "Launch moves to May.",
                "Vendors to be compared."],
    "decisions": ["Launch moves to May"],
    "open_issues": ["Which vendor to choose"],
    "people": [{"name": "Ahmad", "aliases": ["أحمد"]}, {"name": "Omar", "aliases": ["عمر"]}],
    "tasks": [
        {"owner": "أحمد", "task": "Send the Q3 budget", "deadline": "Sunday"},
        {"owner": "", "owner_candidates": ["Ahmad", "عمر"], "task": "Call the vendor", "deadline": ""},
        {"owner": "Ahmad", "task": "Book the launch venue", "deadline": ""},
    ],
}


def test_short_recording_is_transcribed_briefed_and_sent_to_the_manager_only(recorder, tmp_path, monkeypatch, caplog):
    import hermes_yaml as yaml
    mod = recorder
    managed = yaml.safe_load((REPO_ROOT / "tact" / "managed-config.yaml").read_text())
    assert "tact-recorder" in managed["plugins"]["enabled"]

    stt_calls = _fake_stt(monkeypatch, [SECRET_LINE])
    llm = FakeLlm(ANALYSIS)
    monkeypatch.setattr(mod, "_LLM", llm)
    gw = FakeGateway()
    caplog.set_level(logging.DEBUG)

    # Someone off the allowlist: untouched (the gateway's own auth refuses it next), nothing sent.
    stranger = _event(media=[_wav(tmp_path / "s.wav", 1)], user="99", raw=SimpleNamespace(audio=SimpleNamespace(file_size=32000)))
    assert _dispatch(mod, FakeGateway(authorized=False), stranger) is stranger
    # A short voice note stays an ordinary chat message.
    note = _event(message_type="VOICE", media=[_wav(tmp_path / "v.ogg", 1)], mime="audio/ogg",
                  raw=SimpleNamespace(voice=SimpleNamespace(duration=20, file_size=9000)))
    assert _dispatch(mod, gw, note) is note
    assert not stt_calls and not gw.adapter.sent

    cached = _wav(tmp_path / "meeting.wav", 2)
    event = _event(text="weekly sync", media=[cached], raw=SimpleNamespace(audio=SimpleNamespace(file_size=64000)))
    assert _dispatch(mod, gw, event) is None

    started, brief, prompt = gw.adapter.sent
    assert started == mod.STARTED
    assert len(stt_calls) == 1
    home_rec = Path(mod.store.recordings_dir())
    transcript = home_rec / "1" / "transcript.txt"
    assert transcript.read_text(encoding="utf-8").strip() == SECRET_LINE
    assert not cached.exists() and not list((home_rec / "1").glob("audio*"))  # audio deleted after success
    sent_to_llm = llm.calls[0]["input"][0]["text"]
    assert SECRET_LINE in sent_to_llm and "weekly sync" in sent_to_llm

    for header in ("📝 SUMMARY", "✅ DECISIONS", "❓ OPEN ISSUES", "📋 TASKS"):
        assert header in brief
    assert "1. Ahmad → Send the Q3 budget → Sunday" in brief
    assert "3. Ahmad → Book the launch venue → NOT MENTIONED" in brief
    assert "Ahmad: 1) Send the Q3 budget (Sunday) [#1] 2) Book the launch venue (NOT MENTIONED) [#3]" in brief
    assert "confirm all" in prompt
    tasks = mod.store.tasks_for(1)
    assert [t["status"] for t in tasks] == ["pending"] * 3
    assert mod.store.get_meeting(1, "telegram", CHAT)["status"] == "pending"
    assert SECRET_LINE not in caplog.text  # transcript contents never reach the logs


def test_long_recording_is_split_and_transcribed_in_order(recorder, tmp_path, monkeypatch):
    audio = recorder.audio
    parts = [tmp_path / f"c{i}.m4a" for i in range(3)]
    for p in parts:
        p.write_bytes(b"x")
    monkeypatch.setattr(audio, "split_audio", lambda path, work: parts)
    calls = _fake_stt(monkeypatch, ["first part", "second part", "third part"])
    assert audio.transcribe_recording(tmp_path / "long.m4a") == "first part\nsecond part\nthird part"
    assert calls == [str(p) for p in parts]

    from tools import transcription_tools
    monkeypatch.setattr(transcription_tools, "transcribe_audio",
                        lambda p, m=None, s=None: {"success": p != str(parts[1]), "error": "quota", "transcript": "t"})
    with pytest.raises(audio.TranscriptionError, match="part 2 of 3"):
        audio.transcribe_recording(tmp_path / "long.m4a")


def test_ffmpeg_splits_into_mono_16k_chunks_under_the_upload_cap(recorder, tmp_path, monkeypatch):
    from tools.transcription_audio import _find_ffmpeg_binary
    from tools.transcription_common import MAX_FILE_SIZE
    if not _find_ffmpeg_binary():
        pytest.skip("ffmpeg not installed")
    audio = recorder.audio
    monkeypatch.setattr(audio, "CHUNK_SECONDS", 10)
    work = tmp_path / "work"
    work.mkdir()
    chunks = audio.split_audio(_wav(tmp_path / "long.wav", 25), work)
    assert len(chunks) == 3
    assert all(0 < c.stat().st_size <= MAX_FILE_SIZE and c.suffix == ".m4a" for c in chunks)


def test_too_large_recording_gets_the_compress_or_split_reply(recorder, monkeypatch):
    mod = recorder
    gw = FakeGateway()
    big = 30 * 1024 * 1024
    voice = _event(message_type="VOICE", raw=SimpleNamespace(voice=SimpleNamespace(file_size=big, duration=5400)))
    assert _dispatch(mod, gw, voice) is None
    doc = _event(message_type="DOCUMENT",
                 raw=SimpleNamespace(document=SimpleNamespace(file_size=big, mime_type="audio/x-m4a", file_name="m.m4a")))
    assert _dispatch(mod, gw, doc) is None
    assert gw.adapter.sent == [mod.TOO_LARGE, mod.TOO_LARGE]
    # A large non-audio document is not a recording.
    pdf = _event(message_type="DOCUMENT",
                 raw=SimpleNamespace(document=SimpleNamespace(file_size=big, mime_type="application/pdf", file_name="r.pdf")))
    assert _dispatch(mod, gw, pdf) is pdf


def test_unclear_owner_is_flagged_and_spellings_merge_into_one_person(recorder):
    fmt = recorder.brief
    for lang, unclear in (("en", "❓ UNCLEAR: Ahmad or Omar?"), ("ar", "❓ غير واضح: أحمد أو عمر؟")):
        parsed = dict(ANALYSIS, language=lang)
        if lang == "ar":
            parsed["people"] = [{"name": "أحمد", "aliases": ["Ahmad"]}, {"name": "عمر", "aliases": ["Omar"]}]
        language, brief, tasks = fmt.normalize(parsed)
        rows = [dict(t, position=i, status="pending") for i, t in enumerate(tasks, 1)]
        text = fmt.brief_text(1, "2026-09-30T10:00:00", language, brief, rows)
        assert f"2. {unclear} → Call the vendor" in text
        owner = "Ahmad" if lang == "en" else "أحمد"
        assert [t["person"] for t in tasks] == [owner, "", owner]
        grouped = fmt.by_person(rows, language)
        assert grouped[0].startswith(f"{owner}: 1) Send the Q3 budget") and "2) Book the launch venue" in grouped[0]
        assert grouped[-1].startswith(unclear.split(":")[0])  # unclear owners listed last, never guessed


def test_confirm_edit_remove_flow_with_text_replies_and_buttons(recorder, monkeypatch):
    mod = recorder
    store = mod.store
    language, brief, tasks = mod.brief.normalize(ANALYSIS)
    meeting_id = store.create_meeting("telegram", CHAT, MANAGER)
    store.save_analysis(meeting_id, language, brief, tasks)
    gw = FakeGateway()

    def reply(text):
        gw.adapter.sent.clear()
        assert _dispatch(mod, gw, _event(text=text, message_type="TEXT")) is None, text
        return gw.adapter.sent

    assert reply("remove 3") == ["❌ Task 3 removed. (2 still to confirm)"]
    edited = reply("edit 2: Omar, due Sunday")[0]
    assert "2. Omar → Call the vendor → Sunday" in edited
    # Buttons: only the manager who sent the recording may act on it.
    assert mod.handle_button(f"rec:c:{meeting_id}:1", "telegram", CHAT, "99") == ([], "Not allowed.", False)
    replies, _, done = mod.handle_button(f"rec:c:{meeting_id}:1", "telegram", CHAT, MANAGER)
    assert replies == ["✅ Task 1 confirmed. (1 still to confirm)"] and not done
    # ✏️ then a plain message edits that task.
    assert "task 2" in mod.handle_button(f"rec:e:{meeting_id}:2", "telegram", CHAT, MANAGER)[0][0]
    assert "→ Call the vendor about prices →" in reply("task: Call the vendor about prices")[0]

    final = reply("confirm all")[-1]
    assert final.startswith(f"✅ Confirmed tasks — Meeting #{meeting_id}")
    assert "Ahmad: 1) Send the Q3 budget (Sunday) [#1]" in final
    assert "Omar: 1) Call the vendor about prices (Sunday) [#2]" in final
    assert "Book the launch venue" not in final
    assert [t["status"] for t in store.tasks_for(meeting_id)] == ["confirmed", "confirmed", "removed"]
    assert store.get_meeting(meeting_id, "telegram", CHAT)["status"] == "confirmed"
    # Nothing pending any more: confirmation words are ordinary messages again.
    later = _event(text="confirm all", message_type="TEXT")
    assert _dispatch(mod, gw, later) is later

    listing = mod._cmd_recordings("")
    assert f"#{meeting_id}" in listing and "✅ confirmed" in listing
    shown = mod._cmd_brief(str(meeting_id))
    assert "✅ 1. Ahmad → Send the Q3 budget → Sunday" in shown and "❌ 3." in shown


def _short_voice(tmp_path, name, text=""):
    return _event(text=text, message_type="VOICE", media=[_wav(tmp_path / name, 1)], mime="audio/ogg",
                  raw=SimpleNamespace(voice=SimpleNamespace(duration=20, file_size=9000)))


def test_minutes_makes_the_next_short_voice_note_or_a_captioned_file_a_recording(recorder, tmp_path, monkeypatch):
    from hermes_cli.commands_platforms import telegram_menu_commands
    from hermes_cli.plugins import get_plugin_command_handler
    mod = recorder
    stt_calls = _fake_stt(monkeypatch, ["first meeting", "second meeting"])
    llm = FakeLlm(ANALYSIS)
    monkeypatch.setattr(mod, "_LLM", llm)
    gw = FakeGateway()

    assert "minutes" in dict(telegram_menu_commands()[0])  # in the Telegram command menu
    assert get_plugin_command_handler("minutes")("") == "🎙️ Send the meeting recording now."
    first = _short_voice(tmp_path, "a.ogg")
    assert _dispatch(mod, gw, first) is None
    assert gw.adapter.sent[0] == mod.STARTED and len(stt_calls) == 1
    # The arming is used up: the next short voice note is an ordinary message again.
    again = _short_voice(tmp_path, "b.ogg")
    assert _dispatch(mod, gw, again) is again and len(stt_calls) == 1

    # A file captioned /minutes needs no arming; the rest of the caption is the manager's note.
    captioned = _short_voice(tmp_path, "c.ogg", text="/minutes weekly sync")
    assert _dispatch(mod, gw, captioned) is None and len(stt_calls) == 2
    note = llm.calls[-1]["input"][0]["text"]
    assert "weekly sync" in note and "/minutes" not in note
    assert [r["id"] for r in mod.store.recent_meetings("telegram", CHAT)] == [2, 1]


def test_minutes_is_cancelled_by_a_text_message_or_after_ten_minutes(recorder, tmp_path, monkeypatch):
    mod = recorder
    stt_calls = _fake_stt(monkeypatch, ["unused"])
    gw = FakeGateway()

    mod._cmd_minutes("")
    hello = _event(text="Remind me to call Khalid at 3", message_type="TEXT")
    assert _dispatch(mod, gw, hello) is hello  # passes to the agent, silently cancelling /minutes
    voice = _short_voice(tmp_path, "a.ogg")
    assert _dispatch(mod, gw, voice) is voice

    mod._cmd_minutes("")
    armed_until = mod._minutes_armed[("telegram", CHAT)]
    assert armed_until - mod.time.monotonic() > mod.MINUTES_TTL_SECONDS - 5
    mod._minutes_armed[("telegram", CHAT)] = armed_until - mod.MINUTES_TTL_SECONDS - 1  # ten minutes pass
    late = _short_voice(tmp_path, "b.ogg")
    assert _dispatch(mod, gw, late) is late
    assert not stt_calls and not gw.adapter.sent
