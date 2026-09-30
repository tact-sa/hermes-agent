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


class FakeBot:
    """Telegram bot stand-in: every message it sends or edits, by message id."""

    def __init__(self):
        self.messages, self.edits, self._next = {}, [], 100

    async def send_message(self, chat_id, text, reply_markup=None):
        self._next += 1
        self.messages[self._next] = (text, reply_markup)
        return SimpleNamespace(message_id=self._next)

    async def edit_message_text(self, chat_id, message_id, text, reply_markup=None):
        assert message_id in self.messages, "only our own messages are edited"
        self.edits.append(message_id)
        self.messages[message_id] = (text, reply_markup)


def _buttons(markup):
    return [[(b.text, b.callback_data) for b in row] for row in markup.inline_keyboard] if markup else []


@pytest.fixture
def telegram_stub(monkeypatch):
    """The few python-telegram-bot classes the plugin builds (the test env has no telegram extra)."""
    import sys
    tg = SimpleNamespace(
        InlineKeyboardButton=lambda text, callback_data: SimpleNamespace(text=text, callback_data=callback_data),
        InlineKeyboardMarkup=lambda rows: SimpleNamespace(inline_keyboard=rows))
    ext = SimpleNamespace(CallbackQueryHandler=lambda callback, pattern: SimpleNamespace(callback=callback, pattern=pattern))
    monkeypatch.setitem(sys.modules, "telegram", tg)
    monkeypatch.setitem(sys.modules, "telegram.ext", ext)


class FakeGateway:
    def __init__(self, authorized=True, bot=False):
        self.adapter = FakeAdapter()
        if bot:
            self.adapter._bot = FakeBot()
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


def test_short_recording_is_transcribed_briefed_and_sent_to_the_manager_only(recorder, telegram_stub, tmp_path,
                                                                            monkeypatch, caplog):
    import hermes_yaml as yaml
    mod = recorder
    managed = yaml.safe_load((REPO_ROOT / "tact" / "managed-config.yaml").read_text())
    assert "tact-recorder" in managed["plugins"]["enabled"]

    stt_calls = _fake_stt(monkeypatch, [SECRET_LINE])
    llm = FakeLlm(ANALYSIS)
    monkeypatch.setattr(mod, "_LLM", llm)
    gw = FakeGateway(bot=True)
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

    started, brief = gw.adapter.sent
    assert started == mod.STARTED
    assert len(stt_calls) == 1
    home_rec = Path(mod.store.recordings_dir())
    transcript = home_rec / "1" / "transcript.txt"
    assert transcript.read_text(encoding="utf-8").strip() == SECRET_LINE
    assert not cached.exists() and not list((home_rec / "1").glob("audio*"))  # audio deleted after success
    sent_to_llm = llm.calls[0]["input"][0]["text"]
    assert SECRET_LINE in sent_to_llm and "weekly sync" in sent_to_llm

    # The brief carries no task list; the tasks follow as one message each, with their own buttons.
    for header in ("📝 SUMMARY", "✅ DECISIONS", "❓ OPEN ISSUES"):
        assert header in brief
    assert brief.endswith("\n\n📋 Tasks (3) in the following messages") and "Send the Q3 budget" not in brief
    (card1, m1), (card2, m2), (card3, _), (all_text, all_markup) = gw.adapter._bot.messages.values()
    assert card1 == "📋 Task 1 of 3\n👤 Ahmad\n📌 Send the Q3 budget\n📅 Sunday"
    assert _buttons(m1) == [[("✅ Confirm", "rec:c:1:1"), ("✏️ Edit", "rec:e:1:1"), ("❌ Cancel", "rec:r:1:1")]]
    assert card2 == "📋 Task 2 of 3\n👤 ❓ UNCLEAR: Ahmad or Omar?\n📌 Call the vendor\n📅 NOT MENTIONED"
    assert card3.startswith("📋 Task 3 of 3\n👤 Ahmad")
    assert _buttons(all_markup) == [[("✅ Confirm all (3)", "rec:a:1:0")]] and "confirm 2" in all_text
    assert not any("[#" in text for text, _ in gw.adapter._bot.messages.values())
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
    for lang, unclear, missing in (("en", "❓ UNCLEAR: Ahmad or Omar?", "NOT MENTIONED"),
                                   ("ar", "❓ غير واضح: أحمد أو عمر؟", "غير مذكور")):
        parsed = dict(ANALYSIS, language=lang)
        if lang == "ar":
            parsed["people"] = [{"name": "أحمد", "aliases": ["Ahmad"]}, {"name": "عمر", "aliases": ["Omar"]}]
        language, brief, tasks = fmt.normalize(parsed)
        owner = "Ahmad" if lang == "en" else "أحمد"
        assert [t["person"] for t in tasks] == [owner, "", owner]
        rows = [dict(t, position=i, status="confirmed") for i, t in enumerate(tasks, 1)]
        assert f"👤 {unclear}\n📌 Call the vendor\n📅 {missing}" in fmt.task_card(rows[1], 3, language)
        # The final summary: one labelled block per task under its own number; no brackets.
        title, name, task, due = (("✅ Confirmed tasks — Meeting 1", "Name", "Task", "Deadline") if lang == "en"
                                  else ("✅ المهام المؤكدة — اجتماع 1", "الاسم", "المهمة", "الموعد"))
        assert fmt.final_text(1, language, rows) == (
            f"{title}\n\n1.\n👤 {name}: {owner}\n📌 {task}: Send the Q3 budget\n📅 {due}: Sunday"
            f"\n\n2.\n👤 {name}: {unclear.rstrip('?؟')}\n📌 {task}: Call the vendor\n📅 {due}: {missing}"
            f"\n\n3.\n👤 {name}: {owner}\n📌 {task}: Book the launch venue\n📅 {due}: {missing}")
    assert fmt.brief_text(1, "2026-09-30T10:00:00", "ar", brief, 3).endswith("📋 المهام (3) في الرسائل التالية")


def test_text_replies_edit_labelled_fields_and_cancel(recorder, monkeypatch):
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

    def task(n):
        return store.tasks_for(meeting_id)[n - 1]

    # No task cards to edit on this platform: each change is confirmed with a short text reply.
    assert reply("إلغاء 3") == ["❌ Task 3 cancelled. (2 still to confirm)"]
    # Without a label nothing is guessed: the manager is asked which field.
    assert "Which field of task 2?" in reply("edit 2: Omar, due Sunday")[0]
    assert (task(2)["person"], task(2)["deadline"]) == ("", "")
    # Labelled, any order, several fields; a name turns the unclear task into that person's.
    assert "👤 Omar\n📌 Call the vendor\n📅 Thursday" in reply("edit 2: deadline: Thursday, name: Omar")[0]
    assert task(2)["candidates"] == [] and task(2)["person_key"] == "omar"
    reply("تعديل 1: الموعد: يوم الخميس، والمهمة: إرسال الميزانية")
    assert (task(1)["person"], task(1)["task"], task(1)["deadline"]) == ("Ahmad", "إرسال الميزانية", "يوم الخميس")
    assert reply("confirm 1") == ["✅ Task 1 confirmed. (1 still to confirm)"]

    # "حذف" (the old wording) still cancels; with nothing pending the final summary follows.
    cancelled, final = reply("حذف 2")
    assert cancelled == "❌ Task 2 cancelled."
    assert final == (
        f"✅ Confirmed tasks — Meeting {meeting_id}\n\n"
        "1.\n👤 Name: Ahmad\n📌 Task: إرسال الميزانية\n📅 Deadline: يوم الخميس\n\n"
        "❌ Cancelled tasks\n\n"
        "2.\n👤 Name: Omar\n📌 Task: Call the vendor\n📅 Deadline: Thursday\n\n"
        "3.\n👤 Name: Ahmad\n📌 Task: Book the launch venue\n📅 Deadline: NOT MENTIONED")
    assert [t["status"] for t in store.tasks_for(meeting_id)] == ["confirmed", "removed", "removed"]
    assert store.get_meeting(meeting_id, "telegram", CHAT)["status"] == "confirmed"
    # Nothing pending any more: confirmation words are ordinary messages again.
    later = _event(text="confirm all", message_type="TEXT")
    assert _dispatch(mod, gw, later) is later

    listing = mod._cmd_recordings("")
    assert f"#{meeting_id}" in listing and "✅ confirmed" in listing
    shown = mod._cmd_brief(str(meeting_id))  # handled tasks: the same block layout
    assert final in shown and "📋 Task 1 of 3" not in shown and "[#" not in shown


def test_labelled_edit_text_is_split_by_label_only(recorder):
    parse = recorder.confirm.parse_labelled
    assert parse("name: Omar, deadline: Thursday") == {"person": "Omar", "deadline": "Thursday"}
    assert parse("الاسم: عمر، الموعد: الخميس") == {"person": "عمر", "deadline": "الخميس"}
    assert parse("Due: end of month; Task: send the report, then call") == {
        "deadline": "end of month", "task": "send the report, then call"}
    assert parse("الاسم: احمد او عمر (يحدد لاحقا)") == {"person": "احمد او عمر (يحدد لاحقا)"}
    for unlabelled in ("يوم الخميس", "Omar, due Sunday", "call Omar: urgent"):
        assert parse(unlabelled) == {}, unlabelled


def _seed(mod, n_tasks=3):
    language, brief, tasks = mod.brief.normalize(ANALYSIS)
    if n_tasks > len(tasks):
        tasks = [dict(tasks[0], task=f"Task number {i}") for i in range(1, n_tasks + 1)]
    meeting_id = mod.store.create_meeting("telegram", CHAT, MANAGER)
    mod.store.save_analysis(meeting_id, language, brief, tasks)
    return meeting_id


def test_task_cards_are_edited_in_place_by_buttons_and_text_replies(recorder, telegram_stub):
    mod = recorder
    meeting_id = _seed(mod)
    gw = FakeGateway(bot=True)
    bot = gw.adapter._bot
    # /brief <id> shows the same layout: the brief, a card per task, then "Confirm all".
    assert _dispatch(mod, gw, _event(text=f"/brief {meeting_id}", message_type="TEXT")) is None
    assert gw.adapter.sent[0].endswith("📋 Tasks (3) in the following messages")
    card1, card2, card3, all_id = bot.messages
    handlers = []
    mod._telegram_handlers(SimpleNamespace(add_handler=handlers.append), gw.adapter)

    def press(data, message_id, user=MANAGER):
        toasts = []

        async def answer(text=None):
            toasts.append(text)
        query = SimpleNamespace(data=data, from_user=SimpleNamespace(id=int(user)), answer=answer,
                                message=SimpleNamespace(chat=SimpleNamespace(id=int(CHAT)), message_id=message_id))
        asyncio.run(handlers[0].callback(SimpleNamespace(callback_query=query), None))
        return toasts

    def reply(text):
        gw.adapter.sent.clear()
        assert _dispatch(mod, gw, _event(text=text, message_type="TEXT")) is None, text
        return gw.adapter.sent

    def task(n):
        return mod.store.tasks_for(meeting_id)[n - 1]

    normal = [[("✅ Confirm", f"rec:c:{meeting_id}:2"), ("✏️ Edit", f"rec:e:{meeting_id}:2"),
               ("❌ Cancel", f"rec:r:{meeting_id}:2")]]
    picker = [[("👤 Name", f"rec:f:{meeting_id}:2:n"), ("📌 Task", f"rec:f:{meeting_id}:2:t"),
               ("📅 Deadline", f"rec:f:{meeting_id}:2:d")], [("↩️ Back", f"rec:b:{meeting_id}:2")]]

    gw.adapter.sent.clear()
    press(f"rec:e:{meeting_id}:2", card2)  # ✏️: the card itself offers the fields
    assert _buttons(bot.messages[card2][1]) == picker and gw.adapter.sent == []
    # Each field button: the next message replaces only that field, exactly as typed.
    for code, prompt, value, field in (
            ("n", "Send the new name for task 2", "عمر (يحدد لاحقا)", "person"),
            ("d", "Send the new deadline for task 2", "يوم الخميس", "deadline"),
            ("t", "Send the new task text for task 2", "الاسم: احمد او عمر (يحدد لاحقا)", "task")):
        before = task(2)
        press(f"rec:e:{meeting_id}:2", card2)
        press(f"rec:f:{meeting_id}:2:{code}", card2)
        assert gw.adapter.sent == [prompt]
        assert reply(f"  {value} ") == []  # the card is edited instead of a new message
        after = task(2)
        assert after[field] == value
        assert {k: after[k] for k in ("person", "task", "deadline") if k != field} == \
            {k: before[k] for k in ("person", "task", "deadline") if k != field}
        assert _buttons(bot.messages[card2][1]) == normal
    assert bot.messages[card2][0] == ("📋 Task 2 of 3\n👤 عمر (يحدد لاحقا)\n📌 الاسم: احمد او عمر (يحدد لاحقا)"
                                      "\n📅 يوم الخميس")
    assert task(2)["candidates"] == []  # the unclear task now belongs to that person
    # ↩️ puts the normal buttons back and nothing is waiting for a value.
    press(f"rec:e:{meeting_id}:2", card2)
    press(f"rec:b:{meeting_id}:2", card2)
    assert _buttons(bot.messages[card2][1]) == normal
    hello = _event(text="hello", message_type="TEXT")
    assert _dispatch(mod, gw, hello) is hello

    assert press(f"rec:c:{meeting_id}:1", card1, user="99") == ["Not allowed."]
    press(f"rec:c:{meeting_id}:1", card1)
    assert bot.messages[card1] == ("📋 Task 1 of 3\n👤 Ahmad\n📌 Send the Q3 budget\n📅 Sunday\n\n✅ Confirmed", None)
    reply("إلغاء 3")  # a text reply edits that task's card, like its button would
    assert bot.messages[card3][0].endswith("\n\n❌ Cancelled") and bot.messages[card3][1] is None
    assert _buttons(bot.messages[all_id][1]) == [[("✅ Confirm all (1)", f"rec:a:{meeting_id}:0")]]

    gw.adapter.sent.clear()
    press(f"rec:a:{meeting_id}:0", all_id)
    assert bot.messages[card2][0].endswith("✅ Confirmed") and bot.messages[card2][1] is None
    assert bot.messages[all_id] == ("✅ All tasks handled.", None)
    (final,) = gw.adapter.sent
    assert final.startswith(f"✅ Confirmed tasks — Meeting {meeting_id}\n\n1.\n👤 Name: Ahmad\n")
    assert "\n\n❌ Cancelled tasks\n\n3.\n👤 Name: Ahmad\n📌 Task: Book the launch venue" in final
    # Arabic meetings get Arabic buttons, with "إلغاء" for cancel.
    ar_picker = mod.present._picker_markup(meeting_id, 2, "ar")
    assert [b for b, _ in _buttons(ar_picker)[0]] + [b for b, _ in _buttons(ar_picker)[1]] == [
        "👤 الاسم", "📌 المهمة", "📅 الموعد", "↩️ رجوع"]
    assert [b for b, _ in _buttons(mod.present._card_markup(meeting_id, dict(task(1), status="pending"), "ar"))[0]] == [
        "✅ تأكيد", "✏️ تعديل", "❌ إلغاء"]


def test_more_than_fifteen_tasks_go_in_one_list_with_text_replies(recorder, telegram_stub):
    mod = recorder
    meeting_id = _seed(mod, n_tasks=16)
    gw = FakeGateway(bot=True)
    assert _dispatch(mod, gw, _event(text=f"/brief {meeting_id}", message_type="TEXT")) is None
    brief, tasks = gw.adapter.sent
    assert brief.endswith("📋 Tasks (16) in the next message") and not gw.adapter._bot.messages
    assert tasks.startswith("📋 Tasks (16)") and "📋 Task 16 of 16\n👤 Ahmad\n📌 Task number 16" in tasks
    assert '"confirm 2"' in tasks


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


AR_TRANSCRIPT = "اتفقنا أن يرسل أحمد الميزانية"


def test_brief_json_is_recovered_from_code_fences_and_prose(recorder, monkeypatch):
    fmt = recorder.brief
    body = json.dumps(ANALYSIS, ensure_ascii=False)
    for text in (f"```json\n{body}\n```", f"Here is the brief you asked for:\n{body}\nLet me know if you need more."):
        llm = FakeLlm(None)
        llm.acomplete_structured = lambda _t=text, **kw: _result(llm, kw, _t)
        language, brief, tasks = asyncio.run(fmt.analyze(llm, "transcript"))
        assert language == "en" and brief["decisions"] == ["Launch moves to May"] and len(tasks) == 3
        assert len(llm.calls) == 1 and llm.calls[0]["max_tokens"] >= 8000

    # Only some keys (strict schema failed): the rest is filled safely, never invented.
    partial = '{"summary": ["اجتماع قصير"], "tasks": [{"task": "إرسال الميزانية"}]}'
    llm = FakeLlm(None)
    llm.acomplete_structured = lambda **kw: _result(llm, kw, partial)
    language, brief, tasks = asyncio.run(fmt.analyze(llm, AR_TRANSCRIPT))
    card = fmt.task_card(dict(tasks[0], position=1, status="pending"), 1, language)
    assert language == "ar" and brief["decisions"] == [] and brief["open_issues"] == []
    assert card == "📋 مهمة 1 من 1\n👤 ❓ غير واضح\n📌 إرسال الميزانية\n📅 غير مذكور"


async def _result(llm, kw, text):
    llm.calls.append(kw)
    return SimpleNamespace(parsed=None, text=text, finish_reason="stop")


def test_broken_json_is_retried_then_brief_retry_rewrites_it_from_the_transcript(recorder, tmp_path, monkeypatch, caplog):
    import hermes_yaml as yaml
    mod = recorder
    home = Path(mod.store.recordings_dir()).parent
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": ["tact-recorder"],
                    "entries": {"tact-recorder": {"llm": {"allow_model_override": True}}}},
        "tact_recorder": {"brief_model": "anthropic/claude-sonnet-4.5"}}))
    _fake_stt(monkeypatch, [SECRET_LINE])
    broken = '{"summary": ["The team reviewed Q3", "tasks": [{"owner": "Ahmad", "task": "Send the'
    replies = []  # model outputs, consumed in order by the real ctx.llm through an injected caller
    calls = []

    async def caller(**kw):
        calls.append(kw)
        text = replies.pop(0)
        choice = SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="length" if text == broken else "stop")
        return "openrouter", kw["model_override"] or "main", SimpleNamespace(choices=[choice], usage=None)
    monkeypatch.setattr(mod._LLM, "_async_caller", caller)
    gw = FakeGateway()
    caplog.set_level(logging.DEBUG)
    failed = "⚠️ Meeting #1: the brief could not be written. Send /brief 1 retry to try again."

    replies[:] = [broken, broken]
    event = _event(media=[_wav(tmp_path / "m.wav", 1)], raw=SimpleNamespace(audio=SimpleNamespace(file_size=32000)))
    assert _dispatch(mod, gw, event) is None
    assert gw.adapter.sent == [mod.STARTED, failed]
    sent_text = [json.dumps(c["messages"], ensure_ascii=False) for c in calls]
    assert len(calls) == 2 and mod.brief.STRICT_RETRY not in sent_text[0] and mod.brief.STRICT_RETRY in sent_text[1]
    assert all(c["model_override"] == "anthropic/claude-sonnet-4.5" and c["max_tokens"] >= 8000 for c in calls)
    assert f"{len(broken)} chars, finish_reason=length" in caplog.text
    assert SECRET_LINE not in caplog.text and "Send the" not in caplog.text  # no transcript or model output
    assert mod._cmd_brief("1") == failed

    def retry():
        gw.adapter.sent.clear()
        assert _dispatch(mod, gw, _event(text="/brief 1 retry", message_type="TEXT")) is None
        return gw.adapter.sent

    replies[:] = [broken, broken]  # the retry fails too: the same message, so the manager can try again
    assert retry() == ["📝 Writing the brief for meeting #1 again…", failed]
    replies[:] = [broken, "```json\n" + json.dumps(ANALYSIS) + "\n```"]
    started, brief, tasks = retry()  # no inline buttons here: the tasks come as one list
    assert brief.endswith("📋 Tasks (3) in the next message")
    assert "📋 Task 1 of 3\n👤 Ahmad\n📌 Send the Q3 budget\n📅 Sunday" in tasks and "confirm all" in tasks
    assert SECRET_LINE in json.dumps(calls[-1]["messages"])  # rebuilt from the saved transcript
    assert mod.store.get_meeting(1, "telegram", CHAT)["status"] == "pending"
    assert retry() == ["Meeting #1 already has a brief. Show it with /brief 1."]


class ScriptedLlm:
    """Answers each structured call from ``replies[schema_name]`` and records the calls."""

    def __init__(self, **replies):
        self.replies, self.calls = replies, []

    async def acomplete_structured(self, **kw):
        self.calls.append(kw)
        reply = self.replies[kw["schema_name"]]
        return SimpleNamespace(parsed=reply, text=json.dumps(reply, ensure_ascii=False), finish_reason="stop")


LONG_AR = ("قررنا خفض ميزانية السفر عشرين بالمئة وموافقين على الإطلاق يوم خمسة عشر نوفمبر. " * 30)


def test_decisions_and_open_issues_survive_objects_nested_lists_and_other_keys(recorder):
    fmt = recorder.brief
    _, brief, _ = fmt.normalize({
        "summary": "اجتماع المبيعات",
        "decisions": [{"decision": "خفض ميزانية السفر 20%"}, [{"text": "الإطلاق في 15 نوفمبر"}]],
        "open_questions": [["خصم 15% للعميل"], {"topic": "تأخير التصميم", "status": "pending"}]})
    assert brief == {"summary": ["اجتماع المبيعات"],
                     "decisions": ["خفض ميزانية السفر 20%", "الإطلاق في 15 نوفمبر"],
                     "open_issues": ["خصم 15% للعميل", "تأخير التصميم — pending"]}
    _, brief, _ = fmt.normalize({"القرارات": ["خفض السفر"], "القضايا_المفتوحة": [{"القضية": "توظيف مطور"}]})
    assert brief["decisions"] == ["خفض السفر"] and brief["open_issues"] == ["توظيف مطور"]


def test_empty_decisions_and_open_issues_get_one_followup_call(recorder, caplog):
    fmt = recorder.brief
    caplog.set_level(logging.INFO)
    main = {"language": "ar", "summary": ["قررنا خفض السفر"], "decisions": [], "open_issues": [], "tasks": []}
    llm = ScriptedLlm(meeting_brief=main, meeting_decisions={
        "decisions": ["خفض ميزانية السفر 20%", "الإطلاق في 15 نوفمبر"], "open_issues": [{"issue": "توظيف مطور"}]})
    _, brief, _ = asyncio.run(fmt.analyze(llm, LONG_AR))
    assert brief["decisions"] == ["خفض ميزانية السفر 20%", "الإطلاق في 15 نوفمبر"]
    assert brief["open_issues"] == ["توظيف مطور"]
    assert [c["schema_name"] for c in llm.calls] == ["meeting_brief", "meeting_decisions"]
    # Counts, key names and item types are logged; the meeting's content is not.
    assert "brief counts: decisions=2 open_issues=1 tasks=0 with_deadline=0 follow-ups=decisions" in caplog.text
    assert "decisions=list[empty]" in caplog.text and "open_issues=list[dictx1]{issue}" in caplog.text
    assert "خفض" not in caplog.text and "توظيف" not in caplog.text
    # A short meeting may genuinely decide nothing: no follow-up call then.
    llm = ScriptedLlm(meeting_brief=main)
    asyncio.run(fmt.analyze(llm, "اجتماع قصير"))
    assert len(llm.calls) == 1


def test_tasks_accept_other_keys_and_deadlines_glued_to_the_task(recorder):
    _, _, tasks = recorder.brief.normalize({"tasks": [
        {"description": "Send the deck", "assignee": "Omar", "due_date": "Tuesday"},
        {"المهمة": "مراجعة العقد", "المسؤول": "أحمد", "الموعد_النهائي": "يوم الأحد القادم"},
        {"task": "تجهيز العرض التقديمي لشركة النخبة وترتيب اجتماع معهم قبل نهاية الأسبوع القادم", "owner": "سارة"},
        {"action": "إرسال التقرير", "responsible": ["أحمد", "عمر"], "when": "بكرة"},
        "مراجعة خطة الأسبوع"]})
    assert [(t["person"], t["task"], t["deadline"], t["candidates"]) for t in tasks] == [
        ("Omar", "Send the deck", "Tuesday", []),
        ("أحمد", "مراجعة العقد", "يوم الأحد القادم", []),
        ("سارة", "تجهيز العرض التقديمي لشركة النخبة وترتيب اجتماع معهم", "قبل نهاية الأسبوع القادم", []),
        ("", "إرسال التقرير", "بكرة", ["أحمد", "عمر"]),  # two owners named: unclear, not guessed
        ("", "مراجعة خطة الأسبوع", "", [])]


def test_missing_deadlines_get_one_followup_call(recorder):
    fmt = recorder.brief
    main = {"language": "ar", "summary": ["x"], "decisions": ["d"],
            "tasks": [{"owner": "أحمد", "task": "إرسال التقرير"}, {"owner": "عمر", "task": "تنظيم الأرشيف"},
                      {"owner": "سارة", "task": "حجز القاعة"}]}
    llm = ScriptedLlm(meeting_brief=main, meeting_deadlines={"deadlines": [
        {"n": 1, "deadline": "بكرة"}, {"n": 2, "deadline": ""}, {"n": 3, "deadline": "يوم الثلاثاء"}]})
    _, _, tasks = asyncio.run(fmt.analyze(llm, "يا أحمد أرسل التقرير بكرة، وسارة احجزي القاعة يوم الثلاثاء"))
    assert [t["deadline"] for t in tasks] == ["بكرة", "", "يوم الثلاثاء"]
    assert [c["schema_name"] for c in llm.calls] == ["meeting_brief", "meeting_deadlines"]
    assert "1. أحمد — إرسال التقرير" in llm.calls[1]["input"][0]["text"]
    # No deadline words in the meeting: nothing to recover, no follow-up call.
    llm = ScriptedLlm(meeting_brief=main)
    asyncio.run(fmt.analyze(llm, "ناقشنا تنظيم الأرشيف وحجز القاعة"))
    assert len(llm.calls) == 1
