"""tact-recorder: a meeting recording becomes a transcript, a brief and tasks the manager confirms.

Loads the bundled plugin through the real ``PluginManager`` against a temp HERMES_HOME and drives
the gateway's own ``pre_gateway_dispatch`` step with a stand-in gateway/adapter. STT
(``tools.transcription_tools.transcribe_audio``) and the LLM (``ctx.llm``) are mocked; the SQLite
store, the file handling and (when installed) ffmpeg's chunking are the real ones.
"""

import asyncio
import json
import re
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
        self.messages, self.edits, self.to, self._next = {}, [], {}, 100

    async def send_message(self, chat_id, text, reply_markup=None):
        self._next += 1
        self.messages[self._next] = (text, reply_markup)
        self.to[self._next] = str(chat_id)
        return SimpleNamespace(message_id=self._next)

    def sent_to(self, chat_id):
        return [self.messages[i][0] for i, c in self.to.items() if c == str(chat_id)]

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
    class Filter:
        def __init__(self, *parts):
            self.parts = parts

        def __and__(self, other):
            return Filter(self, other)
    ext = SimpleNamespace(
        CallbackQueryHandler=lambda callback, pattern: SimpleNamespace(callback=callback, pattern=pattern),
        MessageHandler=lambda flt, callback: SimpleNamespace(callback=callback, filters=flt),
        filters=SimpleNamespace(Regex=lambda pattern: Filter(pattern), ChatType=SimpleNamespace(PRIVATE=Filter())))
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
    assert card1 == "📋 Task 1 of 3\n👤 Ahmad\n📌 Send the Q3 budget\n📅 Sunday\n📞 unknown"
    assert _buttons(m1) == [[("✅ Confirm", "rec:c:1:1"), ("✏️ Edit", "rec:e:1:1"), ("❌ Cancel", "rec:r:1:1")]]
    assert card2 == "📋 Task 2 of 3\n👤 ❓ UNCLEAR: Ahmad or Omar?\n📌 Call the vendor\n📅 NOT MENTIONED\n📞 unknown"
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
        title, name, task, due, how, unknown = (
            ("✅ Confirmed tasks — Meeting 1", "Name", "Task", "Deadline", "Contact", "unknown") if lang == "en"
            else ("✅ المهام المؤكدة — اجتماع 1", "الاسم", "المهمة", "الموعد", "التواصل", "غير معروف"))
        assert fmt.final_text(1, language, rows) == (
            f"{title}\n\n1.\n👤 {name}: {owner}\n📌 {task}: Send the Q3 budget\n📅 {due}: Sunday\n📞 {how}: {unknown}"
            f"\n\n2.\n👤 {name}: {unclear.rstrip('?؟')}\n📌 {task}: Call the vendor\n📅 {due}: {missing}"
            f"\n📞 {how}: {unknown}"
            f"\n\n3.\n👤 {name}: {owner}\n📌 {task}: Book the launch venue\n📅 {due}: {missing}\n📞 {how}: {unknown}")
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
        "1.\n👤 Name: Ahmad\n📌 Task: إرسال الميزانية\n📅 Deadline: يوم الخميس\n📞 Contact: unknown\n\n"
        "❌ Cancelled tasks\n\n"
        "2.\n👤 Name: Omar\n📌 Task: Call the vendor\n📅 Deadline: Thursday\n📞 Contact: unknown\n\n"
        "3.\n👤 Name: Ahmad\n📌 Task: Book the launch venue\n📅 Deadline: NOT MENTIONED\n📞 Contact: unknown")
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
    picker = [[("👤 Name", f"rec:f:{meeting_id}:2:n"), ("📌 Task", f"rec:f:{meeting_id}:2:t")],
              [("📅 Deadline", f"rec:f:{meeting_id}:2:d"), ("📞 Contact", f"rec:f:{meeting_id}:2:k")],
              [("↩️ Back", f"rec:b:{meeting_id}:2")]]

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
                                      "\n📅 يوم الخميس\n📞 unknown")
    assert task(2)["candidates"] == []  # the unclear task now belongs to that person
    # ↩️ puts the normal buttons back and nothing is waiting for a value.
    press(f"rec:e:{meeting_id}:2", card2)
    press(f"rec:b:{meeting_id}:2", card2)
    assert _buttons(bot.messages[card2][1]) == normal
    hello = _event(text="hello", message_type="TEXT")
    assert _dispatch(mod, gw, hello) is hello

    assert press(f"rec:c:{meeting_id}:1", card1, user="99") == ["Not allowed."]
    press(f"rec:c:{meeting_id}:1", card1)
    assert bot.messages[card1] == ("📋 Task 1 of 3\n👤 Ahmad\n📌 Send the Q3 budget\n📅 Sunday\n📞 unknown"
                                   "\n\n✅ Confirmed", None)
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
    assert [b for row in _buttons(ar_picker) for b, _ in row] == [
        "👤 الاسم", "📌 المهمة", "📅 الموعد", "📞 التواصل", "↩️ رجوع"]
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
    assert card == "📋 مهمة 1 من 1\n👤 ❓ غير واضح\n📌 إرسال الميزانية\n📅 غير مذكور\n📞 غير معروف"


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
        ("أحمد، عمر", "إرسال التقرير", "بكرة", []),  # two owners named: joint owners (sent to both)
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


def test_assignments_filed_as_decisions_become_tasks(recorder):
    _, brief, tasks = recorder.brief.normalize({
        "language": "ar", "summary": ["x"],
        "people": [{"name": "خالد"}, {"name": "عمر"}, {"name": "سارة"}],
        "decisions": ["تكليف خالد بإعداد تقرير المبيعات قبل يوم الخميس", "خفض ميزانية السفر 20%",
                      "تكليف عمر بالتواصل مع شركة النخبة بكرة", "تكليف سارة بإرسال الدعوات"],
        "tasks": [{"owner": "", "owner_candidates": ["سارة", "عمر"], "task": "سارة: إرسال الدعوات"},
                  {"owner": "", "owner_candidates": ["خالد", "عمر"], "task": "الاتصال بالمورد"}]})
    assert brief["decisions"] == ["خفض ميزانية السفر 20%"]
    assert [(t["person"], t["task"], t["deadline"], t["candidates"]) for t in tasks] == [
        ("سارة", "إرسال الدعوات", "", []),  # the text names its one owner: not unclear
        ("", "الاتصال بالمورد", "", ["خالد", "عمر"]),  # genuinely unclear stays unclear
        ("خالد", "إعداد تقرير المبيعات", "قبل يوم الخميس", []),
        ("عمر", "التواصل مع شركة النخبة", "بكرة", [])]  # "تكليف سارة ..." was already task 1
    _, brief, tasks = recorder.brief.normalize({
        "people": [{"name": "Omar"}], "tasks": [],
        "decisions": ["Omar will send the deck by Tuesday", "Launch moves to May", "Task force created"]})
    assert brief["decisions"] == ["Launch moves to May", "Task force created"]
    assert [(t["person"], t["task"], t["deadline"]) for t in tasks] == [("Omar", "send the deck", "by Tuesday")]


class RoutedLlm(ScriptedLlm):
    """ScriptedLlm whose calls can fail for one model (as an unavailable OpenRouter model would)."""

    def __init__(self, broken_model=None, **replies):
        super().__init__(**replies)
        self.broken_model = broken_model

    async def acomplete_structured(self, **kw):
        if kw.get("model") and kw["model"] == self.broken_model:
            self.calls.append(kw)
            raise RuntimeError("model unavailable")
        return await super().acomplete_structured(**kw)


GEMINI = "google/gemini-3.8-flash"
GAPPY = {"language": "ar", "summary": ["x"], "decisions": [], "open_issues": [],
         "tasks": [{"owner": "أحمد", "task": "إرسال التقرير"}]}
GAPPY_TRANSCRIPT = LONG_AR + " يا أحمد أرسل التقرير بكرة"


def test_brief_model_writes_the_brief_and_its_follow_ups(recorder):
    import hermes_yaml as yaml
    managed = yaml.safe_load((REPO_ROOT / "tact" / "managed-config.yaml").read_text())
    # The deployment sends only the brief to its own model, which the plugin is trusted to choose.
    assert managed["tact_recorder"]["brief_model"]
    assert managed["plugins"]["entries"]["tact-recorder"]["llm"]["allow_model_override"] is True

    llm = RoutedLlm(meeting_brief=GAPPY, meeting_decisions={"decisions": ["d"], "open_issues": []},
                    meeting_deadlines={"deadlines": [{"n": 1, "deadline": "بكرة"}]})
    _, brief, tasks = asyncio.run(recorder.brief.analyze(llm, GAPPY_TRANSCRIPT, model=GEMINI))
    assert [c["schema_name"] for c in llm.calls] == ["meeting_brief", "meeting_decisions", "meeting_deadlines"]
    assert all(c["model"] == GEMINI for c in llm.calls)
    assert brief["decisions"] == ["d"] and tasks[0]["deadline"] == "بكرة"

    # The strict retry goes to the brief model too.
    llm = RoutedLlm(meeting_brief=GAPPY)
    replies = iter([SimpleNamespace(parsed=None, text="not json", finish_reason="stop"),
                    SimpleNamespace(parsed=dict(GAPPY, decisions=["d"], tasks=[]), text="", finish_reason="stop")])

    async def flaky(**kw):
        llm.calls.append(kw)
        return next(replies)
    llm.acomplete_structured = flaky
    asyncio.run(recorder.brief.analyze(llm, "short", model=GEMINI))
    assert len(llm.calls) == 2 and all(c["model"] == GEMINI for c in llm.calls)
    assert recorder.brief.STRICT_RETRY in llm.calls[1]["instructions"]


def test_a_failing_brief_model_falls_back_to_the_main_model(recorder, caplog):
    caplog.set_level(logging.INFO)
    llm = RoutedLlm(broken_model=GEMINI, meeting_brief=GAPPY,
                    meeting_decisions={"decisions": ["d"], "open_issues": []},
                    meeting_deadlines={"deadlines": [{"n": 1, "deadline": "بكرة"}]})
    _, brief, tasks = asyncio.run(recorder.brief.analyze(llm, GAPPY_TRANSCRIPT, model=GEMINI))
    assert [(c["schema_name"], c.get("model")) for c in llm.calls] == [
        ("meeting_brief", GEMINI), ("meeting_brief", None),  # one failure, then the main model for the rest
        ("meeting_decisions", None), ("meeting_deadlines", None)]
    assert brief["decisions"] == ["d"] and tasks[0]["deadline"] == "بكرة"
    assert f"brief model {GEMINI} failed (RuntimeError); falling back to the main chat model" in caplog.text
    assert "التقرير" not in caplog.text


# -- contacts, invites and sending ------------------------------------------------------------------

def _press(mod, adapter, data, message_id, user=MANAGER):
    """A button press through the plugin's real Telegram callback handler."""
    handlers = []
    mod._telegram_handlers(SimpleNamespace(add_handler=handlers.append), adapter)
    toasts = []

    async def answer(text=None):
        toasts.append(text)
    query = SimpleNamespace(data=data, from_user=SimpleNamespace(id=int(user)), answer=answer,
                            message=SimpleNamespace(chat=SimpleNamespace(id=int(CHAT)), message_id=message_id))
    asyncio.run(handlers[0].callback(SimpleNamespace(callback_query=query), None))
    return toasts


def test_contacts_come_from_the_meeting_or_the_book_or_read_unknown(recorder):
    mod = recorder
    mod.contacts.save_contact("سارة", "sara@tact.sa")
    llm = ScriptedLlm(meeting_brief={"language": "ar", "summary": ["x"], "decisions": ["d"], "tasks": [
        {"owner": "فهد", "task": "إرسال العرض", "deadline": "بكرة", "contact": "fahad@tact.sa"},
        {"owner": "عمر", "task": "حجز القاعة", "deadline": "بكرة", "contact": "omar@invented.com"},
        {"owner": "ساره", "task": "مراجعة العقد", "deadline": "بكرة"},
        {"owner": "خالد", "task": "تجهيز الميزانية", "deadline": "بكرة"}]})
    transcript = "يا فهد أرسل العرض بكرة على fahad@tact.sa وعمر يحجز القاعة"
    lang, _, tasks = asyncio.run(mod.brief.analyze(llm, transcript))
    assert [t["contact"] for t in tasks] == ["fahad@tact.sa", "", "", ""]  # never one the meeting didn't say
    rows = mod.contacts.annotate([dict(t, position=i, status="pending") for i, t in enumerate(tasks, 1)], lang)
    assert [mod.brief.contact_text(t, lang) for t in rows] == ["fahad@tact.sa", "غير معروف", "sara@tact.sa", "غير معروف"]
    assert "📞 sara@tact.sa" in mod.brief.task_card(rows[2], 4, lang)  # the book, matched across spellings
    assert "📞 التواصل: غير معروف" in mod.brief.task_block(rows[3], lang)


def test_editing_a_contact_validates_it_and_offers_to_save_it(recorder, telegram_stub):
    mod = recorder
    meeting_id = _seed(mod)
    gw = FakeGateway(bot=True)
    bot = gw.adapter._bot
    assert _dispatch(mod, gw, _event(text=f"/brief {meeting_id}", message_type="TEXT")) is None
    card1 = next(i for i, (text, _) in bot.messages.items() if text.startswith("📋 Task 1 of 3"))

    def reply(text):
        gw.adapter.sent.clear()
        assert _dispatch(mod, gw, _event(text=text, message_type="TEXT")) is None, text
        return gw.adapter.sent

    _press(mod, gw.adapter, f"rec:e:{meeting_id}:1", card1)
    _press(mod, gw.adapter, f"rec:f:{meeting_id}:1:k", card1)
    assert gw.adapter.sent[-1] == "Send the contact for task 1: an email address or @username"
    assert reply("hello there") == ["📞 That is not a contact. Send an email address, a @username or a phone number."]
    _press(mod, gw.adapter, f"rec:e:{meeting_id}:1", card1)
    _press(mod, gw.adapter, f"rec:f:{meeting_id}:1:k", card1)
    reply("@ahmad_k")
    assert "📞 @ahmad_k (⏳ not registered in the bot yet)" in bot.messages[card1][0]
    question = max(bot.messages)
    assert bot.messages[question][0] == "Save this contact for Ahmad for future meetings?"
    assert mod.contacts.find("Ahmad") is None  # only saved when the manager says so
    _press(mod, gw.adapter, f"rec:s:y:{meeting_id}:1", question)
    assert mod.contacts.find("Ahmad")["telegram_username"] == "@ahmad_k"
    assert "phone numbers can't be messaged yet" in reply("edit 2: contact: +966 50 123 4567")[0]

    reply("confirm all")
    reply("edit 1: contact: ahmad@tact.sa")  # contacts can still be set after confirmation
    task1 = mod.store.tasks_for(meeting_id)[0]
    assert (task1["contact"], task1["status"]) == ("ahmad@tact.sa", "confirmed")
    assert mod.store.get_meeting(meeting_id, "telegram", CHAT)["status"] == "confirmed"


def _invite(mod, gw):
    gw.adapter._current_bot_username = lambda: "tactbot"
    gw.adapter.sent.clear()
    assert _dispatch(mod, gw, _event(text="/invite", message_type="TEXT")) is None
    (reply,) = gw.adapter.sent
    m = re.search(r"https://t\.me/tactbot\?start=join_([A-Za-z0-9_-]+)", reply)
    assert m and "30" in reply
    return m.group(1)


def _join(mod, adapter, token, user=555, username="fahad", name="Fahad K"):
    message = SimpleNamespace(chat=SimpleNamespace(id=user), text=f"/start join_{token}",
                              from_user=SimpleNamespace(id=user, username=username, full_name=name))
    asyncio.run(mod.on_join_message(adapter, message))


def test_invite_registration_needs_the_managers_approval(recorder, telegram_stub):
    mod = recorder
    meeting_id = _seed(mod)
    mod.store.set_task_status(meeting_id, [1], "confirmed")  # Ahmad, a confirmed owner
    mod.contacts.save_contact("Omar", "omar@tact.sa")  # the contacts book comes first
    gw = FakeGateway(bot=True)
    bot = gw.adapter._bot
    token = _invite(mod, gw)
    assert len(token) >= 22  # >= 16 random bytes
    with mod.store.connect() as con:  # only a hash is stored
        assert token not in json.dumps([list(r) for r in con.execute("SELECT * FROM invites").fetchall()])

    _join(mod, gw.adapter, token)
    assert bot.sent_to(555) == ["⏳ طلبك بانتظار موافقة المدير."]
    (request_text,) = bot.sent_to(CHAT)
    assert request_text == "📥 Fahad K (@fahad, 555) يريد استقبال المهام. اربطه بـ:"
    request_id = max(bot.messages)
    labels = [b for row in _buttons(bot.messages[request_id][1]) for b, _ in row]
    assert labels == ["Omar", "Ahmad", "➕ اسم آخر", "❌ رفض"]
    assert mod.contacts.by_chat_id("555") is None  # nothing registered before the manager decides
    # Never matched by display name, and nobody but the manager can decide.
    assert _press(mod, gw.adapter, "rec:j:a:1:0", request_id, user="555") == ["Not allowed."]
    _press(mod, gw.adapter, "rec:j:a:1:0", request_id)
    assert mod.contacts.find("Omar")["telegram_chat_id"] == "555"
    assert bot.sent_to(555)[-1] == "✅ تم تسجيلك لاستقبال المهام."
    assert bot.messages[request_id] == ("✅ تم ربط Fahad K (@fahad, 555) بـ Omar.", None)

    # "➕ اسم آخر": the manager types the name.
    _join(mod, gw.adapter, token, user=556, username="", name="Sara")
    other = max(bot.messages)
    _press(mod, gw.adapter, "rec:j:o:2", other)
    assert _dispatch(mod, gw, _event(text="سارة", message_type="TEXT")) is None
    assert mod.contacts.find("سارة")["telegram_chat_id"] == "556"
    assert "سارة — ✅ تيليجرام" in mod._cmd_contacts("")
    assert mod._cmd_contacts("delete سارة") == "🗑️ حُذف سارة." and mod.contacts.find("سارة") is None


def test_rejected_expired_revoked_and_rate_limited_joins(recorder, telegram_stub, monkeypatch):
    mod = recorder
    gw = FakeGateway(bot=True)
    bot = gw.adapter._bot
    token = _invite(mod, gw)
    _join(mod, gw.adapter, token, user=600)
    _press(mod, gw.adapter, "rec:j:r:1", max(bot.messages))
    assert bot.sent_to(600)[-1] == "❌ لم تتم الموافقة على طلبك." and mod.contacts.by_chat_id("600") is None

    invalid = "⛔ الرابط منتهي أو غير صالح. اطلب رابطاً جديداً من المدير."
    _join(mod, gw.adapter, "x" * 24, user=601)
    assert bot.sent_to(601) == [invalid]
    real_now = mod.invite._now
    monkeypatch.setattr(mod.invite, "_now", lambda: real_now() + 31 * 60)  # 30 minutes have passed
    _join(mod, gw.adapter, token, user=602)
    assert bot.sent_to(602) == [invalid]
    monkeypatch.setattr(mod.invite, "_now", real_now)
    fresh = _invite(mod, gw)
    assert _dispatch(mod, gw, _event(text="/invite revoke", message_type="TEXT")) is None
    _join(mod, gw.adapter, fresh, user=603)
    assert bot.sent_to(603) == [invalid] and not bot.sent_to(CHAT)[1:]  # the manager heard only of #1

    for _ in range(7):  # at most 5 join attempts per user per hour get any answer
        _join(mod, gw.adapter, "y" * 24, user=604)
    assert bot.sent_to(604) == [invalid] * 5


def test_a_registered_member_can_use_nothing_else(recorder, telegram_stub, tmp_path):
    mod = recorder
    meeting_id = _seed(mod)
    mod.store.set_task_status(meeting_id, [1], "confirmed")
    gw = FakeGateway(bot=True)
    token = _invite(mod, gw)
    _join(mod, gw.adapter, token, user=555)
    _press(mod, gw.adapter, "rec:j:a:1:0", max(gw.adapter._bot.messages))
    assert mod.contacts.by_chat_id("555")

    member = FakeGateway(authorized=False, bot=True)  # the gateway's allowlist still refuses them
    for text, mtype in (("hello", "TEXT"), ("/recordings", "COMMAND"), ("/brief 1", "COMMAND"),
                        ("/invite", "COMMAND"), ("/start", "COMMAND"), ("confirm all", "TEXT")):
        event = _event(text=text, message_type=mtype, user="555")
        assert _dispatch(mod, member, event) is event, text  # untouched: the gateway drops it as before
    voice = _event(message_type="AUDIO", media=[_wav(tmp_path / "m.wav", 1)], user="555",
                   raw=SimpleNamespace(audio=SimpleNamespace(file_size=32000)))
    assert _dispatch(mod, member, voice) is voice
    assert not member.adapter.sent and not member.adapter._bot.messages
    for text in ("/start", "/start hello", "hello join_" + token, "/start join_x"):
        assert not mod.invite.JOIN_RE.match(text), text  # only the exact join message is handled early
    assert _press(mod, gw.adapter, f"rec:c:{meeting_id}:1", 1, user="555") == ["Not allowed."]
    assert _press(mod, gw.adapter, f"rec:x:s:{meeting_id}", 1, user="555") == ["Not allowed."]


class FakeSMTP:
    sent, logins = [], []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.tls = host, port, False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context=None):
        self.tls = True

    def login(self, user, password):
        FakeSMTP.logins.append((user, password, self.tls))

    def send_message(self, msg):
        FakeSMTP.sent.append(msg)


def _sending_meeting(mod):
    language, brief, _ = mod.brief.normalize(ANALYSIS)
    meeting_id = mod.store.create_meeting("telegram", CHAT, MANAGER, manager_name="أبو فيصل")
    tasks = [{"person": p, "person_key": mod.brief.name_key(p), "candidates": c, "task": t, "deadline": d, "contact": k}
             for p, c, t, d, k in (
                 ("Omar", [], "Send the deck", "Tuesday", ""),
                 ("Sara", [], "Review the contract", "", ""),
                 ("", ["Omar", "Sara"], "Call the vendor", "", ""),
                 ("Fahad", [], "Book the venue", "", ""),
                 ("Omar، Sara", [], "Prepare the launch plan", "Sunday", ""),
                 ("Omar", [], "Cancelled idea", "", ""),
                 ("Khalid", [], "Update the website", "", "@khalid_k"))]
    mod.store.save_analysis(meeting_id, "en", brief, tasks)
    mod.store.set_task_status(meeting_id, [1, 2, 3, 4, 5, 7], "confirmed")
    mod.store.set_task_status(meeting_id, [6], "removed")
    mod.store.update_meeting(meeting_id, status="confirmed")
    mod.contacts.link_telegram("Omar", "777", "omar_t")
    mod.contacts.save_contact("Sara", "sara@tact.sa")
    return meeting_id


def test_tasks_are_sent_only_after_the_managers_approval(recorder, telegram_stub, monkeypatch, caplog):
    mod = recorder
    caplog.set_level(logging.INFO)
    for var in mod.sending.SMTP_VARS:
        monkeypatch.delenv(var, raising=False)
    meeting_id = _sending_meeting(mod)
    gw = FakeGateway(bot=True)
    bot = gw.adapter._bot
    assert _dispatch(mod, gw, _event(text=f"/brief {meeting_id}", message_type="TEXT")) is None
    ask = max(bot.messages)
    assert _buttons(bot.messages[ask][1]) == [[("📤 Send tasks", f"rec:x:p:{meeting_id}")]]

    _press(mod, gw.adapter, f"rec:x:p:{meeting_id}", ask)  # the preview sends nothing
    preview = max(bot.messages)
    text, markup = bot.messages[preview]
    assert "👤 Omar — Telegram\n  • 1. Send the deck\n  • 5. Prepare the launch plan" in text
    for line in ("2. Sara — email sending not configured", "3. ❓ UNCLEAR: Omar or Sara — owner unclear",
                 "4. Fahad — contact unknown", "5. Sara — email sending not configured",
                 "7. Khalid — ⏳ not registered in the bot yet"):
        assert line in text, line
    assert "Cancelled idea" not in text
    assert _buttons(markup) == [[("✅ Send", f"rec:x:s:{meeting_id}"), ("❌ Cancel", f"rec:x:c:{meeting_id}")]]
    assert bot.sent_to("777") == []  # nothing without ✅

    _press(mod, gw.adapter, f"rec:x:c:{meeting_id}", preview)
    assert bot.messages[preview] == ("Nothing was sent.", None) and bot.sent_to("777") == []

    _press(mod, gw.adapter, f"rec:x:p:{meeting_id}", ask)
    _press(mod, gw.adapter, f"rec:x:s:{meeting_id}", max(bot.messages))
    assert bot.sent_to("777") == [f"📋 Your tasks from the meeting on {mod.store.get_meeting(meeting_id, 'telegram', CHAT)['created_at'][:10]}"
                                  " with أبو فيصل:\n1. Send the deck — Deadline: Tuesday\n"
                                  "2. Prepare the launch plan — Deadline: Sunday"]
    tasks = mod.store.tasks_for(meeting_id)
    assert [t["sent_via"] for t in tasks] == ["telegram", None, None, None, "telegram", None, None]
    assert gw.adapter.sent[-1].startswith(f"📤 Sending result — Meeting {meeting_id}\n✅ Omar — Telegram: 2 task(s)")

    # Email once SMTP is configured; the joint task now also reaches Sara. Omar's tasks were sent
    # before, so the preview says they would go again.
    for var, value in zip(mod.sending.SMTP_VARS, ("smtp.gmail.com", "587", "bot@tact.sa", "app-secret-pw", "bot@tact.sa")):
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(mod.sending.smtplib, "SMTP", FakeSMTP)
    FakeSMTP.sent.clear()
    _press(mod, gw.adapter, f"rec:x:p:{meeting_id}", ask)
    assert "👤 Sara — email (sara@tact.sa)\n  • 2. Review the contract\n  • 5. Prepare the launch plan" in \
        bot.messages[max(bot.messages)][0]
    assert "were sent before and will be sent again" in bot.messages[max(bot.messages)][0]
    _press(mod, gw.adapter, f"rec:x:s:{meeting_id}", max(bot.messages))
    (email,) = FakeSMTP.sent
    assert email["To"] == "sara@tact.sa" and "Review the contract" in email.get_content()
    assert FakeSMTP.logins == [("bot@tact.sa", "app-secret-pw", True)]  # STARTTLS before login
    assert mod.store.tasks_for(meeting_id)[4]["sent_via"] == "telegram,email"
    assert "app-secret-pw" not in caplog.text and "Send the deck" not in caplog.text


def test_name_pickers_offer_only_real_confirmed_people(recorder):
    mod = recorder
    meeting_id = mod.store.create_meeting("telegram", CHAT, MANAGER)
    rows = [("Omar", []), ("احمد او عمر (يحدد لاحقا)", []), ("فهد او نورة", []), ("Sara / Omar", []),
            ("Khalid", []), ("Nasser", []), ("", ["Ali", "Hassan"]), ("عمر، Layla", []), ("Abdul Rahman Al Saud", []),
            ("سره", [])]
    mod.store.save_analysis(meeting_id, "ar", {"summary": []}, [
        {"person": p, "person_key": mod.brief.name_key(p), "candidates": c, "task": f"t{i}", "deadline": ""}
        for i, (p, c) in enumerate(rows, 1)])
    mod.store.set_task_status(meeting_id, [1, 2, 3, 4, 7, 8, 9], "confirmed")  # Khalid pending, Nasser, سره
    mod.store.set_task_status(meeting_id, [6, 10], "removed")                   # cancelled
    mod.contacts.save_contact("فهد", "fahad@tact.sa")
    picker = mod.contacts.picker_names  # contacts book first, then confirmed owners; one per person
    assert picker("telegram", CHAT) == ["فهد", "Omar", "عمر", "Layla"]
    assert mod.contacts.is_person_name("عبد الله") and not mod.contacts.is_person_name("Omar or Sara")

    # /contacts delete also hides a name that only lives in past tasks; rename fixes it everywhere.
    assert mod._cmd_contacts("delete Layla") == "🗑️ حُذف Layla."
    assert mod._cmd_contacts("delete Nobody") == "لا توجد جهة اتصال باسم Nobody."
    assert mod._cmd_contacts("rename Omar -> عمر") == "✏️ أُعيدت تسمية Omar إلى عمر (في 1 مهمة)."
    assert picker("telegram", CHAT) == ["فهد", "عمر"]
    # Only real owner names are renamed; a placeholder like "Sara / Omar" is left as it was.
    assert [t["person"] for t in mod.store.tasks_for(meeting_id)][:4] == ["عمر", "احمد او عمر (يحدد لاحقا)",
                                                                         "فهد او نورة", "Sara / Omar"]
    assert mod._cmd_contacts("rename فهد → Fahad").startswith("✏️ أُعيدت تسمية فهد إلى Fahad")
    assert mod.contacts.find("Fahad")["email"] == "fahad@tact.sa" and picker("telegram", CHAT)[0] == "Fahad"
    assert mod._cmd_contacts("rename Fahad -> x / y") == "«x / y» ليس اسماً صالحاً."


def test_recordings_delete_and_clear_need_confirmation(recorder, telegram_stub, monkeypatch):
    mod = recorder
    first, second, third = _seed(mod), _seed(mod), _seed(mod)
    for number in (first, second, third):
        (mod.store.meeting_dir(number) / "transcript.txt").write_text("secret words", encoding="utf-8")
        mod.store.update_meeting(number, transcript_path=str(mod.store.meeting_dir(number) / "transcript.txt"))
    mod.contacts.save_contact("Omar", "omar@tact.sa")
    gw = FakeGateway(bot=True)
    bot = gw.adapter._bot
    folder = Path(mod.store.recordings_dir())

    assert _dispatch(mod, gw, _event(text=f"/recordings delete {first}", message_type="TEXT")) is None
    question = max(bot.messages)
    assert bot.messages[question][0] == f"🗑️ حذف الاجتماع #{first} ومهامه ونصّه نهائياً؟"
    assert mod.store.get_meeting(first, "telegram", CHAT) is not None  # nothing deleted before ✅
    assert _press(mod, gw.adapter, f"rec:d:y:{first}", question, user="99") == ["Not allowed."]
    _press(mod, gw.adapter, f"rec:d:n:{first}", question)
    assert bot.messages[question] == ("لم يُحذف شيء.", None) and (folder / str(first)).exists()

    assert _dispatch(mod, gw, _event(text=f"/recordings delete {first}", message_type="TEXT")) is None
    _press(mod, gw.adapter, f"rec:d:y:{first}", max(bot.messages))
    assert mod.store.get_meeting(first, "telegram", CHAT) is None and mod.store.tasks_for(first) == []
    assert not (folder / str(first)).exists()
    assert mod.store.get_meeting(second, "telegram", CHAT) is not None

    # Without buttons: the command asks for an explicit "confirm".
    monkeypatch.setenv("HERMES_SESSION_USER_ID", MANAGER)
    assert mod._cmd_recordings(f"delete {second}") == f"للتأكيد أرسل: /recordings delete {second} confirm"
    assert mod._cmd_recordings(f"delete {second} confirm") == f"🗑️ حُذف الاجتماع #{second}."
    assert not (folder / str(second)).exists()

    assert _dispatch(mod, gw, _event(text="/recordings clear", message_type="TEXT")) is None
    assert "كل الاجتماعات المسجّلة (1)" in bot.messages[max(bot.messages)][0]
    _press(mod, gw.adapter, "rec:d:y:0", max(bot.messages))
    assert mod.store.recent_meetings("telegram", CHAT) == [] and not (folder / str(third)).exists()
    assert mod.contacts.find("Omar")["email"] == "omar@tact.sa"  # the contacts book is kept
