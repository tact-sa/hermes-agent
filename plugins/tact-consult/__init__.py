"""tact-consult — an executive strategy consultant that runs ONLY on ``/consult <question>``.

Skills have no "user-invocable only" flag: every installed skill is listed in the model's skill
index and loadable through ``skill_view``, so a ``consult`` skill could be applied to ordinary
messages. The consultant instructions therefore live in this plugin (``consult.md``, outside every
skills directory) where no model tool can reach them, and enter a turn only through this path:

- ``pre_gateway_dispatch`` rewrites an incoming ``/consult <question>`` into the instructions plus
  the question, using the same scaffold markers as a skill slash command, so memory providers store
  just the question and transcripts render it as ``/consult — <question>``. Auth, pairing and the
  normal agent turn then run on the rewritten message exactly as for any other message.
- The registered ``/consult`` command puts it in the platform command menus and, since a question
  is always rewritten first, only ever answers a bare ``/consult`` with the usage hint.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

INSTRUCTIONS_PATH = Path(__file__).with_name("consult.md")
# "/consult", "/consult@TactBot" (Telegram groups), then the question on the same or later lines.
_CONSULT_RE = re.compile(r"^\s*/consult(?:@\S+)?(?:\s+(?P<question>.*))?$", re.IGNORECASE | re.DOTALL)

USAGE = ("Usage: /consult <question>, e.g. /consult Should we open a Riyadh office this year "
         "or keep serving clients remotely from Jeddah?")
# Byte-identical to agent.skill_commands.build_skill_invocation_message's activation note, so
# extract_user_instruction_from_skill_message / describe_skill_invocation recognise the turn.
ACTIVATION_NOTE = ('[IMPORTANT: The user has invoked the "consult" skill, indicating they want '
                   "you to follow its instructions. The full skill content is loaded below.]")


def parse_question(text: Any) -> Optional[str]:
    """The question after ``/consult``: ``None`` when *text* is not a /consult command, ``""`` when
    it is a bare ``/consult``."""
    if not isinstance(text, str):
        return None
    match = _CONSULT_RE.match(text)
    if match is None:
        return None
    return (match.group("question") or "").strip()


def build_consult_message(question: str) -> str:
    """The model-facing turn: activation note, consultant instructions, then the question."""
    from agent.prompt_cache_boundary import register_stable_prefix
    from agent.skill_commands import append_user_instruction

    parts = [ACTIVATION_NOTE, "", INSTRUCTIONS_PATH.read_text(encoding="utf-8").strip(), ""]
    stable_prefix = append_user_instruction(parts, question)
    message = "\n".join(parts)
    register_stable_prefix(stable_prefix)
    return message


def _on_pre_gateway_dispatch(event: Any = None, **_kw) -> Optional[Dict[str, Any]]:
    question = parse_question(getattr(event, "text", None))
    if not question:  # not /consult, or bare /consult (the command handler replies with USAGE)
        return None
    try:
        return {"action": "rewrite", "text": build_consult_message(question)}
    except Exception:
        logger.exception("tact-consult: could not build the /consult turn")
        return None


def _cmd_consult(raw_args: str) -> str:
    # A /consult with a question is rewritten by the hook before command dispatch, so reaching
    # here with one means the rewrite failed (logged above).
    if not (raw_args or "").strip():
        return USAGE
    return "⚠️ /consult is unavailable right now. Please try again in a moment."


def register(ctx) -> None:
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
    ctx.register_command("consult", _cmd_consult, args_hint="<question>",
                         description="Executive strategy consultant for a business decision")
