"""tact-guard — human approval before the browser submits anything.

``post_tool_call`` remembers, per task/session, the ``ref -> (role, label)`` map from the latest
browser snapshot plus the current page URL. ``pre_tool_call`` then:

- escalates ``browser_click`` to the human approval gate when the ref is unknown, or when a
  submit-capable role (anything outside ``SAFE_ROLES``) has a submit-type label (English/Arabic)
  or no label at all. Links only navigate, so a link asks only for non-booking submit words
  (pay, checkout, confirm, delete, cancel, buy, send, ...), never for "Book now"/"احجز";
- escalates ``browser_press`` Enter (it may submit a focused form);
- blocks ``browser_type`` into password / OTP / card / national-ID fields so the manager fills
  them personally;
- blocks in-page script execution (``browser_console``, ``browser_cdp``, ``browser_exec``) and the
  saved-password vault (``browser_vault_*``).

Every approval carries a fresh nonce in its ``rule_key`` so an "always"/"session" answer never
turns into a standing allowlist entry. The one exception is a retry: once the manager approves a
click and it actually runs, the same label on the same domain in the same session is allowed
again for ``APPROVAL_TTL_SECONDS`` (a click that did not change the page is usually retried).
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

_SUBMIT_EN = (
    "submit", "book", "booking", "reserve", "confirm", "pay", "checkout", "place order", "order",
    "buy", "purchase", "send", "register", "sign up", "apply", "schedule", "request", "delete",
    "remove", "cancel", "unsubscribe",
)
_SUBMIT_AR = (
    "إرسال", "ارسال", "حجز", "احجز", "تأكيد", "أكد", "تسجيل", "سجل", "دفع", "ادفع", "شراء", "طلب",
    "اطلب", "حذف", "إلغاء", "الغاء",
)
# Booking words are fine on a link: it only opens the booking page, and the final submit is a button.
_LINK_OK_EN = frozenset({"book", "booking", "reserve", "schedule", "apply", "register", "request"})
_LINK_OK_AR = frozenset({"حجز", "احجز"})
_SENSITIVE_EN = ("password", "passcode", "otp", "card", "cvv", "cvc", "iban", "iqama", "national id")
_SENSITIVE_AR = ("هوية", "إقامة", "كلمة المرور", "رمز التحقق", "بطاقة")


def _word_regex(words: Tuple[str, ...]) -> "re.Pattern[str]":
    # English words match on word boundaries ("Facebook" is not "book"); spaces inside a phrase
    # accept any run of whitespace, hyphen or underscore ("sign-up", "place  order").
    alts = sorted((re.escape(w).replace(r"\ ", r"[\s_-]+") for w in words), key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(alts) + r")(?![a-z0-9])", re.IGNORECASE)


_SUBMIT_EN_RE = _word_regex(_SUBMIT_EN)
_LINK_ASK_EN_RE = _word_regex(tuple(w for w in _SUBMIT_EN if w not in _LINK_OK_EN))
_LINK_ASK_AR = tuple(w for w in _SUBMIT_AR if w not in _LINK_OK_AR)
_SENSITIVE_EN_RE = _word_regex(_SENSITIVE_EN)


def _matches(label: str, en_re: "re.Pattern[str]", ar_words: Tuple[str, ...]) -> bool:
    # Arabic attaches prefixes/suffixes to the word itself ("والتأكيد"), so substring matching.
    return bool(en_re.search(label)) or any(w in label for w in ar_words)


def is_submit_label(label: str) -> bool:
    return _matches(label, _SUBMIT_EN_RE, _SUBMIT_AR)


def is_link_submit_label(label: str) -> bool:
    return _matches(label, _LINK_ASK_EN_RE, _LINK_ASK_AR)


def is_sensitive_label(label: str) -> bool:
    return _matches(label, _SENSITIVE_EN_RE, _SENSITIVE_AR)


# `- button "Book now" [ref=e12]`, `  - textbox [ref=e3]`, and agent-browser 0.26's shared
# attribute bracket: `- heading "Web form" [level=1, ref=e1]`, `- option "One" [selected, ref=e9]`
_SNAPSHOT_LINE_RE = re.compile(
    r'^\s*-\s*(?P<role>[A-Za-z][\w-]*)(?:\s+"(?P<label>(?:[^"\\]|\\.)*)")?[^\n]*?\[[^\]\n]*?\bref=(?P<ref>[^\],\s]+)[^\]\n]*\]',
    re.MULTILINE,
)

# Roles whose click selects, focuses or toggles but cannot submit a form.
SAFE_ROLES = frozenset({
    "checkbox", "radio", "option", "combobox", "listbox", "textbox", "searchbox", "spinbutton",
    "slider", "switch", "tab", "treeitem", "gridcell", "heading", "img", "StaticText",
})

_MAX_TRACKED = 256
APPROVAL_TTL_SECONDS = 120.0
_lock = threading.Lock()
# state key -> {"refs": {ref: (role, label)}, "url": str}
_pages: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
# pending key -> (state key, domain, label) for a click sent to the approval gate
_pending: "OrderedDict[str, Tuple[str, str, str]]" = OrderedDict()
# (state key, domain, label) -> monotonic time the approved click ran
_approved: "OrderedDict[Tuple[str, str, str], float]" = OrderedDict()


def _now() -> float:
    return time.monotonic()


def _pending_key(state_key: str, tool_call_id: Any, ref: str) -> str:
    return f"{state_key}\x00{tool_call_id or '@' + ref}"


def _trim(d: "OrderedDict[Any, Any]") -> None:
    while len(d) > _MAX_TRACKED:
        d.popitem(last=False)


def _recently_approved(approval: Tuple[str, str, str]) -> bool:
    with _lock:
        at = _approved.get(approval)
        if at is None:
            return False
        if _now() - at <= APPROVAL_TTL_SECONDS:
            return True
        del _approved[approval]
        return False


def _record_click_outcome(state_key: str, tool_call_id: Any, args: Any, status: Any) -> None:
    """A gated click that reached the tool (status not "blocked") was approved by the manager."""
    ref = _norm_ref(args.get("ref")) if isinstance(args, dict) else ""
    with _lock:
        approval = _pending.pop(_pending_key(state_key, tool_call_id, ref), None)
        if approval is not None and status != "blocked":
            _approved.pop(approval, None)
            _approved[approval] = _now()
            _trim(_approved)


def _state_key(task_id: Any, session_id: Any) -> str:
    return str(task_id or session_id or "default")


def _norm_ref(ref: Any) -> str:
    return str(ref or "").strip().lstrip("@").strip()


def parse_snapshot(text: str) -> Dict[str, Tuple[str, str]]:
    refs: Dict[str, Tuple[str, str]] = {}
    for m in _SNAPSHOT_LINE_RE.finditer(text or ""):
        label = (m.group("label") or "").replace('\\"', '"').strip()
        refs[_norm_ref(m.group("ref"))] = (m.group("role"), label)
    return refs


def _result_parts(result: Any) -> Tuple[str, str]:
    """(snapshot text, page url) from a browser tool result (JSON string, dict, or raw text)."""
    data: Any = result
    if isinstance(result, str):
        try:
            data = json.loads(result)
        except ValueError:
            return result, ""
    if not isinstance(data, dict):
        return (result if isinstance(result, str) else ""), ""
    url = data.get("url") if isinstance(data.get("url"), str) else ""
    snap = data.get("snapshot")
    return (snap if isinstance(snap, str) else ""), url


def _on_post_tool_call(tool_name: str = "", args: Optional[dict] = None, result: Any = None,
                       task_id: str = "", session_id: str = "", tool_call_id: str = "",
                       status: str = "", **_: Any) -> None:
    if not str(tool_name).startswith("browser_"):
        return
    if tool_name == "browser_click":
        _record_click_outcome(_state_key(task_id, session_id), tool_call_id, args, status)
    snapshot, url = _result_parts(result)
    if not url and tool_name == "browser_navigate" and isinstance(args, dict):
        url = str(args.get("url") or "")
    refs = parse_snapshot(snapshot)
    if not refs and not url:
        return
    key = _state_key(task_id, session_id)
    with _lock:
        page = _pages.pop(key, None) or {"refs": {}, "url": ""}
        if refs:  # a new snapshot renumbers every ref, so it replaces the old map
            page["refs"] = refs
        if url:
            page["url"] = url
        _pages[key] = page
        _trim(_pages)


def _page(task_id: Any, session_id: Any) -> Dict[str, Any]:
    with _lock:
        page = _pages.get(_state_key(task_id, session_id))
        return {"refs": dict(page["refs"]), "url": page["url"]} if page else {"refs": {}, "url": ""}


def _domain(url: str) -> str:
    host = urlparse(url).hostname if url else None
    return host or "the current page"


def _approve(kind: str, message: str) -> Dict[str, str]:
    return {"action": "approve", "message": message, "rule_key": f"tact-guard:{kind}:{uuid.uuid4().hex}"}


_SCRIPT_TOOLS = frozenset({"browser_console", "browser_cdp", "browser_exec"})
_VAULT_TOOLS = frozenset({
    "browser_vault_list", "browser_vault_unlock", "browser_vault_fill", "browser_vault_save_login",
    "browser_vault_enter_code",
})


def _on_pre_tool_call(tool_name: str = "", args: Optional[dict] = None, task_id: str = "",
                      session_id: str = "", tool_call_id: str = "", **_: Any) -> Optional[Dict[str, str]]:
    if tool_name in _SCRIPT_TOOLS:
        return {"action": "block", "message": "Running scripts in pages is disabled for safety."}
    if tool_name in _VAULT_TOOLS:
        return {"action": "block", "message": "Saved passwords are disabled. Ask the manager to log in themselves."}
    if tool_name not in ("browser_click", "browser_press", "browser_type"):
        return None
    args = args if isinstance(args, dict) else {}
    page = _page(task_id, session_id)
    domain = _domain(page["url"])

    if tool_name == "browser_press":
        key = str(args.get("key") or "").strip().split("+")[-1].lower()
        if key in ("enter", "return"):
            return _approve("enter", f"Press Enter on {domain}, may submit a form")
        return None

    ref = _norm_ref(args.get("ref"))
    role, label = page["refs"].get(ref, ("", ""))

    if tool_name == "browser_type":
        if label and is_sensitive_label(label):
            return {"action": "block", "message": (
                f'The field "{label}" on {domain} holds sensitive data (password, verification code, '
                "card or ID number). Do not type it. Ask the manager to fill that field themselves.")}
        return None

    # browser_click. An unknown ref could be anything, so it asks. Known select/toggle/field roles
    # cannot submit. Anything else asks on a submit word, or when unlabeled (except a plain link).
    if not role:
        shown = f"@{ref}" if ref else "an unknown element"
        return _approve("click", f'Click "{shown}" on {domain}')
    if role in SAFE_ROLES:
        return None
    if not label:
        return None if role == "link" else _approve("click", f'Click "unlabeled {role} @{ref}" on {domain}')
    if not (is_link_submit_label(label) if role == "link" else is_submit_label(label)):
        return None
    key = _state_key(task_id, session_id)
    approval = (key, domain, label)
    if _recently_approved(approval):
        return None
    with _lock:
        _pending[_pending_key(key, tool_call_id, ref)] = approval
        _trim(_pending)
    return _approve("click", f'Click "{label}" on {domain}')


def register(ctx) -> None:
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
