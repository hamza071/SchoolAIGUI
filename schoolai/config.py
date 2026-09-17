"""Central configuration for the SchoolAI local coding harness.

Everything an operator may need to change lives here, plus a tiny settings
helper so the GUI can persist a Bearer token without editing source.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# NebulaONE endpoint
# ---------------------------------------------------------------------------

API_URL = (
    "https://ai-chat.hhs.nl/api/internal/userConversations/"
    "byGptSystemId/2e6987f9-be64-4fbc-83e5-53c147935e4b"
)

# >>> Hardcode a token here if you prefer, or set NEBULA_BEARER_TOKEN, or use
# the GUI's token field (which saves to ~/.schoolai/config.json). <<<
# Leave it as-is to run without a hardcoded token; the "Bearer " scheme is
# added automatically, so a bare JWT works too.
BEARER_TOKEN = "YOUR_BEARER_TOKEN"

# Values that mean "no token configured yet".
_UNSET_TOKENS = frozenset({"", "YOUR_BEARER_TOKEN"})

# Where the GUI stores the token the user pastes in.
SETTINGS_PATH = Path(os.environ.get("SCHOOLAI_CONFIG", "~/.schoolai/config.json")).expanduser()

ENV_TOKEN_VAR = "NEBULA_BEARER_TOKEN"


def load_settings() -> Dict[str, Any]:
    """Read the small JSON settings file. Corrupt/missing file -> {}."""
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(settings: Dict[str, Any]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    payload = json.dumps(settings, indent=2)
    # Create the temp file already at 0600 and write through the descriptor, so
    # the credential is never world-readable, not even briefly. os.fdopen takes
    # ownership of the descriptor and closes it on every path, including when
    # the write fails -- so there is deliberately no manual close here.
    descriptor = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(payload)
    os.replace(tmp, SETTINGS_PATH)


def normalize_bearer_token(raw: str) -> str:
    """Ensure the Authorization value carries the ``Bearer `` scheme.

    The endpoint expects ``authorization: Bearer <jwt>``. Pasting a bare JWT
    (or one copied without the prefix) otherwise fails with a confusing 401,
    so the scheme is added when it is missing.
    """
    token = (raw or "").strip()
    if not token:
        return ""

    # If the value already carries the Bearer scheme, return it as-is.
    if token.lower().startswith("bearer "):
        return token

    # Heuristic: users sometimes paste surrounding text after the JWT (for
    # example: "<jwt> gebruik de verbeterde code ..."). Try to extract a
    # typical JWT-looking substring (starts with "eyJ" and has two dots).
    m = re.search(r"(eyJ[0-9A-Za-z_\-]+\.[0-9A-Za-z_\-]+\.[0-9A-Za-z_\-]+)", token)
    if m:
        return f"Bearer {m.group(1)}"

    return f"Bearer {token}"


def resolve_bearer_token() -> str:
    """env var  >  saved settings  >  hardcoded BEARER_TOKEN  >  '' (unset)."""
    candidates = (
        os.environ.get(ENV_TOKEN_VAR, ""),
        str(load_settings().get("bearer_token", "")),
        BEARER_TOKEN,
    )
    for candidate in candidates:
        cleaned = candidate.strip()
        if cleaned and cleaned not in _UNSET_TOKENS:
            return normalize_bearer_token(cleaned)
    return ""


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------

DEPLOYMENT_IDENTIFIER = "Foundry|gpt-5.4-mini"
CAPABILITIES: List[str] = ["ImageCreation", "InternetSearch", "CodeInterpreter"]
REASONING_MODE = "Balanced"
TIMEZONE = "Europe/Amsterdam"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"
)

# (connect timeout, read timeout) in seconds.
REQUEST_TIMEOUT = (10, 300)

# Safety valve so a confused model cannot loop read/write forever.
MAX_TOOL_ROUNDS = 25

# ...nor request thousands of tool calls inside a single round.
MAX_TOOL_CALLS_PER_ROUND = 32

# The endpoint keeps NO server-side conversation memory (verified against the
# live service: two POSTs sharing a sessionIdentifier are unrelated), so the
# transcript is resent on every round. This caps how large it may grow before
# the oldest middle turns are dropped; the original task is always kept.
HISTORY_MAX_CHARS = 120_000

# Appended after the transcript so the model knows it is the assistant's turn.
CONTINUE_CUE = (
    "Continue as the Assistant. If you need a file operation, reply with "
    "exactly one tool tag and nothing else; otherwise give your final answer "
    "to the user."
)

# ---------------------------------------------------------------------------
# SSE contract
#
# The upstream emits base64-encoded text chunks on `response-updated`. These
# sets are configurable because the exact event vocabulary is undocumented;
# run `python run.py --debug-sse` to dump raw frames and extend them if the
# stream clearly carries text on another event name.
# ---------------------------------------------------------------------------

STREAM_TEXT_EVENTS = {"response-updated"}
# Captured from the live endpoint: the terminal event is `no-more-data`.
STREAM_COMPLETION_EVENTS = {"no-more-data", "response-completed", "message-completed"}
STREAM_ERROR_EVENTS = {"error", "exception"}


def new_session_id() -> str:
    """Fresh UUID used as `session.sessionIdentifier` for a conversation."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# System prompt injected into the hidden prompt context on every request.
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """You are a coding assistant. You have access to the user's workspace. To read a file, output EXACTLY <READ_FILE>filename.ext</READ_FILE>. To write/overwrite a file, output EXACTLY <WRITE_FILE name="filename.ext">code</WRITE_FILE>.

Rules you must follow:
- Paths are relative to the workspace root. Never use absolute paths, "..", "~" or a leading "/".
- Emit ONE tool tag per step and wait for its result before emitting the next one.
- Put the complete file contents between <WRITE_FILE name="..."> and </WRITE_FILE>, with no markdown code fences and no commentary inside the tag.
- After you receive a READ_FILE result the file contents arrive as your input; continue the task.
- When the task is finished, reply with a short plain-text summary of what you changed. Do not invent tool results."""
