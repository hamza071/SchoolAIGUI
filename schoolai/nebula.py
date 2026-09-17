"""NebulaONE transport: SSE + base64 stream decoding.

The endpoint is undocumented, so the decoder is deliberately tolerant: it
accepts a base64-encoded chunk, a base64-encoded JSON envelope, a raw JSON
envelope, or plain text, and extracts the text from whichever shape it finds.

Run ``python run.py --debug-sse`` to dump every raw frame to
``schoolai_sse.log`` if the real event vocabulary turns out to differ.
"""
from __future__ import annotations

import base64
import binascii
import json
import threading
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import requests

from . import config


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class NebulaError(Exception):
    """Any failure talking to the NebulaONE endpoint."""


class AuthError(NebulaError):
    """401/403 — the Bearer token is missing, expired or wrong."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


class StreamInterrupted(NebulaError):
    """The connection dropped mid-answer."""


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


def build_payload(question: str, session_id: str) -> Dict[str, object]:
    return {
        "question": question,
        "visionImageIds": [],
        "attachmentIds": [],
        "session": {"sessionIdentifier": session_id},
        "segmentTraceLogLevel": "NonPersisted",
        "answerGenerationOptions": {
            "deploymentIdentifier": config.DEPLOYMENT_IDENTIFIER,
            "capabilities": list(config.CAPABILITIES),
            "capabilitiesExplicitlyProvided": True,
            "reasoningMode": config.REASONING_MODE,
        },
        "isTemporary": False,
    }


def build_headers(token: str) -> Dict[str, str]:
    return {
        "authorization": token,
        "content-type": "application/json",
        "accept": "*/*",
        "user-agent": config.USER_AGENT,
        "origin": "https://ai-chat.hhs.nl",
        "referer": "https://ai-chat.hhs.nl/chat/onechat",
        "x-timezone": config.TIMEZONE,
    }


def check_status(status: int, body: str) -> None:
    """Turn an HTTP error status into a typed exception (or return silently)."""
    if status in (401, 403):
        raise AuthError(
            f"Authentication failed (HTTP {status}). "
            "Your Bearer token is missing, expired or invalid — update it and retry.",
            status=status,
        )
    if status == 429:
        raise NebulaError("Rate limited by NebulaONE (HTTP 429). Wait and retry.")
    if status >= 400:
        snippet = (body or "").strip().replace("\n", " ")[:300]
        raise NebulaError(f"NebulaONE returned HTTP {status}: {snippet}")


# ---------------------------------------------------------------------------
# Chunk decoding
# ---------------------------------------------------------------------------

# Keys that plausibly hold the text of a streamed delta, most specific first.
_TEXT_KEYS = ("text", "delta", "content", "chunk", "value", "token", "answer", "message")


def _extract_text(obj: object) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return "".join(_extract_text(item) for item in obj)
    if isinstance(obj, dict):
        for key in _TEXT_KEYS:
            if key in obj:
                extracted = _extract_text(obj[key])
                if extracted:
                    return extracted
        for key in ("data", "payload", "response", "result"):
            if key in obj:
                extracted = _extract_text(obj[key])
                if extracted:
                    return extracted
        return ""
    return ""


# Control characters that legitimately appear in streamed text. Anything else
# in the C0 range is a strong signal that we decoded binary garbage rather than
# text, which is what makes the base64/plain-text discrimination safe.
_ALLOWED_CONTROL_CHARS = frozenset("\n\r\t")


def _looks_like_text(value: str) -> bool:
    return not any(
        ord(ch) < 32 and ch not in _ALLOWED_CONTROL_CHARS for ch in value
    )


def _b64_decode(data: str) -> Optional[str]:
    """Strict base64 -> utf-8. Returns None when the input is not base64.

    Strictness matters: with ``validate=False`` any punctuation is silently
    discarded and the remainder decodes to junk, so plain text would be
    corrupted ("Go!" -> "\x1a") instead of passed through untouched.
    """
    compact = "".join(data.split())
    if not compact:
        return None
    padded = compact + "=" * (-len(compact) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    # Reject decodes that are clearly not text (random binary).
    if not _looks_like_text(text):
        return None
    return text


def decode_chunk(data: str) -> str:
    """Decode one ``data:`` payload from a text-carrying SSE event."""
    if not data:
        return ""

    # 1. Some deployments send the JSON envelope in the clear.
    obj = _try_json(data)
    if obj is not None:
        return _extract_text(obj)

    # 2. Documented path: base64-encoded text (optionally a JSON envelope).
    decoded = _b64_decode(data)
    if decoded is None:
        # 3. Fall back to treating it as raw text.
        return data

    inner = _try_json(decoded)
    if inner is not None:
        return _extract_text(inner)
    return decoded


def _try_json(value: str) -> Optional[object]:
    stripped = value.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        return json.loads(stripped)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# SSE framing
# ---------------------------------------------------------------------------

# Upper bound on a single SSE frame's accumulated `data:` payload.
MAX_SSE_DATA_CHARS = 4 * 1024 * 1024


class SSEDecoder:
    """Incremental decoder for NebulaONE's SSE dialect.

    This server does not follow the SSE spec: it puts an ``event:`` line and
    its payload in *separate* frames, separated by blank lines. Real capture::

        event: response-updated
        <blank>
        data: VGhpbmtpbmc=
        <blank>

    A spec-compliant parser never sees an event and its data in the same frame,
    so it tags the payload with the default ``message`` and every chunk is
    discarded. The event name is therefore treated as sticky state that applies
    to the ``data:`` lines which follow — which is also how the original working
    script behaved. Standard ``event: X / data: Y / blank`` frames still work,
    because there the data arrives before the frame ends.
    """

    def __init__(self) -> None:
        self._event: Optional[str] = None
        self._data: List[str] = []
        self._size = 0

    def feed_line(self, line: str) -> Optional[Tuple[str, str]]:
        """Consume one line; return a completed ``(event, data)`` frame or None."""
        if line.startswith(":"):
            return None  # comment / keep-alive

        if line == "":
            # A frame only produces output when it carried data. A bare
            # `event:` frame merely updates the sticky event name.
            return self.flush()

        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]

        if field == "event":
            self._event = value.strip()
        elif field == "data":
            self._size += len(value) + 1
            if self._size > MAX_SSE_DATA_CHARS:
                # A hostile or broken upstream must not be able to exhaust
                # memory with a single oversized frame.
                self._event = None
                self._data = []
                self._size = 0
                raise NebulaError(
                    f"SSE frame exceeded {MAX_SSE_DATA_CHARS} characters; aborting."
                )
            self._data.append(value)
        # "id" and "retry" are irrelevant to us.
        return None

    @property
    def current_event(self) -> Optional[str]:
        """The sticky event name that the next ``data:`` line will belong to."""
        return self._event

    def flush(self) -> Optional[Tuple[str, str]]:
        """Return a buffered frame, or None when nothing is pending.

        Also called at end-of-stream: if the connection drops between a
        ``data:`` line and its terminating blank line, the trailing text would
        otherwise be silently discarded.
        """
        if not self._data:
            return None
        frame = (self._event or "message", "\n".join(self._data))
        self._data = []
        self._size = 0
        return frame


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def is_terminal_event(event: Optional[str]) -> bool:
    """True for an event that ends the answer.

    An event that is also a text event is never treated as terminal, so adding
    one to both sets cannot silently drop its payload.
    """
    return bool(event) and event in config.STREAM_COMPLETION_EVENTS and (
        event not in config.STREAM_TEXT_EVENTS
    )


class NebulaClient:
    """Streams decoded answer text for a question in a session.

    Implements the small interface :class:`schoolai.agent.AgentHarness` expects:
    ``stream(question, session_id, stop_event)`` -> iterator of ``str``.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        api_url: str = config.API_URL,
        timeout=config.REQUEST_TIMEOUT,
        debug_path: Optional[str] = None,
    ) -> None:
        self.token = token if token is not None else config.resolve_bearer_token()
        self.api_url = api_url
        self.timeout = timeout
        self.debug_path = debug_path
        self._debug_lock = threading.Lock()

    # -- debug logging -----------------------------------------------------

    def _log_raw(self, event: str, data: str) -> None:
        if not self.debug_path:
            return
        try:
            with self._debug_lock, open(self.debug_path, "a", encoding="utf-8") as handle:
                handle.write(f"event: {event}\ndata: {data}\n\n")
        except OSError:
            self.debug_path = None  # never let logging break the request

    # -- streaming ---------------------------------------------------------

    def stream(
        self,
        question: str,
        session_id: str,
        stop_event: Optional[threading.Event] = None,
        on_raw: Optional[Callable[[str, str], None]] = None,
    ) -> Iterator[str]:
        payload = build_payload(question, session_id)
        headers = build_headers(self.token)

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise NebulaError(f"Could not reach NebulaONE: {exc}") from exc

        with response:
            if response.status_code >= 400:
                body = ""
                try:
                    body = response.text
                except requests.RequestException:
                    pass
                check_status(response.status_code, body)

            decoder = SSEDecoder()
            try:
                for raw_line in response.iter_lines(decode_unicode=False):
                    if stop_event is not None and stop_event.is_set():
                        return
                    if raw_line is None:
                        continue
                    line = raw_line.decode("utf-8", "replace")
                    frame = decoder.feed_line(line)
                    if frame is None:
                        # This server sends `event:` and its `data:` in separate
                        # frames, so a terminal event may never produce a frame
                        # of its own. Stop as soon as it is named, otherwise we
                        # block until the socket closes.
                        if is_terminal_event(decoder.current_event):
                            return
                        continue
                    event, data = frame
                    self._log_raw(event, data)
                    if on_raw is not None:
                        on_raw(event, data)

                    if event in config.STREAM_ERROR_EVENTS:
                        raise NebulaError(f"NebulaONE error event: {data[:300]}")

                    if event in config.STREAM_TEXT_EVENTS:
                        text = decode_chunk(data)
                        if text:
                            yield text

                    if is_terminal_event(event):
                        return

                # The connection dropped between a `data:` line and its blank
                # terminator: don't lose that last chunk.
                trailing = decoder.flush()
                if trailing is not None:
                    event, data = trailing
                    self._log_raw(event, data)
                    if on_raw is not None:
                        on_raw(event, data)
                    if event in config.STREAM_ERROR_EVENTS:
                        raise NebulaError(f"NebulaONE error event: {data[:300]}")
                    if event in config.STREAM_TEXT_EVENTS:
                        text = decode_chunk(data)
                        if text:
                            yield text
            except requests.RequestException as exc:
                raise StreamInterrupted(f"Stream interrupted mid-answer: {exc}") from exc
