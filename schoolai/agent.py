"""Agentic layer.

Three pieces:

* :class:`TagStreamParser` — an incremental scanner that turns a token stream
  into visible text plus completed tool calls, correctly handling tags that are
  split across chunk boundaries.
* :func:`execute_tool_call` — performs the read/write through
  :mod:`schoolai.security`, never touching the disk directly.
* :class:`AgentHarness` — drives the read -> feed back -> continue loop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, Union

from . import config, security

# ---------------------------------------------------------------------------
# Tag grammar
# ---------------------------------------------------------------------------

READ_CLOSE = "</READ_FILE>"
WRITE_CLOSE = "</WRITE_FILE>"

# A *complete* open tag. Case-insensitive: models are inconsistent about case.
OPEN_TAG_RE = re.compile(r"<READ_FILE\s*>|<WRITE_FILE\b[^>]*>", re.IGNORECASE)

_WRITE_NAME_RE = re.compile(
    r"""name\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.IGNORECASE
)

# Lower-case signatures used to decide whether a trailing "<..." fragment might
# still grow into a real tag (and therefore must not be shown to the user yet).
_TAG_SIGNATURES = ("<read_file", "<write_file")

# A genuine open tag is a few dozen bytes. Holding back more than this while
# waiting for one to complete means it never will, so the buffer is released
# rather than grown without bound.
MAX_PENDING_TAG_CHARS = 8192


# ---------------------------------------------------------------------------
# Parser events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Text:
    """Plain assistant prose that is safe to display."""

    text: str


@dataclass(frozen=True)
class ToolStart:
    """A tool tag opened; its body is still streaming in."""

    kind: str  # "read" | "write"
    target: str  # filename for write; "" for read (it is the tag body)


@dataclass(frozen=True)
class ToolCall:
    """A fully received tool invocation."""

    kind: str  # "read" | "write"
    target: str  # filename, as the model wrote it (untrusted)
    content: str = ""  # file body for writes
    error: str = ""  # set when the tag was aborted (e.g. an oversized body)


ParserEvent = Union[Text, ToolStart, ToolCall]


# ---------------------------------------------------------------------------
# Incremental tag scanner
# ---------------------------------------------------------------------------


class TagStreamParser:
    """Feed it stream chunks, get back :class:`Text` / :class:`ToolStart` /
    :class:`ToolCall` events.

    The scanner deliberately holds back a trailing fragment that could still be
    the beginning of a tag, so a tag split across two chunks (or even a single
    bare ``"<"``) never leaks into the chat as visible text.
    """

    def __init__(
        self,
        max_tool_bytes: int = security.MAX_WRITE_BYTES,
        max_pending_tag_chars: int = MAX_PENDING_TAG_CHARS,
    ) -> None:
        self._buf = ""
        self._open: Optional[dict] = None
        self._max_tool_bytes = max_tool_bytes
        self._max_pending_tag_chars = max_pending_tag_chars

    # -- public API --------------------------------------------------------

    @property
    def pending_chars(self) -> int:
        """Characters held back while waiting for a tag to complete."""
        return len(self._buf)

    def feed(self, chunk: str) -> List[ParserEvent]:
        if chunk:
            self._buf += chunk
        return self._scan()

    def finish(self) -> List[ParserEvent]:
        """Flush at end-of-stream. Anything still unterminated is surfaced as
        text so the user is never left staring at a swallowed response."""
        events = self._scan()
        if self._buf:
            if self._open is not None:
                # The model opened a tag and never closed it. Show it verbatim
                # instead of silently dropping the rest of the answer.
                events.append(
                    Text("\n[unterminated tool tag, shown verbatim]\n" + self._buf)
                )
            else:
                events.append(Text(self._buf))
            self._buf = ""
        self._open = None
        return events

    # -- internals ---------------------------------------------------------

    def _scan(self) -> List[ParserEvent]:
        events: List[ParserEvent] = []

        while True:
            if self._open is None:
                match = OPEN_TAG_RE.search(self._buf)
                if match is None:
                    safe = self._safe_prefix_len(self._buf)
                    if safe:
                        events.append(Text(self._buf[:safe]))
                        self._buf = self._buf[safe:]
                    elif len(self._buf) > self._max_pending_tag_chars:
                        # Still looks like a tag prefix but is far too long to
                        # ever become one. Release it instead of buffering
                        # without bound (a hostile stream can send endless
                        # `<WRITE_FILE name="aaaa...` with no `>`).
                        events.append(Text(self._buf))
                        self._buf = ""
                    return events

                if match.start() > 0:
                    events.append(Text(self._buf[: match.start()]))
                self._buf = self._buf[match.start() :]

                tag = match.group(0)
                self._open = self._open_block(tag)
                self._buf = self._buf[len(tag) :]
                events.append(ToolStart(self._open["kind"], self._open["target"]))
                continue

            # Inside an open block: look for the matching close tag.
            close = self._open["close"]
            index = self._buf.lower().find(close.lower())
            if index == -1:
                if len(self._buf) > self._max_tool_bytes:
                    # Hard cap: abandon the block rather than buffering forever.
                    kind = self._open["kind"]
                    target = self._open["target"]
                    self._open = None
                    self._buf = ""
                    events.append(
                        ToolCall(
                            kind=kind,
                            target=target,
                            error=(
                                f"the <{kind.upper()}_FILE> tag exceeded "
                                f"{self._max_tool_bytes} bytes without a closing tag"
                            ),
                        )
                    )
                return events  # need more input

            body = self._buf[:index]
            self._buf = self._buf[index + len(close) :]
            kind = self._open["kind"]
            declared = self._open["target"]
            self._open = None

            if kind == "read":
                events.append(ToolCall(kind="read", target=body.strip()))
            else:
                events.append(ToolCall(kind="write", target=declared, content=body))

    @staticmethod
    def _open_block(tag: str) -> dict:
        low = tag.lower()
        if low.startswith("<read_file"):
            return {"kind": "read", "target": "", "close": READ_CLOSE}
        name_match = _WRITE_NAME_RE.search(tag)
        name = ""
        if name_match:
            name = next((g for g in name_match.groups() if g), "")
        return {"kind": "write", "target": name, "close": WRITE_CLOSE}

    @staticmethod
    def _safe_prefix_len(buf: str) -> int:
        """Length of the prefix of ``buf`` that is guaranteed not to be part of
        a tool tag."""
        index = buf.rfind("<")
        while index != -1:
            if TagStreamParser._could_be_tag_start(buf[index:]):
                return index
            index = buf.rfind("<", 0, index)
        return len(buf)

    @staticmethod
    def _could_be_tag_start(tail: str) -> bool:
        low = tail.lower()
        if any(sig.startswith(low) for sig in _TAG_SIGNATURES):
            return True
        # "<WRITE_FILE name=\"..." with no closing ">" yet.
        if low.startswith("<write_file") and ">" not in low:
            return True
        return False


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


@dataclass
class ToolOutcome:
    """What happened, for both the model and the user."""

    ok: bool
    kind: str
    target: str
    result_text: str  # sent back to the model as the next (hidden) message
    ui_message: str  # one line shown in the chat


_READ_RETRY_HINT = (
    "Use a path relative to the workspace root, without '..' or a leading '/'."
)


def execute_tool_call(workspace, call: ToolCall) -> ToolOutcome:
    """Run one tool call against the workspace, never raising."""
    if call.error:
        return ToolOutcome(
            ok=False,
            kind=call.kind,
            target=call.target,
            result_text=f"ERROR: {call.error}. Re-issue the tag with a closing tag.",
            ui_message=f"⚠️ Tool call rejected ({call.kind}): {call.error}",
        )
    if call.kind == "read":
        return _execute_read(workspace, call)
    if call.kind == "write":
        return _execute_write(workspace, call)
    return ToolOutcome(
        ok=False,
        kind=call.kind,
        target=call.target,
        result_text=f"ERROR: unknown tool {call.kind!r}.",
        ui_message=f"⚠️ Unknown tool: {call.kind}",
    )


def _execute_read(workspace, call: ToolCall) -> ToolOutcome:
    try:
        rel, contents = security.read_text_file(workspace, call.target)
    except security.WorkspaceError as exc:
        return ToolOutcome(
            ok=False,
            kind="read",
            target=call.target,
            result_text=f'ERROR reading "{call.target}": {exc}. {_READ_RETRY_HINT}',
            ui_message=f"⚠️ Read blocked ({call.target}): {exc}",
        )
    return ToolOutcome(
        ok=True,
        kind="read",
        target=rel,
        result_text=(
            f'Contents of "{rel}" ({len(contents)} chars):\n'
            f"```\n{contents}\n```\n(end of file)"
        ),
        ui_message=f"📖 Read {rel} ({len(contents)} chars)",
    )


def _execute_write(workspace, call: ToolCall) -> ToolOutcome:
    if not call.target.strip():
        return ToolOutcome(
            ok=False,
            kind="write",
            target="",
            result_text=(
                'ERROR: the <WRITE_FILE> tag is missing the name attribute. '
                'Use <WRITE_FILE name="path/to/file.ext">...</WRITE_FILE>.'
            ),
            ui_message="❌ Write failed: no filename in the tag",
        )
    try:
        rel, written = security.write_text_file(workspace, call.target, call.content)
    except security.WorkspaceError as exc:
        return ToolOutcome(
            ok=False,
            kind="write",
            target=call.target,
            result_text=f'ERROR writing "{call.target}": {exc}. {_READ_RETRY_HINT}',
            ui_message=f"❌ Write failed ({call.target}): {exc}",
        )
    return ToolOutcome(
        ok=True,
        kind="write",
        target=rel,
        result_text=f'OK: wrote {written} bytes to "{rel}".',
        # Required wording from the specification — keep verbatim.
        ui_message=f"✅ Bestand {rel} succesvol bewerkt",
    )


# ---------------------------------------------------------------------------
# Harness-level (UI-facing) events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssistantText:
    text: str


@dataclass(frozen=True)
class SystemMessage:
    text: str


Emit = Callable[[object], None]


class ToolCallCollector:
    """Accumulates the tool calls of a single round, under a hard cap.

    The cap has to be enforced *while* collecting: a hostile stream can emit
    thousands of tags in one round, and each write call can carry megabytes of
    body, so truncating only after the round ends would still let the list grow
    without bound.
    """

    def __init__(self, limit: int) -> None:
        # Clamp to >=1: a zero limit would silently execute nothing and inform
        # nobody, because run_turn returns before reporting the drops.
        self.limit = max(1, int(limit))
        self.calls: List[ToolCall] = []
        self.dropped = 0

    @property
    def accepting(self) -> bool:
        return len(self.calls) < self.limit

    def add(self, call: ToolCall) -> None:
        if self.accepting:
            self.calls.append(call)
        else:
            self.dropped += 1


# ---------------------------------------------------------------------------
# The agentic loop
# ---------------------------------------------------------------------------


class AgentHarness:
    """Runs one user turn, including any number of tool round-trips.

    The endpoint has **no server-side conversation memory**: two POSTs sharing a
    ``sessionIdentifier`` behave like two unrelated requests (verified against
    the live service). The transcript is therefore kept client-side and re-sent
    as the ``question`` every round — without that, the model forgets the
    original task the instant it performs its first tool call.

    It is deliberately transport-agnostic: ``client`` only has to expose
    ``stream(question, session_id, stop_event)`` yielding decoded text, which
    keeps the whole loop testable without a network.
    """

    def __init__(
        self,
        client,
        workspace,
        session_id: str,
        max_tool_rounds: int = config.MAX_TOOL_ROUNDS,
        max_tool_calls_per_round: int = config.MAX_TOOL_CALLS_PER_ROUND,
        history_max_chars: int = config.HISTORY_MAX_CHARS,
    ) -> None:
        self.client = client
        self.workspace = workspace
        self.session_id = session_id
        self.max_tool_rounds = max_tool_rounds
        self.max_tool_calls_per_round = max_tool_calls_per_round
        self.history_max_chars = history_max_chars
        self.history: List[Tuple[str, str]] = []

    # -- prompt assembly ---------------------------------------------------

    def _history_chars(self) -> int:
        return sum(len(content) for _, content in self.history)

    def _trim_history(self) -> None:
        """Bound the transcript without ever dropping the original task.

        A single oversized entry (one enormous file read) is truncated first:
        the removal loop below only deletes whole entries and never touches
        index 0, so on its own it could not bring one huge entry under budget.
        """
        # Never let the per-entry budget exceed the overall cap, or a small cap
        # would stop binding altogether.
        per_entry = min(self.history_max_chars, max(4_000, self.history_max_chars // 4))
        for index, (role, content) in enumerate(self.history):
            if len(content) > per_entry:
                keep = per_entry // 2
                elided = len(content) - keep * 2
                self.history[index] = (
                    role,
                    f"{content[:keep]}\n\n"
                    f"... [{elided} characters elided to bound the prompt] ...\n\n"
                    f"{content[-keep:]}",
                )

        while len(self.history) > 1 and self._history_chars() > self.history_max_chars:
            # Drop the oldest exchange, a pair at a time so a tool result is not
            # orphaned from the call that produced it. Index 0 always survives.
            del self.history[1 : 3 if len(self.history) > 3 else 2]

    def render_question(self) -> str:
        """Render the hidden instructions plus the whole transcript.

        The endpoint has no dedicated system-prompt field, so instructions ride
        along in the user message and are never rendered in the chat.
        """
        self._trim_history()
        parts = [config.SYSTEM_INSTRUCTIONS, "", "---", "Conversation so far:", ""]
        for role, content in self.history:
            parts.append("User:" if role == "user" else "Assistant:")
            parts.append(content)
            parts.append("")
        parts.append(config.CONTINUE_CUE)
        return "\n".join(parts)

    # -- main entry point --------------------------------------------------

    def run_turn(self, user_text: str, emit: Emit, stop_event=None) -> None:
        self.history.append(("user", user_text))

        for round_index in range(self.max_tool_rounds + 1):
            parser = TagStreamParser()
            collector = ToolCallCollector(self.max_tool_calls_per_round)
            raw: List[str] = []

            for chunk in self.client.stream(
                self.render_question(), self.session_id, stop_event=stop_event
            ):
                if stop_event is not None and stop_event.is_set():
                    parser.finish()
                    emit(SystemMessage("⏹️ Stopped."))
                    return
                raw.append(chunk)
                self._dispatch(parser.feed(chunk), emit, collector)

            self._dispatch(parser.finish(), emit, collector)

            # Remember what the model itself said, tags included, so it can see
            # its own earlier steps when the transcript is resent next round.
            self.history.append(("assistant", "".join(raw)))

            calls = collector.calls
            if not calls:
                return  # nothing more to do — normal completion

            if collector.dropped:
                emit(
                    SystemMessage(
                        f"⚠️ Model asked for {len(calls) + collector.dropped} tool calls "
                        f"in one round; only the first {self.max_tool_calls_per_round} "
                        "were run."
                    )
                )

            if round_index >= self.max_tool_rounds:
                emit(
                    SystemMessage(
                        f"⚠️ Stopped after {self.max_tool_rounds} tool rounds "
                        "(possible loop)."
                    )
                )
                return

            results: List[str] = []
            for call in calls:
                outcome = execute_tool_call(self.workspace, call)
                emit(SystemMessage(outcome.ui_message))
                results.append(outcome.result_text)

            if collector.dropped:
                # Tell the model too, otherwise it happily assumes the dropped
                # writes landed.
                results.append(
                    f"ERROR: {collector.dropped} further tool call(s) in this round "
                    f"were NOT executed (limit {self.max_tool_calls_per_round} per "
                    "round). Re-issue them in smaller batches."
                )

            if stop_event is not None and stop_event.is_set():
                emit(SystemMessage("⏹️ Stopped."))
                return

            self.history.append(("user", "\n\n".join(results)))

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _dispatch(events: Iterable[ParserEvent], emit: Emit, collector: ToolCallCollector) -> None:
        for event in events:
            if isinstance(event, Text):
                if event.text:
                    emit(AssistantText(event.text))
            elif isinstance(event, ToolStart):
                # Only writes stream a potentially long body, so only they get
                # a "working..." line; reads resolve almost immediately.
                if event.kind == "write" and collector.accepting:
                    emit(SystemMessage(f"⏳ Writing {event.target or '…'}…"))
            elif isinstance(event, ToolCall):
                collector.add(event)
