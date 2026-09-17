"""Offline self-test: `python run.py --selftest`.

Nothing here touches the network. The agent loop is exercised end-to-end
against a scripted fake transport, so the read / write / feed-back cycle is
verified against a temp workspace on disk. One check drives the real Tk GUI and
skips itself when there is no display.
"""
from __future__ import annotations

import base64
import json
import os
import signal
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

from . import agent, config, nebula, security


# ---------------------------------------------------------------------------
# Tiny test harness
# ---------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    ok: bool
    detail: str = ""


def _raise_timeout(*_args) -> None:
    raise AssertionError("timed out")


def _expect_raises(exc_types, fn: Callable, *args, **kwargs) -> BaseException:
    try:
        fn(*args, **kwargs)
    except exc_types as exc:  # type: ignore[misc]
        return exc
    except Exception as exc:  # wrong exception type
        raise AssertionError(
            f"expected {exc_types} but got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"expected {exc_types} but nothing was raised")


def _events(parser: agent.TagStreamParser, chunks: List[str]) -> List[object]:
    out: List[object] = []
    for chunk in chunks:
        out.extend(parser.feed(chunk))
    out.extend(parser.finish())
    return out


def _text_of(events: List[object]) -> str:
    return "".join(e.text for e in events if isinstance(e, agent.Text))


def _shape(events: List[object]) -> List[Tuple[str, str, str]]:
    shaped = []
    for event in events:
        if isinstance(event, agent.Text):
            shaped.append(("text", "", event.text))
        elif isinstance(event, agent.ToolStart):
            shaped.append(("start", event.kind, event.target))
        elif isinstance(event, agent.ToolCall):
            shaped.append(("call", event.kind, event.target))
    return shaped


# ---------------------------------------------------------------------------
# security.py
# ---------------------------------------------------------------------------


def test_security_reads_and_traversal(tmp: Path) -> None:
    ws = tmp / "ws"
    (ws / "sub").mkdir(parents=True)
    (ws / "a.txt").write_text("hello", encoding="utf-8")
    (ws / "sub" / "b.txt").write_text("bee", encoding="utf-8")
    outside = tmp / "secret.txt"
    outside.write_text("top secret", encoding="utf-8")

    assert security.read_text_file(ws, "a.txt") == ("a.txt", "hello")
    assert security.read_text_file(ws, "sub/b.txt") == (("sub/b.txt"), "bee")

    # Quotes and backticks models like to add are stripped.
    assert security.read_text_file(ws, '"a.txt"')[0] == "a.txt"
    assert security.read_text_file(ws, "`a.txt`")[0] == "a.txt"

    # --- the important part: escapes must be refused ---
    for evil in (
        "../secret.txt",
        "../../etc/passwd",
        "sub/../../secret.txt",
        "..\\..\\etc\\passwd",  # windows-style separators
        str(outside),  # absolute path outside
        "/etc/passwd",
    ):
        _expect_raises(security.PathTraversalError, security.read_text_file, ws, evil)

    # symlink to a file outside the workspace
    (ws / "link.txt").symlink_to(outside)
    _expect_raises(security.PathTraversalError, security.read_text_file, ws, "link.txt")

    # symlinked directory escaping the workspace
    (ws / "linkdir").symlink_to(tmp)
    _expect_raises(security.PathTraversalError, security.read_text_file, ws, "linkdir/secret.txt")

    # malformed / unusable targets
    _expect_raises(security.PathRejectedError, security.read_text_file, ws, "missing.txt")
    _expect_raises(security.PathRejectedError, security.read_text_file, ws, "sub")
    _expect_raises(security.PathRejectedError, security.read_text_file, ws, "   ")
    _expect_raises(security.PathRejectedError, security.read_text_file, ws, "a\x00b")
    _expect_raises(security.WorkspaceNotSetError, security.read_text_file, None, "a.txt")


def test_security_writes(tmp: Path) -> None:
    ws = tmp / "wsw"
    ws.mkdir()

    rel, written = security.write_text_file(ws, "nested/deep/c.txt", "xyz")
    assert rel == "nested/deep/c.txt" and written == 3
    assert (ws / "nested" / "deep" / "c.txt").read_text(encoding="utf-8") == "xyz"

    _expect_raises(security.PathTraversalError, security.write_text_file, ws, "../escape.txt", "x")
    _expect_raises(security.PathTraversalError, security.write_text_file, ws, "/tmp/evil.txt", "x")
    assert not (tmp / "escape.txt").exists()

    _expect_raises(security.PathRejectedError, security.write_text_file, ws, "nested", "x")
    _expect_raises(
        security.PathRejectedError, security.write_text_file, ws, "big.txt", "x" * (security.MAX_WRITE_BYTES + 1)
    )


# ---------------------------------------------------------------------------
# agent.TagStreamParser
# ---------------------------------------------------------------------------


def test_security_rejects_non_regular_files(tmp: Path) -> None:
    ws = tmp / "fifows"
    ws.mkdir()
    os.mkfifo(ws / "pipe")
    # Opening a FIFO blocks forever; without the lstat + O_NONBLOCK guard this
    # test would hang the whole suite instead of failing.
    _expect_raises(security.PathRejectedError, security.read_text_file, ws, "pipe")
    _expect_raises(security.PathRejectedError, security.write_text_file, ws, "pipe", "x")

    # Sanity: a real file in the same directory still works.
    (ws / "ok.txt").write_text("fine", encoding="utf-8")
    assert security.read_text_file(ws, "ok.txt") == ("ok.txt", "fine")


def test_settings_file_permissions(tmp: Path) -> None:
    target = tmp / "cfg" / "config.json"
    original = config.SETTINGS_PATH
    config.SETTINGS_PATH = target
    try:
        config.save_settings({"bearer_token": "super-secret"})
        assert target.exists()
        assert json.loads(target.read_text(encoding="utf-8"))["bearer_token"] == "super-secret"
        # Credential file and its directory must not be world-readable.
        assert oct(target.stat().st_mode & 0o777) == "0o600", oct(target.stat().st_mode & 0o777)
        assert oct(target.parent.stat().st_mode & 0o777) == "0o700", oct(
            target.parent.stat().st_mode & 0o777
        )
        assert not (tmp / "cfg" / "config.json.tmp").exists()  # replaced atomically
        # Overwriting keeps the restrictive mode.
        config.save_settings({"bearer_token": "second"})
        assert oct(target.stat().st_mode & 0o777) == "0o600"
    finally:
        config.SETTINGS_PATH = original


def test_parser_does_not_buffer_without_bound() -> None:
    # A tag prefix that never completes must be released, not accumulated.
    parser = agent.TagStreamParser(max_pending_tag_chars=256)
    payload = '<WRITE_FILE name="' + "a" * 1000
    events = parser.feed(payload)
    assert parser.pending_chars == 0, parser.pending_chars
    assert _text_of(events) == payload

    # An opened tag whose body never closes is aborted at the size cap.
    parser = agent.TagStreamParser(max_tool_bytes=128)
    events = parser.feed('<WRITE_FILE name="x.txt">')
    events += parser.feed("B" * 500)
    assert parser.pending_chars == 0, parser.pending_chars
    calls = [e for e in events if isinstance(e, agent.ToolCall)]
    assert len(calls) == 1, calls
    assert calls[0].error, calls[0]
    assert calls[0].target == "x.txt"
    assert parser.finish() == [], "aborted body must not be resurrected"


def test_aborted_tool_call_never_touches_disk(tmp: Path) -> None:
    ws = tmp / "abortws"
    ws.mkdir()
    outcome = agent.execute_tool_call(
        ws, agent.ToolCall(kind="write", target="x.txt", error="too big")
    )
    assert outcome.ok is False
    assert "ERROR" in outcome.result_text
    assert not (ws / "x.txt").exists()


def test_parser_plain_text() -> None:
    assert _shape(_events(agent.TagStreamParser(), ["hello world"])) == [
        ("text", "", "hello world")
    ]
    # A literal "<" that never becomes a tag must be released, not swallowed.
    assert _text_of(_events(agent.TagStreamParser(), ["a < b", " and c < d"])) == "a < b and c < d"
    # ...including when the chunk boundary lands right after the "<".
    assert _text_of(_events(agent.TagStreamParser(), ["a <", " b"])) == "a < b"
    # ...and when the stream simply ends on a stray "<".
    assert _text_of(_events(agent.TagStreamParser(), ["done <"])) == "done <"
    # No tools means no tool events at all.
    assert _shape(_events(agent.TagStreamParser(), ["a < b"])) == [("text", "", "a < b")]


def test_parser_read_tag(tmp: Path) -> None:
    assert _shape(
        _events(agent.TagStreamParser(), ["prefix <READ_FILE>a.txt</READ_FILE> suffix"])
    ) == [
        ("text", "", "prefix "),
        ("start", "read", ""),
        ("call", "read", "a.txt"),
        ("text", "", " suffix"),
    ]

    # Split across three chunks, mid-tag-name.
    assert _shape(_events(agent.TagStreamParser(), ["<READ_FI", "LE>a.tx", "t</READ_FILE>"])) == [
        ("start", "read", ""),
        ("call", "read", "a.txt"),
    ]

    # Case-insensitive, and two tags back to back.
    assert _shape(
        _events(agent.TagStreamParser(), ["<read_file>a</READ_file><READ_FILE>b</READ_FILE>"])
    ) == [
        ("start", "read", ""),
        ("call", "read", "a"),
        ("start", "read", ""),
        ("call", "read", "b"),
    ]


def test_parser_write_tag() -> None:
    events = _events(
        agent.TagStreamParser(),
        [
            "Sure. <WRITE_FILE na",
            'me="out.txt">print(1)\n',
            "</WRITE_FILE>",
            " Done.",
        ],
    )
    assert _shape(events) == [
        ("text", "", "Sure. "),
        ("start", "write", "out.txt"),
        ("call", "write", "out.txt"),
        ("text", "", " Done."),
    ]
    call = [e for e in events if isinstance(e, agent.ToolCall)][0]
    assert call.content == "print(1)\n"

    # Single-quoted / unquoted names and other attributes.
    assert _shape(_events(agent.TagStreamParser(), ["<WRITE_FILE name='x.py'>y</WRITE_FILE>"])) == [
        ("start", "write", "x.py"),
        ("call", "write", "x.py"),
    ]
    assert _shape(
        _events(agent.TagStreamParser(), ['<WRITE_FILE lang="py" name="z.py">y</WRITE_FILE>'])
    ) == [("start", "write", "z.py"), ("call", "write", "z.py")]

    # Missing name must still produce a call so the harness can report an error.
    assert _shape(_events(agent.TagStreamParser(), ["<WRITE_FILE>y</WRITE_FILE>"])) == [
        ("start", "write", ""),
        ("call", "write", ""),
    ]


def test_parser_unterminated() -> None:
    events = _events(agent.TagStreamParser(), ["<READ_FILE>a.txt"])
    tools = [e for e in events if not isinstance(e, agent.Text)]
    assert _shape(tools) == [("start", "read", "")]
    assert isinstance(events[-1], agent.Text) and "unterminated" in events[-1].text

    # A trailing bare "<" is flushed by finish() rather than lost.
    events = _events(agent.TagStreamParser(), ["done <"])
    assert "".join(e.text for e in events if isinstance(e, agent.Text)) == "done <"


# ---------------------------------------------------------------------------
# nebula.decode_chunk / SSEDecoder / status handling
# ---------------------------------------------------------------------------


def test_decode_chunk() -> None:
    def b64(value: str) -> str:
        return base64.b64encode(value.encode("utf-8")).decode("ascii")

    assert nebula.decode_chunk("") == ""
    assert nebula.decode_chunk(b64("hello")) == "hello"
    assert nebula.decode_chunk(b64("caf\u00e9 \u2713")) == "caf\u00e9 \u2713"

    # base64-wrapped JSON envelope
    assert nebula.decode_chunk(b64(json.dumps({"text": "hoi"}))) == "hoi"
    assert nebula.decode_chunk(b64(json.dumps({"delta": {"text": "deep"}}))) == "deep"

    # raw JSON envelope, no base64
    assert nebula.decode_chunk(json.dumps({"content": "plain"})) == "plain"
    assert nebula.decode_chunk(json.dumps({"text": "hoi"})) == "hoi"

    # non-text JSON yields nothing rather than an exception
    assert nebula.decode_chunk(json.dumps({"unrelated": 1})) == ""

    # Plain text must survive untouched. The base64 branch used to discard
    # punctuation and decode the remainder, mangling chunks like "Go!" -> "\x1a".
    for plain in ("Go!", "Hi!", "Hello, world!", "a < b", "Go", "done.", "x"):
        assert nebula.decode_chunk(plain) == plain, (plain, nebula.decode_chunk(plain))

    # ...while real base64 still decodes, including embedded newlines/tabs.
    assert nebula.decode_chunk(b64("line1\nline2")) == "line1\nline2"
    assert nebula.decode_chunk(b64("\tindented")) == "\tindented"


def test_sse_decoder() -> None:
    d = nebula.SSEDecoder()
    assert d.feed_line(": keep-alive") is None
    assert d.feed_line("event: response-updated") is None
    assert d.feed_line("data: abc") is None
    assert d.feed_line("") == ("response-updated", "abc")

    # multi-line data, default event name, leading space stripped
    d = nebula.SSEDecoder()
    d.feed_line("data: one")
    d.feed_line("data: two")
    assert d.feed_line("") == ("message", "one\ntwo")

    d = nebula.SSEDecoder()
    assert d.feed_line("event: x") is None
    assert d.feed_line("data:") is None
    assert d.feed_line("") == ("x", "")


def test_sse_decoder_nebula_dialect() -> None:
    """Pin the real wire format: `event:` and `data:` arrive in SEPARATE frames.

    Captured live from the endpoint. A spec-compliant decoder tags the payload
    with the default `message` and the whole answer is silently dropped, which
    is exactly the regression this guards.
    """
    lines = [
        "event: conversation-and-segment-id", "",
        "data: eyJDb252ZXJzYXRpb25JZCI6IngifQ==", "",
        "event: step-update", "",
        "data: VGhpbmtpbmc=", "",
        "event: response-updated", "",
        "data: ZWVu", "",
        "event: response-updated", "",
        "data: IHR3ZWU=", "",
        "event: no-more-data", "",
        "data: ", "",
    ]
    decoder = nebula.SSEDecoder()
    frames = [f for f in (decoder.feed_line(line) for line in lines) if f is not None]

    # Exactly one frame per data line -- no spurious blank "message" frames.
    assert [name for name, _ in frames] == [
        "conversation-and-segment-id",
        "step-update",
        "response-updated",
        "response-updated",
        "no-more-data",
    ], frames

    text = "".join(
        nebula.decode_chunk(data)
        for name, data in frames
        if name in nebula.config.STREAM_TEXT_EVENTS
    )
    assert text == "een twee", text

    # The live stream ends with `no-more-data`, so the client must stop there.
    assert "no-more-data" in nebula.config.STREAM_COMPLETION_EVENTS


def test_sse_terminal_event_and_flush() -> None:
    # `no-more-data` is a terminal event, `response-updated` is not.
    assert nebula.is_terminal_event("no-more-data") is True
    assert nebula.is_terminal_event("response-updated") is False
    assert nebula.is_terminal_event(None) is False
    assert nebula.is_terminal_event("") is False

    # A data line left unterminated by an abrupt EOF must still be recoverable.
    payload = base64.b64encode(b"partial").decode("ascii")
    decoder = nebula.SSEDecoder()
    assert decoder.feed_line("event: response-updated") is None
    assert decoder.feed_line("data: " + payload) is None
    assert decoder.current_event == "response-updated"
    assert decoder.flush() == ("response-updated", payload)
    assert decoder.flush() is None  # nothing pending any more


def test_transport_stops_on_event_only_terminal_frame() -> None:
    """`no-more-data` arrives as an event-only frame, with no data of its own.

    Stopping must not depend on the socket closing, or the GUI would hang until
    the server dropped the connection.
    """

    def b64(value: str) -> str:
        return base64.b64encode(value.encode("utf-8")).decode("ascii")

    lines = [
        "event: response-updated", "", "data: " + b64("hello"), "",
        "event: no-more-data", "",
        # Must never be read: the stream should already have stopped.
        "event: response-updated", "", "data: " + b64("SHOULD-NOT-APPEAR"), "",
    ]
    collected: List[str] = []
    _with_fake_post(
        FakeResponse(200, lines),
        lambda: collected.extend(nebula.NebulaClient(token="t").stream("q", "s")),
    )
    assert collected == ["hello"], collected


def test_harness_keeps_client_side_history(tmp: Path) -> None:
    """The endpoint has no memory, so the transcript must be resent each round."""
    ws = tmp / "histws"
    ws.mkdir()
    (ws / "a.txt").write_text("AAA", encoding="utf-8")

    client = FakeClient(
        [
            ["<READ_FILE>a.txt</READ_FILE>"],  # turn 1, round 1
            ["Done with step one."],  # turn 1, round 2 (no tags -> stop)
            ["Second turn answer."],  # turn 2
        ]
    )
    harness = agent.AgentHarness(client=client, workspace=str(ws), session_id="s")
    harness.run_turn("first task", lambda event: None)

    assert "first task" in client.questions[0]
    # The model's own tool tag AND the tool result are both in the transcript,
    # otherwise it cannot tell what it already did.
    assert "<READ_FILE>a.txt</READ_FILE>" in client.questions[1]
    assert 'Contents of "a.txt"' in client.questions[1]

    harness.run_turn("second task", lambda event: None)
    last = client.questions[2]
    assert "first task" in last, "earlier turns must survive into later ones"
    assert "second task" in last
    assert 'Contents of "a.txt"' in last


def test_harness_history_is_bounded(tmp: Path) -> None:
    """One huge file read must not be able to blow past the prompt budget."""
    ws = tmp / "trimws"
    ws.mkdir()
    (ws / "big.txt").write_text("Z" * 50_000, encoding="utf-8")

    client = FakeClient([["<READ_FILE>big.txt</READ_FILE>"], ["ok"]])
    harness = agent.AgentHarness(
        client=client, workspace=str(ws), session_id="s", history_max_chars=8_000
    )
    harness.run_turn("read the big file", lambda event: None)

    rendered = client.questions[1]
    assert len(rendered) < 12_000, len(rendered)
    assert "elided" in rendered, "oversized entry should have been truncated"
    assert 'Contents of "big.txt"' in rendered, "head of the read is still useful"
    assert harness.history[0] == ("user", "read the big file"), "original task kept"


def test_sse_frame_size_cap() -> None:
    d = nebula.SSEDecoder()
    assert d.feed_line("event: response-updated") is None
    _expect_raises(
        nebula.NebulaError,
        d.feed_line,
        "data: " + "a" * (nebula.MAX_SSE_DATA_CHARS + 10),
    )
    # The decoder recovers instead of staying poisoned.
    assert d.feed_line("") is None
    assert d.feed_line("data: ok") is None
    assert d.feed_line("") == ("message", "ok")


def test_check_status() -> None:
    nebula.check_status(200, "")
    nebula.check_status(201, "")
    nebula.check_status(201, "")

    exc = _expect_raises(nebula.AuthError, nebula.check_status, 401, "nope")
    assert exc.status == 401
    _expect_raises(nebula.AuthError, nebula.check_status, 403, "nope")
    _expect_raises(nebula.NebulaError, nebula.check_status, 500, "boom")
    _expect_raises(nebula.NebulaError, nebula.check_status, 429, "")


def test_payload_and_headers() -> None:
    payload = nebula.build_payload("q", "session-1")
    assert set(payload) == {
        "question",
        "visionImageIds",
        "attachmentIds",
        "session",
        "segmentTraceLogLevel",
        "answerGenerationOptions",
        "isTemporary",
    }
    assert payload["question"] == "q"
    assert payload["session"] == {"sessionIdentifier": "session-1"}
    assert payload["segmentTraceLogLevel"] == "NonPersisted"
    assert payload["isTemporary"] is False
    opts = payload["answerGenerationOptions"]
    assert opts["deploymentIdentifier"] == config.DEPLOYMENT_IDENTIFIER
    assert opts["capabilities"] == config.CAPABILITIES
    assert opts["capabilitiesExplicitlyProvided"] is True
    assert opts["reasoningMode"] == config.REASONING_MODE

    headers = nebula.build_headers("token-abc")
    assert headers["authorization"] == "token-abc"
    assert headers["content-type"] == "application/json"
    assert headers["origin"] == "https://ai-chat.hhs.nl"


# ---------------------------------------------------------------------------
# Real transport code path, with requests.post faked
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int, lines: List[str], text: str = "") -> None:
        self.status_code = status_code
        self._lines = lines
        self.text = text
        self.encoding = "utf-8"

    def iter_lines(self, decode_unicode: bool = False):
        for line in self._lines:
            yield line.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _with_fake_post(response: FakeResponse, fn: Callable) -> None:
    original = nebula.requests.post
    nebula.requests.post = lambda *a, **k: response  # type: ignore[assignment]
    try:
        fn()
    finally:
        nebula.requests.post = original  # type: ignore[assignment]


def test_transport_stream_and_auth() -> None:
    def b64(value: str) -> str:
        return base64.b64encode(value.encode("utf-8")).decode("ascii")

    lines = [
        "event: response-updated",
        "data: " + b64("Hel"),
        "",
        ": keep-alive",
        "event: response-updated",
        "data: " + b64("lo"),
        "",
        "event: response-completed",
        "data: " + b64("{}"),
        "",
        "event: response-updated",
        "data: " + b64("SHOULD-NOT-APPEAR"),
        "",
    ]
    client = nebula.NebulaClient(token="t")

    collected: List[str] = []
    _with_fake_post(FakeResponse(200, lines), lambda: collected.extend(client.stream("q", "s")))
    assert collected == ["Hel", "lo"], collected

    # 401 -> AuthError, with the auth-specific message.
    def expect_401() -> None:
        list(nebula.NebulaClient(token="bad").stream("q", "s"))

    exc = _expect_raises(
        nebula.AuthError, _with_fake_post, FakeResponse(401, [], "unauthorized"), expect_401
    )
    assert "token" in str(exc).lower()

    # 500 -> NebulaError carrying the status
    def expect_500() -> None:
        list(nebula.NebulaClient(token="t").stream("q", "s"))

    exc = _expect_raises(
        nebula.NebulaError, _with_fake_post, FakeResponse(500, [], "kaboom"), expect_500
    )
    assert "500" in str(exc) and "kaboom" in str(exc)


# ---------------------------------------------------------------------------
# End-to-end agentic loop against a scripted transport
# ---------------------------------------------------------------------------


class FakeClient:
    """Stands in for NebulaClient: replays scripted chunks, one script per round.

    Once the scripts run out the stream is empty, which is how a real turn
    ends. ``repeat_last`` keeps replaying the final script instead, to exercise
    the loop guard.
    """

    def __init__(self, scripts: List[List[str]], repeat_last: bool = False) -> None:
        self.scripts = scripts
        self.repeat_last = repeat_last
        self.questions: List[str] = []
        self.sessions: List[str] = []

    def stream(self, question: str, session_id: str, stop_event=None):
        self.questions.append(question)
        self.sessions.append(session_id)
        index = len(self.questions) - 1
        if index >= len(self.scripts):
            if not self.repeat_last:
                return
            index = len(self.scripts) - 1
        for chunk in self.scripts[index]:
            if stop_event is not None and stop_event.is_set():
                return
            yield chunk


def test_end_to_end_read_write(tmp: Path) -> None:
    ws = tmp / "e2e"
    ws.mkdir()
    (ws / "hello.txt").write_text("HI-THERE", encoding="utf-8")

    client = FakeClient(
        [
            # Round 1: read a file, then keep talking.
            ["Let me look at ", "<READ_FILE>hello.tx", "t</READ_FILE>", " Now writing."],
            # Round 2: write a file (after being handed the read result).
            [
                'Writing it.\n<WRITE_FILE name="out/result.txt">',
                "print('hi')\n</WRITE_FILE>",
                "\nFiles updated.",
            ],
            # Round 3: react to the write confirmation and stop.
            ["Everything is in place."],
        ]
    )
    harness = agent.AgentHarness(client=client, workspace=str(ws), session_id="sess-1")

    assistant: List[str] = []
    systems: List[str] = []

    def emit(event: object) -> None:
        if isinstance(event, agent.AssistantText):
            assistant.append(event.text)
        elif isinstance(event, agent.SystemMessage):
            systems.append(event.text)

    harness.run_turn("read hello.txt then write something", emit)

    # 1. The file was actually written to disk.
    assert (ws / "out" / "result.txt").read_text(encoding="utf-8") == "print('hi')\n"

    # 2. The read result was fed back as a second (hidden) user message, and the
    #    write confirmation became the third. One request per round, no more.
    assert len(client.questions) == 3, client.questions
    assert 'Contents of "hello.txt"' in client.questions[1]
    assert "HI-THERE" in client.questions[1]
    assert 'OK: wrote 12 bytes to "out/result.txt"' in client.questions[2]

    # 3. Every message carries the hidden system instructions + the session id.
    for question in client.questions:
        assert question.startswith(config.SYSTEM_INSTRUCTIONS)
    assert client.sessions == ["sess-1"] * 3

    # 4. Streamed prose reached the UI, and tags never leaked as prose.
    joined = "".join(assistant)
    assert "Let me look at " in joined and "Files updated." in joined
    assert "Everything is in place." in joined
    assert "<READ_FILE>" not in joined and "<WRITE_FILE" not in joined

    # 5. The mandated UI message is exact.
    assert "✅ Bestand out/result.txt succesvol bewerkt" in systems, systems
    assert any("Read hello.txt" in s for s in systems), systems
    assert not any(s.startswith("⚠️") or s.startswith("❌") for s in systems), systems


def test_end_to_end_blocks_traversal(tmp: Path) -> None:
    ws = tmp / "e2e2"
    ws.mkdir()

    client = FakeClient(
        [
            ["Sneaky: <READ_FILE>../../etc/passwd</READ_FILE> done"],
            ["Fine, I'll behave."],
        ]
    )
    harness = agent.AgentHarness(client=client, workspace=str(ws), session_id="s")

    assistant: List[str] = []
    systems: List[str] = []

    def emit(event: object) -> None:
        if isinstance(event, agent.AssistantText):
            assistant.append(event.text)
        elif isinstance(event, agent.SystemMessage):
            systems.append(event.text)

    harness.run_turn("try to escape", emit)

    # Blocked, reported to the user, and the error went back to the model.
    assert any("Read blocked" in s for s in systems), systems
    assert len(client.questions) == 2
    assert "ERROR reading" in client.questions[1]
    assert "passwd" in client.questions[1].lower()

    # Writing outside the workspace is blocked too.
    client2 = FakeClient(
        [['<WRITE_FILE name="../pwned.txt">x</WRITE_FILE>']],
    )
    harness2 = agent.AgentHarness(client=client2, workspace=str(ws), session_id="s2")
    harness2.run_turn("write outside", emit)
    assert not (tmp / "pwned.txt").exists()
    assert any("Write failed" in s for s in systems), systems


def test_tool_calls_per_round_cap(tmp: Path) -> None:
    ws = tmp / "capws"
    ws.mkdir()
    (ws / "a.txt").write_text("x", encoding="utf-8")

    # One round asking for 50 reads must not run (or accumulate) all 50.
    burst = "<READ_FILE>a.txt</READ_FILE>" * 50
    client = FakeClient([[burst]])
    harness = agent.AgentHarness(
        client=client, workspace=str(ws), session_id="s", max_tool_calls_per_round=5
    )

    systems: List[str] = []
    harness.run_turn(
        "go", lambda e: systems.append(e.text) if isinstance(e, agent.SystemMessage) else None
    )

    assert any("tool calls in one round" in s for s in systems), systems
    reads = [s for s in systems if s.startswith("📖")]
    assert len(reads) == 5, reads
    # The feedback message carries only the executed calls, so it stays bounded.
    assert client.questions[1].count("Contents of") == 5, client.questions[1].count("Contents of")
    # ...and the model is told about the dropped ones so it does not assume they ran.
    assert "were NOT executed" in client.questions[1], client.questions[1]


def test_tool_call_collector_bounds_memory() -> None:
    # The cap must hold *during* collection, not be applied afterwards, or a
    # hostile stream could still build an unbounded list of multi-MB calls.
    collector = agent.ToolCallCollector(limit=3)
    for index in range(1000):
        collector.add(agent.ToolCall(kind="write", target=f"f{index}.txt", content="X" * 4096))
    assert len(collector.calls) == 3, len(collector.calls)
    assert collector.dropped == 997, collector.dropped
    assert collector.accepting is False

    # A zero or negative limit is clamped rather than silently inert.
    assert agent.ToolCallCollector(0).limit == 1
    assert agent.ToolCallCollector(-5).limit == 1


def test_token_normalisation(tmp: Path) -> None:
    """A bare JWT must get the `Bearer ` scheme added, or every call 401s."""
    assert config.normalize_bearer_token("abc.def.ghi") == "Bearer abc.def.ghi"
    assert config.normalize_bearer_token("Bearer abc") == "Bearer abc"
    assert config.normalize_bearer_token("bearer abc") == "bearer abc"  # left as given
    assert config.normalize_bearer_token("  Bearer abc  ") == "Bearer abc"
    assert config.normalize_bearer_token("") == ""
    assert config.normalize_bearer_token(None) == ""

    saved_env = os.environ.get(config.ENV_TOKEN_VAR)
    original_settings = config.SETTINGS_PATH
    config.SETTINGS_PATH = tmp / "no-such-config.json"  # so saved settings are absent
    try:
        os.environ.pop(config.ENV_TOKEN_VAR, None)
        assert config.resolve_bearer_token() == "", "unset token must resolve to empty"
        os.environ[config.ENV_TOKEN_VAR] = "jwt-only"
        assert config.resolve_bearer_token() == "Bearer jwt-only"
    finally:
        config.SETTINGS_PATH = original_settings
        if saved_env is None:
            os.environ.pop(config.ENV_TOKEN_VAR, None)
        else:
            os.environ[config.ENV_TOKEN_VAR] = saved_env


def test_ui_send_cycle(tmp: Path) -> None:
    """Regression guard: a finished run must reset the UI to idle.

    An earlier version built the end-of-run sentinel with its default run id in
    the worker, so the pump discarded it and the Send button stayed disabled
    forever. This drives the real HarnessApp.send() -> worker -> pump path.

    Skipped silently when tkinter is missing or there is no display.
    """
    try:
        import tkinter as tk
    except ImportError:
        return

    from . import ui

    try:
        root = tk.Tk()
    except tk.TclError:
        return  # headless environment

    root.withdraw()
    alarm_supported = hasattr(signal, "SIGALRM")
    if alarm_supported:
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(60)
    try:
        app = ui.HarnessApp(root)
        ws = tmp / "uiws"
        ws.mkdir()
        app.workspace = str(ws)
        app.token_var.set("dummy-token")  # avoids the "no token" prompt

        seen_questions: List[str] = []

        class FakeClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def stream(self, question, session_id, stop_event=None):
                seen_questions.append(question)
                return iter(["Hi there."])

        def one_send(text: str) -> None:
            app.input.delete("1.0", "end")
            app.input.insert("1.0", text)
            app.send()
            app.worker.join(timeout=20)
            assert not app.worker.is_alive(), "worker thread did not finish"
            while not app.events.empty():
                app._handle_event(app.events.get_nowait())

        original = ui.nebula.NebulaClient
        ui.nebula.NebulaClient = FakeClient  # type: ignore[assignment]
        try:
            one_send("hello")
            assert app._run_id == 1, app._run_id
            assert str(app.send_button["state"]) == "normal", (
                f"Send button stuck in {app.send_button['state']!r}: the end-of-run "
                "sentinel did not carry the current run id"
            )
            assert str(app.stop_button["state"]) == "disabled"
            assert "Hi there." in app.chat.get("1.0", "end")

            first_harness = app.harness
            one_send("second message")
            assert app._run_id == 2, app._run_id
        finally:
            ui.nebula.NebulaClient = original  # type: ignore[assignment]

        # The harness (and therefore the transcript) must survive across sends,
        # because the server keeps no conversation state of its own.
        assert app.harness is first_harness, "harness was rebuilt, losing history"
        assert len(seen_questions) == 2, seen_questions
        assert "second message" in seen_questions[1]
        assert "hello" in seen_questions[1], "earlier turn missing from transcript"

        # "New chat" really does start over.
        app.new_chat()
        assert app.harness is None
    finally:
        if alarm_supported:
            signal.alarm(0)
        root.destroy()


def _widget_chain(widget) -> List[object]:
    """Widget plus every ancestor, up to the root."""
    chain = []
    node = widget
    while node is not None:
        chain.append(node)
        node = getattr(node, "master", None)
    return chain


def _grid_weight(widget, axis: str, index: int) -> int:
    """Read a grid row/column weight.

    Tk hands the configuration back either as a dict (some tkinter builds) or as
    a flat ('weight', '1', ...) tuple, so accept both.
    """
    getter = widget.grid_rowconfigure if axis == "row" else widget.grid_columnconfigure
    config = getter(index)
    if isinstance(config, dict):
        return int(config["weight"])
    items = list(config)
    return int(items[items.index("weight") + 1])


def test_ui_layout_is_wired(tmp: Path) -> None:
    """Guards the bug class that `winfo_exists()` cannot see.

    A widget can exist and still be invisible: nothing manages it, it sits in
    the wrong parent, or no weight lets it grow into the space. The chat pane
    once rendered as an empty grey panel for exactly this reason. This checks
    the geometry wiring that has to hold for it to be visible.

    Skipped silently when tkinter is missing or there is no display.
    """
    try:
        import tkinter as tk
    except ImportError:
        return

    from . import ui

    try:
        root = tk.Tk()
    except tk.TclError:
        return  # headless environment

    root.withdraw()
    try:
        app = ui.HarnessApp(root)
        root.update_idletasks()

        # 1. Every interactive widget must be managed by a geometry manager.
        for label, widget in (
            ("chat display", app.chat),
            ("input field", app.input),
            ("send button", app.send_button),
            ("stop button", app.stop_button),
            ("token entry", app.token_entry),
            ("file tree", app.tree),
        ):
            manager = widget.winfo_manager()
            assert manager in ("grid", "pack", "place"), f"{label}: no layout manager"
            assert widget.winfo_reqwidth() > 1 and widget.winfo_reqheight() > 1, (
                f"{label}: degenerate requested size "
                f"{widget.winfo_reqwidth()}x{widget.winfo_reqheight()}"
            )

        # 2. Chat widgets must live under the chat frame, never the sidebar.
        for label, widget in (("chat display", app.chat), ("input field", app.input)):
            chain = set(_widget_chain(widget))
            assert app.chat_frame in chain, f"{label} is not inside the chat frame"
            assert app.sidebar_frame not in chain, f"{label} leaked into the sidebar"

        # 3. Both columns are children of the main frame, which fills the root.
        assert app.sidebar_frame.master is app.main_frame, "sidebar is not in main_frame"
        assert app.chat_frame.master is app.main_frame, "chat frame is not in main_frame"
        # Tk reorders the sticky string (e.g. "nsew" -> "nesw"), so compare as a set.
        sticky = app.main_frame.grid_info()["sticky"]
        assert set(sticky) == set("nsew"), f"main frame sticky is {sticky!r}"

        # 4. The weights that actually let things grow.
        assert _grid_weight(app.root, "row", 1) == 1, "root row 1 cannot grow"
        assert _grid_weight(app.root, "column", 0) == 1, "root column 0 cannot grow"
        assert _grid_weight(app.main_frame, "row", 0) == 1, "main row cannot grow"
        assert _grid_weight(app.main_frame, "column", 1) == 1, "chat column cannot grow"
        assert _grid_weight(app.chat_frame, "row", 0) == 1, "chat display cannot grow"

        # 5. The sidebar keeps a usable minimum instead of collapsing.
        assert int(app.main_frame.grid_columnconfigure(0)["minsize"]) >= 200

        # 6. The empty state must not be a blank panel that looks like a
        #    rendering failure.
        assert app.chat.get("1.0", "end").strip(), "chat opens blank"
    finally:
        root.destroy()


def test_tool_round_limit(tmp: Path) -> None:
    ws = tmp / "loopy"
    ws.mkdir()
    (ws / "a.txt").write_text("x", encoding="utf-8")

    # The model never stops reading; the harness must bail out.
    client = FakeClient([["<READ_FILE>a.txt</READ_FILE>"]], repeat_last=True)
    harness = agent.AgentHarness(client=client, workspace=str(ws), session_id="s", max_tool_rounds=3)

    systems: List[str] = []
    harness.run_turn("loop", lambda e: systems.append(e.text) if isinstance(e, agent.SystemMessage) else None)

    assert any("tool rounds" in s for s in systems), systems
    assert len(client.questions) == 4  # initial + 3 tool rounds


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run() -> int:
    tests: List[Tuple[str, Callable]] = [
        ("security: reads + traversal", test_security_reads_and_traversal),
        ("security: writes", test_security_writes),
        ("security: rejects non-regular files", test_security_rejects_non_regular_files),
        ("config: settings file permissions", test_settings_file_permissions),
        ("config: bearer token normalisation", test_token_normalisation),
        ("parser: plain text", test_parser_plain_text),
        ("parser: READ_FILE", test_parser_read_tag),
        ("parser: WRITE_FILE", test_parser_write_tag),
        ("parser: unterminated tags", test_parser_unterminated),
        ("parser: bounded buffering", test_parser_does_not_buffer_without_bound),
        ("parser: aborted call touches no disk", test_aborted_tool_call_never_touches_disk),
        ("nebula: decode_chunk", test_decode_chunk),
        ("nebula: SSE decoder", test_sse_decoder),
        ("nebula: SSE decoder (live dialect)", test_sse_decoder_nebula_dialect),
        ("nebula: SSE frame size cap", test_sse_frame_size_cap),
        ("nebula: terminal event + EOF flush", test_sse_terminal_event_and_flush),
        ("nebula: stops on event-only terminal", test_transport_stops_on_event_only_terminal_frame),
        ("nebula: status handling", test_check_status),
        ("nebula: payload + headers", test_payload_and_headers),
        ("nebula: transport stream/401/500", test_transport_stream_and_auth),
        ("e2e: read + write + feed-back", test_end_to_end_read_write),
        ("e2e: traversal blocked", test_end_to_end_blocks_traversal),
        ("e2e: tool round limit", test_tool_round_limit),
        ("e2e: tool calls per round cap", test_tool_calls_per_round_cap),
        ("e2e: call collector is bounded", test_tool_call_collector_bounds_memory),
        ("e2e: client-side history", test_harness_keeps_client_side_history),
        ("e2e: history is bounded", test_harness_history_is_bounded),
        ("ui: send cycle returns to idle", test_ui_send_cycle),
        ("ui: layout is wired", test_ui_layout_is_wired),
    ]

    results: List[Result] = []
    for name, fn in tests:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            try:
                if fn.__code__.co_argcount == 1:
                    fn(tmp)
                else:
                    fn()
            except BaseException as exc:  # noqa: BLE001 - report everything
                detail = f"{type(exc).__name__}: {exc}"
                results.append(Result(name, False, detail))
            else:
                results.append(Result(name, True))

    width = max(len(r.name) for r in results)
    failures = 0
    for result in results:
        if result.ok:
            print(f"  PASS  {result.name.ljust(width)}")
        else:
            failures += 1
            print(f"  FAIL  {result.name.ljust(width)}")
            print(f"        {result.detail}")

    print()
    print(f"{len(results) - failures}/{len(results)} checks passed.")
    if failures:
        print("SELFTEST FAILED", file=sys.stderr)
        return 1
    print("SELFTEST OK")
    return 0
