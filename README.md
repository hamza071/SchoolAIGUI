# SchoolAI — Local AI Coding Harness

A small desktop app that turns the NebulaONE chat endpoint into an agentic
coding assistant with **read/write access to a folder you pick**.

The endpoint has no native function calling, so the app does three things:

1. Injects a hidden system prompt teaching the model a tiny tag language.
2. Parses those tags out of the SSE token stream **while it is still arriving**.
3. Executes the file operations itself, strictly confined to the workspace.

```
<READ_FILE>src/main.py</READ_FILE>
<WRITE_FILE name="src/main.py">...file contents...</WRITE_FILE>
```

For a Dutch deep-dive of the same architecture (modules, flow, troubleshooting),
see [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

---

## Setup

```bash
cd SchoolAIGUI
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Requirements: `requests` (`urllib3` is pinned below 2.x because this project
was built against a Python linked to LibreSSL, where urllib3 2.x warns and is
unsupported). `tkinter` ships with Python but is **not** in a few macOS builds —
if importing it fails, install python.org Python or `brew install python-tk`.

### About Tk on macOS

macOS ships a deprecated Tk 8.5, so Python prints this on every launch:

```
DEPRECATION WARNING: The system version of Tk is deprecated and may be removed
in a future release. Set TK_SILENCE_DEPRECATION=1 to suppress this warning.
```

The app works with it (all widgets report mapped with correct geometry), but Tk
8.5 has known visual quirks — including unreliable widget background rendering.
For current Tk:

```bash
brew install python-tk        # then recreate .venv with that Python
```

or install python.org Python 3.12+ and build the venv from it.

## Run

```bash
.venv/bin/python run.py                # launch the GUI
.venv/bin/python run.py --debug-sse    # also dump every raw SSE frame to schoolai_sse.log
.venv/bin/python run.py --selftest     # offline checks; no network, no GUI
```

### Bearer token

Two ways to get one.

**Automatic (optional).** Click **Login to AI**. If the `playwright` package is
installed, this opens a *visible* Chromium window pointed at
`https://ai-chat.hhs.nl` and watches network requests and `localStorage` for a
Bearer JWT (up to 5 minutes), then stores it. Without Playwright the button just
shows install instructions. The app never sees your password — you sign in on
the real site in a real browser.

```bash
.venv/bin/python -m pip install playwright
.venv/bin/python -m playwright install chromium   # downloads a browser
```

**Manual.** Paste the token into the toolbar and click **Save token** (or press
Enter in the field). Copy it from Chrome DevTools → **Network** → the request to
`.../byGptSystemId/2e6987f9-...` → **Headers** → `authorization`. That value
already includes the `Bearer ` scheme, but the app adds the scheme itself when it
is missing, so pasting only the JWT works too — and so does pasting a noisy blob
containing a JWT.

Resolution order (highest first):

1. `NEBULA_BEARER_TOKEN` environment variable
2. `~/.schoolai/config.json` — written by **Save token** or **Login to AI**
3. `schoolai/config.py` → `BEARER_TOKEN`

The file is stored with `0600` permissions and is never committed (see
`.gitignore`). A `401`/`403` pops an alert and focuses the token field.

These tokens are short-lived — the one used while building this was valid for
30 minutes — so expect to re-copy it periodically. If a request starts failing
with 401, that is almost always why.

---

## How a turn works

1. You type a request. The app prepends the hidden system instructions and the
   transcript so far, then `POST`s
   `{question, session:{sessionIdentifier}, answerGenerationOptions, …}` with
   `stream=True`.
2. `NebulaClient.stream()` reads the SSE frames and base64-decodes each
   `response-updated` payload into a text chunk.
3. `TagStreamParser.feed()` consumes chunks incrementally. Prose is emitted to
   the chat immediately; a tag is only released once its closing tag arrives.
   A trailing fragment that could still become a tag (even a bare `<`) is held
   back, so tags split across chunk boundaries never leak into the chat.
4. When a tag completes, the file is read or written through
   `schoolai/security.py`, a one-line system message is shown, and the result is
   appended to the transcript.
5. The loop repeats — **resending the whole transcript** — until the model
   replies without any tool tags. `MAX_TOOL_ROUNDS` (25) and
   `MAX_TOOL_CALLS_PER_ROUND` (32) stop runaway loops.

### There is no server-side conversation memory

Reusing `sessionIdentifier` does **not** give you a continuing conversation.
Two POSTs with the same id behave like two unrelated requests — measured against
the live service:

```
same session, req 1: "My favourite colour is purple. Remember that."
same session, req 2: "What is my favourite colour?"  ->  "Onbekend"  (unknown)
brand-new session:   "What is my favourite colour?"  ->  "Onbekend"
```

So `AgentHarness` keeps the transcript client-side and re-sends it as the
`question` on every round. Without that the model forgets the original task the
moment it performs its first tool call — the first live run had it read the file
and then ask *"what would you like me to change in notes.txt?"*.
`HISTORY_MAX_CHARS` (120 000) bounds the transcript, dropping the oldest middle
turns while always keeping your original request.

---

## Security model

All model-supplied paths go through `security.resolve_in_workspace()`:

* `..`, `~` and absolute paths are resolved with `os.path.realpath()` and then
  required to land inside the workspace root, so `../../etc/passwd`,
  `/etc/passwd` and `..\..\etc\passwd` are all refused.
* Symlinks are resolved **before** the check, so a link inside the workspace
  pointing outside it is refused — for reads, writes, *and* the sidebar (an
  escaping symlinked directory is listed but not walkable).
* Files are opened with `O_NOFOLLOW | O_NONBLOCK` and must be `S_ISREG`, so a
  FIFO or device node in the workspace cannot wedge or escape the worker.
* Reads are capped at 512 KB, writes at 2 MB. The tag parser, the SSE frame
  decoder and the tool-call collector all enforce hard caps, so neither the
  parse buffers nor the collected tool calls can grow without bound — see the
  caveats below for what is *not* covered.
* Refusals are reported to you *and* sent back to the model as an `ERROR:`
  message, so it can correct itself instead of silently stalling.
* The Bearer token is written to `~/.schoolai/config.json` at mode `0600` from
  the moment the file is created — never a world-readable window.
* Hidden system instructions are never rendered in the chat.

### What this does *not* protect against

* **Content, not path.** Writes are applied automatically, as specified — there
  is no confirmation prompt. Anything the model can name, it can overwrite.
  Keep the workspace pointed at a folder under version control so you can see
  and revert changes.
* **Indirect prompt injection.** File contents read via `<READ_FILE>` are fed
  back into the model's context. A hostile file in the workspace can therefore
  contain instructions that steer the model. This is inherent to giving the
  agent file access; treat the workspace as trusted input.
* **Local TOCTOU races.** A process that can write to the workspace could swap
  a validated path component for a symlink between the check and the open.
  `O_NOFOLLOW` + an `fstat` re-check cover the final component; intermediate
  components and a pre-existing hardlink to an outside file are narrowed but
  not closed — that would need per-component `openat`. Out of scope for a
  single-user desktop tool.
* **One oversized SSE line.** The 4 MB cap bounds the *accumulated* `data:`
  payload of a frame, but `requests.iter_lines()` materialises a single
  physical line before the decoder can reject it, so a hostile upstream can
  still allocate one large buffer.
* **Visible chat text is not capped.** The hidden tool state is bounded, but
  streamed assistant prose scales with the size of the response, as in any
  chat UI.

---

## Layout

| File | Purpose |
| --- | --- |
| `run.py` | Entry point (`--selftest`, `--debug-sse`) |
| `schoolai/config.py` | Endpoint, payload constants, system prompt, token storage |
| `schoolai/security.py` | Workspace confinement — the only place that touches disk |
| `schoolai/agent.py` | Streaming tag parser, tool execution, agentic loop |
| `schoolai/nebula.py` | HTTP + SSE + base64 decoding |
| `schoolai/ui.py` | tkinter GUI (`queue` + `root.after` threading) |
| `schoolai/selftest.py` | Offline test suite |

---

## The SSE dialect (captured live)

The endpoint is SSE-like but **not** spec-compliant: an `event:` line and its
`data:` payload arrive in *separate* frames, separated by blank lines.

```
event: response-updated
<blank>
data: VGhpbmtpbmc=
<blank>
```

A spec-compliant parser never sees an event with data attached: it tags the
payload with the default `message` and discards the entire answer. `SSEDecoder`
therefore treats the event name as sticky state that applies to the `data:`
lines which follow — which is also why the original hand-written script worked.
Ordinary `event: X / data: Y / blank` frames still parse correctly.

Event vocabulary observed on the live stream:

| Event | Payload | Meaning |
| --- | --- | --- |
| `conversation-and-segment-id` | base64 JSON | conversation/segment ids |
| `step-update` | base64 text (`Thinking`, `Thinking.`) | progress hint |
| `response-updated` | **base64 answer text** | the streaming answer |
| `cosmos-db-session-tokens` | base64 JSON | internal bookkeeping |
| `response-model` | base64 JSON | model / title metadata |
| `no-more-data` | empty | end of the answer |

Only `response-updated` carries answer text and `no-more-data` ends the stream,
so those are the defaults. `step-update` is a natural hook if you later want a
real "thinking" indicator. If the vocabulary shifts, run `--debug-sse` and
extend the sets in `schoolai/config.py`:

```python
STREAM_TEXT_EVENTS = {"response-updated"}          # events carrying answer text
STREAM_COMPLETION_EVENTS = {"no-more-data",        # events that end the answer
                            "response-completed",
                            "message-completed"}
STREAM_ERROR_EVENTS = {"error", "exception"}
```

Decoding stays tolerant: `response-updated` is base64, but the decoder also
accepts base64-wrapped JSON, raw JSON and plain text. A decode is only accepted
if it is valid UTF-8 with no C0 control characters other than `\n`, `\r`, `\t` —
that is what stops plain text being mangled by the base64 branch. One ambiguity
is unavoidable: a payload that is valid as *both* is treated as base64, so the
literal `TWFu` would come through as `Man`.

## Known limitations

* Verified against the live endpoint (stream decoding, the read → write agentic
  loop, and the 401 path). A full live run produced `summary.txt` with exactly
  the right contents. The workspace-escape guard is covered by the offline
  suite rather than live traffic, because the model refused the attack itself
  before any tool call was emitted.
* Writes are applied without confirmation (see above) — keep the workspace
  under version control.
* The GUI is intentionally the plain tkinter/ttk baseline; `ui.py` has no
  dependency on the rest of the app, so it can be swapped for customtkinter or
  PyQt without touching the agent or transport layers.
* Single conversation per window; "New chat" starts a fresh session id.
* UI strings are English except the file-written message, which the
  specification mandates verbatim in Dutch
  (`✅ Bestand <pad> succesvol bewerkt`).
