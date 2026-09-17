"""Tkinter GUI.

Deliberately standard-library only (tkinter + ttk) so the app runs with nothing
but `requests` installed. The chat view is a plain `tk.Text` with tags, which
is enough for streaming and keeps the whole UI layer easy to replace later.

Threading model: network work happens on a daemon worker thread. It never
touches a widget. It pushes events onto a `queue.Queue`, and the Tk main thread
drains that queue from a `root.after()` pump. That is the thread-safe pattern
Tk requires.
"""
from __future__ import annotations

import queue
import threading
import time
import re
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from . import agent, config, nebula, security

try:
    # Playwright is optional; graceful fallback is provided if unavailable.
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover - optional at runtime
    sync_playwright = None
# Noise that would drown the sidebar.
IGNORED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".DS_Store",
}

_TREE_PLACEHOLDER = "\u00b7\u00b7\u00b7"  # "···" marker for unexpanded children

PUMP_INTERVAL_MS = 40


# ---------------------------------------------------------------------------
# Worker -> GUI events
# ---------------------------------------------------------------------------


@dataclass
class UiError:
    message: str
    auth: bool = False


@dataclass
class UiDone:
    """Sentinel marking the end of a worker run.

    ``run_id`` lets a late sentinel from an already-finished run be ignored
    once a new one has started, instead of flickering the UI back to idle.
    """

    run_id: int = 0


class HarnessApp:
    def __init__(self, root: tk.Tk, debug_sse: bool = False) -> None:
        self.root = root
        self.workspace: Optional[str] = None
        self.session_id = config.new_session_id()
        self.token = config.resolve_bearer_token()
        self.debug_path: Optional[str] = "schoolai_sse.log" if debug_sse else None

        self.events: "queue.Queue[object]" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.harness: Optional[agent.AgentHarness] = None
        self.stop_event = threading.Event()
        self._assistant_header_open = False
        self._run_id = 0

        self._build_ui()
        self._show_welcome()
        self._set_busy(False)
        self.root.after(PUMP_INTERVAL_MS, self._pump)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _show_welcome(self) -> None:
        """Seed the chat with an empty state.

        Without this the window opens with a large blank text area, which is
        indistinguishable from a widget that failed to render.
        """
        self._append_chat(
            "SchoolAI \u2014 Local Coding Harness\n"
            "\n"
            "  1. Paste your Bearer token above and click Save token.\n"
            "  2. Click Select Workspace and pick a folder.\n"
            "  3. Type your request below and press Enter\n"
            "     (Shift+Enter for a new line).\n"
            "\n"
            "Tip: double-click a file in the sidebar to ask about it.\n",
            "welcome",
        )

    def _build_ui(self) -> None:
        self.root.title("SchoolAI \u2014 Local Coding Harness")
        self.root.geometry("1280x820")
        self.root.minsize(900, 600)

        # Row/column weights are what let the main area actually fill the
        # window. Without the weight on row 1 the chat area keeps only its
        # requested height and the spare space is left blank.
        self.root.rowconfigure(0, weight=0)  # toolbar: fixed height
        self.root.rowconfigure(1, weight=1)  # main area: absorbs all spare space
        self.root.rowconfigure(2, weight=0)  # status bar: fixed height
        self.root.columnconfigure(0, weight=1)

        self._build_toolbar()

        # Two columns laid out with plain grid weights rather than a
        # ttk.PanedWindow. On macOS Tk 8.5 the paned window handed the chat
        # pane a 1x1 area, which rendered as an empty grey panel with no chat
        # display, input field or Send button in it. Grid weights are
        # predictable everywhere; a paned window is not.
        main = ttk.Frame(self.root)
        main.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=0, minsize=240)  # sidebar: keeps its width
        main.columnconfigure(1, weight=1)  # chat: absorbs the remaining space
        self.main_frame = main

        self.sidebar_frame = self._build_sidebar(main)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.chat_frame = self._build_chat(main)
        self.chat_frame.grid(row=0, column=1, sticky="nsew")

        self.status_var = tk.StringVar(value="Ready. Select a workspace to begin.")
        ttk.Label(
            self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=(6, 3)
        ).grid(row=2, column=0, sticky="ew")

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 8, 8, 6))
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(1, weight=1)

        ttk.Button(bar, text="Select Workspace", command=self.select_workspace).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(bar, text="Login to AI", command=self.login_to_ai).grid(
            row=0, column=6, sticky="e", padx=(6, 0)
        )
        self.workspace_var = tk.StringVar(value="(no workspace selected)")
        ttk.Label(bar, textvariable=self.workspace_var, anchor="w", foreground="#374151").grid(
            row=0, column=1, sticky="ew", padx=10
        )

        ttk.Label(bar, text="Bearer token:").grid(row=0, column=2, padx=(10, 4))
        self.token_var = tk.StringVar(value=self.token)
        self.token_entry = ttk.Entry(bar, textvariable=self.token_var, width=34, show="\u2022")
        self.token_entry.grid(row=0, column=3, sticky="ew")
        # Allow pressing Enter in the token field to save immediately.
        self.token_entry.bind("<Return>", lambda _e: self.save_token())

        # Small show/hide toggle for the token so users can verify paste contents.
        self.token_show_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Show", variable=self.token_show_var, command=self._toggle_token_show).grid(row=0, column=4, padx=(6, 2))

        ttk.Button(bar, text="Save token", command=self.save_token).grid(row=0, column=5, padx=6)

        self.new_chat_button = ttk.Button(bar, text="New chat", command=self.new_chat)
        self.new_chat_button.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.stop_button = ttk.Button(bar, text="Stop", command=self.stop)
        self.stop_button.grid(row=1, column=5, sticky="e", pady=(6, 0), padx=6)

    def _build_sidebar(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=(0, 0, 6, 0))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Workspace", font=("Helvetica", 11, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )

        self.tree = ttk.Treeview(frame, show="tree", selectmode="browse")
        self.tree.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)

        self.tree.bind("<<TreeviewOpen>>", self._on_tree_open)
        self.tree.bind("<Double-1>", self._on_tree_double_click)

        ttk.Label(
            frame,
            text="Double-click a file to insert a READ_FILE request.",
            wraplength=220,
            foreground="#6b7280",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        return frame

    def _build_chat(self, parent: ttk.Frame) -> ttk.Frame:
        # row 0 = chat display (grows), row 1 = input row (fixed height).
        frame = ttk.Frame(parent, padding=(6, 0, 0, 0))
        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(1, weight=0)
        frame.columnconfigure(0, weight=1)  # the chat display absorbs width

        # height/width are only a requested-size floor: sticky="nsew" plus the
        # row/column weights below decide the real size. They exist so the
        # widget can never end up a degenerate 1x1 if geometry is misconfigured.
        self.chat = tk.Text(
            frame,
            wrap="word",
            state="disabled",
            height=20,
            width=60,
            padx=10,
            pady=8,
            background="#ffffff",
            relief="solid",
            borderwidth=1,
        )
        self.chat.grid(row=0, column=0, sticky="nsew")
        chat_scroll = ttk.Scrollbar(frame, orient="vertical", command=self.chat.yview)
        chat_scroll.grid(row=0, column=1, sticky="ns")
        self.chat.configure(yscrollcommand=chat_scroll.set)

        mono = ("Menlo", 11)
        self.chat.tag_configure("header", font=("Helvetica", 11, "bold"), foreground="#1f2937")
        self.chat.tag_configure("user", foreground="#1d4ed8", font=mono)
        self.chat.tag_configure("assistant", foreground="#111827", font=mono)
        self.chat.tag_configure("system", foreground="#047857", font=("Helvetica", 11))
        self.chat.tag_configure("error", foreground="#b91c1c", font=("Helvetica", 11))
        self.chat.tag_configure("welcome", foreground="#4b5563", font=("Helvetica", 11))

        input_row = ttk.Frame(frame)
        input_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        input_row.columnconfigure(0, weight=1)
        self.input = tk.Text(input_row, height=4, wrap="word", relief="solid", borderwidth=1, padx=6, pady=4)
        self.input.grid(row=0, column=0, sticky="ew")
        self.input.bind("<Return>", self._on_return)
        self.input.bind("<Shift-Return>", self._on_shift_return)
        self.input.bind("<Command-Return>", self._on_return)

        # Held as an attribute so _set_busy() can disable it during a run.
        self.send_button = ttk.Button(input_row, text="Send", command=self.send)
        self.send_button.grid(row=0, column=1, sticky="ns", padx=(6, 0))
        return frame

    # ------------------------------------------------------------------
    # Workspace / sidebar
    # ------------------------------------------------------------------

    def select_workspace(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo(
                "Busy", "Stop the current response before changing workspace."
            )
            return
        chosen = filedialog.askdirectory(title="Select workspace folder")
        if not chosen:
            return
        self.workspace = str(Path(chosen).resolve())
        self.workspace_var.set(self.workspace)
        # A different folder is a different conversation.
        self.harness = None
        self._refresh_tree()
        self._set_status(f"Workspace: {self.workspace}")

    def _refresh_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        if not self.workspace:
            return
        root_path = Path(self.workspace)
        root_item = self.tree.insert(
            "", "end", text=root_path.name or str(root_path), open=True, values=(str(root_path),)
        )
        self._populate(root_item, root_path)

    def _populate(self, parent_item: str, path: Path) -> None:
        try:
            entries = sorted(
                path.iterdir(), key=lambda p: (p.is_file() or p.is_symlink(), p.name.lower())
            )
        except OSError:
            return
        for child in entries:
            if child.name in IGNORED_DIR_NAMES:
                continue
            try:
                is_dir = child.is_dir()
            except OSError:
                is_dir = False
            item = self.tree.insert(
                parent_item, "end", text=child.name, values=(str(child),), open=False
            )
            if is_dir and self._is_expandable(child):
                # Placeholder so the expand arrow appears; replaced lazily.
                self.tree.insert(item, "end", text=_TREE_PLACEHOLDER)

    def _is_expandable(self, path: Path) -> bool:
        """Refuse to walk through a symlinked directory that leaves the
        workspace, which would list names from outside it. Reads and writes
        would still be blocked, but there is no reason to show them."""
        if not path.is_symlink():
            return True
        try:
            security.resolve_in_workspace(self.workspace, str(path))
        except security.WorkspaceError:
            return False
        return True

    def _on_tree_open(self, _event=None) -> None:
        item = self.tree.focus()
        if not item:
            return
        children = self.tree.get_children(item)
        if len(children) == 1 and self.tree.item(children[0], "text") == _TREE_PLACEHOLDER:
            self.tree.delete(children[0])
            values = self.tree.item(item, "values")
            if values:
                self._populate(item, Path(values[0]))

    def _on_tree_double_click(self, _event=None) -> None:
        item = self.tree.focus()
        if not item or not self.workspace:
            return
        values = self.tree.item(item, "values")
        if not values:
            return
        target = Path(values[0])
        if not target.is_file():
            return
        try:
            rel = target.relative_to(self.workspace)
        except ValueError:
            return
        self.input.insert("end", f"<READ_FILE>{rel}</READ_FILE>\n")
        self.input.focus_set()

    # ------------------------------------------------------------------
    # Token / session
    # ------------------------------------------------------------------

    def save_token(self) -> None:
        token = config.normalize_bearer_token(self.token_var.get())
        self.token = token
        self.token_var.set(token)  # show the canonical form that was stored
        settings = config.load_settings()
        settings["bearer_token"] = token
        try:
            config.save_settings(settings)
        except OSError as exc:
            messagebox.showerror("Could not save token", str(exc))
            return
        self._set_status(f"Token saved to {config.SETTINGS_PATH}")

    def login_to_ai(self) -> None:
        """Launch a browser to capture a Bearer token automatically.

        This uses Playwright when available. The browser is opened visible so the
        user can complete the login flow; network requests and localStorage are
        monitored for an Authorization header or a JWT-like value.
        """
        if sync_playwright is None:
            messagebox.showinfo(
                "Playwright missing",
                "Automated login requires the `playwright` package and browser binaries.\n"
                "Install with: pip install playwright && playwright install\n"
                "Or paste your token into the toolbar and click Save token.",
            )
            return

        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Stop the current response before logging in.")
            return

        # Run the interactive login in a background thread so the UI stays
        # responsive. It will call back into the Tk thread to update state.
        thread = threading.Thread(target=self._run_login_flow, daemon=True)
        thread.start()

    def _run_login_flow(self) -> None:
        """Background thread: open a Playwright browser and capture a token."""
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=False)
                context = browser.new_context()
                page = context.new_page()

                token_found = {}

                def _on_request(req):
                    try:
                        headers = req.headers
                        auth = headers.get("authorization") or headers.get("Authorization")
                        if auth and isinstance(auth, str) and auth.lower().startswith("bearer "):
                            token_found["token"] = auth
                    except Exception:
                        pass

                page.on("request", _on_request)

                # Navigate and let user log in. Poll localStorage for JWT-ish values
                page.goto("https://ai-chat.hhs.nl", timeout=60_000)

                timeout = 300  # seconds
                deadline = time.time() + timeout
                jwt_re = re.compile(r"eyJ[0-9A-Za-z_\-]+\.[0-9A-Za-z_\-]+\.[0-9A-Za-z_\-]+")
                while time.time() < deadline and "token" not in token_found:
                    # Check localStorage for any JWT-ish value
                    try:
                        storage = page.evaluate(
                            "() => { const r = {}; for (let i=0;i<localStorage.length;i++){const k=localStorage.key(i);r[k]=localStorage.getItem(k);} return r;}"
                        )
                        if isinstance(storage, dict):
                            for v in storage.values():
                                if isinstance(v, str):
                                    m = jwt_re.search(v)
                                    if m:
                                        token_found["token"] = f"Bearer {m.group(0)}"
                                        break
                    except Exception:
                        # page.evaluate can fail if the page is in a cross-origin
                        # state briefly; ignore and continue polling.
                        pass

                    if "token" in token_found:
                        break

                    time.sleep(0.5)

                # Close the browser window we opened.
                try:
                    browser.close()
                except Exception:
                    pass

                if "token" not in token_found:
                    # Nothing captured within timeout.
                    self.root.after(
                        0,
                        lambda: messagebox.showinfo(
                            "Login timed out",
                            "Could not detect a Bearer token automatically.\n"
                            "Please paste the token into the toolbar and click Save token.",
                        ),
                    )
                    return

                raw = token_found["token"]
                token = config.normalize_bearer_token(raw)
                # Persist the token and update the UI on the Tk thread.
                def _commit():
                    self.token_var.set(token)
                    self.token = token
                    settings = config.load_settings()
                    settings["bearer_token"] = token
                    try:
                        config.save_settings(settings)
                        self._set_status("Logged in (token captured)")
                        messagebox.showinfo("Logged in", "Bearer token captured and saved.")
                    except OSError as exc:
                        messagebox.showerror("Could not save token", str(exc))

                self.root.after(0, _commit)
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Login failed", str(exc)))

    def _toggle_token_show(self) -> None:
        """Show or hide the token characters in the token Entry."""
        if getattr(self, "token_show_var", None) and self.token_show_var.get():
            self.token_entry.configure(show="")
        else:
            self.token_entry.configure(show="\u2022")

    def new_chat(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Stop the current response first.")
            return
        self.session_id = config.new_session_id()
        self.harness = None  # fresh session -> fresh transcript
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self._assistant_header_open = False
        self._show_welcome()
        self._set_status(f"New session {self.session_id}")

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def _on_return(self, _event=None) -> str:
        self.send()
        return "break"  # don't insert a newline

    def _on_shift_return(self, _event=None) -> None:
        return None  # allow the default newline

    def send(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        text = self.input.get("1.0", "end").strip()
        if not text:
            return
        if not self.workspace:
            messagebox.showwarning("No workspace", "Select a workspace folder first.")
            return

        token = config.normalize_bearer_token(self.token_var.get())
        if not token:
            if not messagebox.askyesno(
                "No Bearer token",
                "No Bearer token is set, so the request will fail with HTTP 401.\n\n"
                "Send anyway?",
            ):
                return
        self.token = token or self.token

        self.input.delete("1.0", "end")
        self._append_chat("\n\n\u258c You\n", "header")
        self._append_chat(text + "\n", "user")
        self._assistant_header_open = False

        self.stop_event.clear()
        self._run_id += 1
        # Build the end-of-run sentinel here, where the run id is known, so the
        # worker cannot accidentally emit one carrying the default id (which
        # the pump would discard, leaving the UI stuck in the busy state).
        done = UiDone(self._run_id)
        self._set_busy(True)
        self._set_status("Thinking\u2026")
        self.worker = threading.Thread(
            target=self._run_worker, args=(text, done), daemon=True
        )
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        self._set_status("Stopping\u2026")

    def _run_worker(self, text: str, done: UiDone) -> None:
        """Runs OFF the Tk thread. Communicates only via the queue."""
        try:
            client = nebula.NebulaClient(token=self.token, debug_path=self.debug_path)
            # Reuse the harness so the transcript (and therefore multi-turn
            # memory, which the server does NOT provide) survives across sends.
            harness = self.harness
            if harness is None or harness.session_id != self.session_id:
                harness = agent.AgentHarness(
                    client=client, workspace=self.workspace, session_id=self.session_id
                )
                self.harness = harness
            else:
                harness.client = client
                harness.workspace = self.workspace
            harness.run_turn(text, emit=self.events.put, stop_event=self.stop_event)
        except nebula.AuthError as exc:
            self.events.put(UiError(str(exc), auth=True))
        except nebula.NebulaError as exc:
            self.events.put(UiError(str(exc)))
        except Exception as exc:
            self.events.put(UiError(f"Unexpected error: {type(exc).__name__}: {exc}"))
        finally:
            self.events.put(done)

    # ------------------------------------------------------------------
    # Event pump (Tk thread)
    # ------------------------------------------------------------------

    def _pump(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)
        self.root.after(PUMP_INTERVAL_MS, self._pump)

    def _handle_event(self, event: object) -> None:
        if isinstance(event, UiDone):
            # A late sentinel from a superseded run must not reset the UI.
            if event.run_id != self._run_id:
                return
            self._set_busy(False)
            if not self.stop_event.is_set():
                self._set_status("Ready.")
            return

        if isinstance(event, agent.AssistantText):
            if not self._assistant_header_open:
                self._append_chat("\n\n\u258c AI\n", "header")
                self._assistant_header_open = True
            self._append_chat(event.text, "assistant")

        elif isinstance(event, agent.SystemMessage):
            self._append_chat("\n  " + event.text + "\n", "system")
            self._assistant_header_open = False

        elif isinstance(event, UiError):
            self._append_chat("\n  \u26a0\ufe0f " + event.message + "\n", "error")
            self._assistant_header_open = False
            self._set_status(event.message)
            if event.auth:
                # Deferred so the pump returns before the modal blocks.
                self.root.after(
                    10,
                    lambda message=event.message: (
                        messagebox.showerror(
                            "Authentication failed",
                            message
                            + "\n\nPaste a fresh Bearer token in the toolbar and click "
                            "'Save token'.",
                        ),
                        self.token_entry.focus_set(),
                    ),
                )

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _append_chat(self, text: str, tag: str) -> None:
        self.chat.configure(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.send_button.configure(state=state)
        self.stop_button.configure(state="normal" if busy else "disabled")
        self.new_chat_button.configure(state=state)
