#!/usr/bin/env python3
"""Entry point for the SchoolAI local coding harness.

Usage:
    python run.py                 # launch the GUI
    python run.py --debug-sse     # also dump every raw SSE frame to schoolai_sse.log
    python run.py --selftest      # offline checks (no network, no GUI)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _launch_gui(debug_sse: bool) -> int:
    try:
        import tkinter as tk
    except ImportError:
        print(
            "tkinter is not available in this Python build.\n"
            "On macOS install python.org Python, or run: brew install python-tk",
            file=sys.stderr,
        )
        return 2

    from schoolai import ui

    root = tk.Tk()
    ui.HarnessApp(root, debug_sse=debug_sse)
    root.mainloop()
    return 0


def main(argv: list) -> int:
    args = set(argv[1:])

    if "--selftest" in args:
        from schoolai import selftest

        return selftest.run()

    if "--help" in args or "-h" in args:
        print(__doc__)
        return 0

    unknown = args - {"--debug-sse"}
    if unknown:
        print(f"Unknown option(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        return 2

    return _launch_gui(debug_sse="--debug-sse" in args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
