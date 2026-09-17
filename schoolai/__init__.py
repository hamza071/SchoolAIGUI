"""SchoolAI — a local AI coding harness with regex-driven file tools.

The upstream model has no native function calling, so instead of a tool API we
inject a system prompt that teaches it XML-ish tags and parse those tags out of
the token stream as it arrives.
"""
from __future__ import annotations

__version__ = "0.1.0"
