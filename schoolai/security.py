"""Workspace confinement.

Every filesystem access the model can trigger goes through this module. It is
the single place that decides whether a model-supplied path is allowed to
touch the disk, so it fails closed: anything that does not resolve to a real
location strictly inside the workspace raises.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Optional, Tuple

# Refuse to slurp or write absurd payloads (the model controls both).
MAX_READ_BYTES = 512 * 1024
MAX_WRITE_BYTES = 2 * 1024 * 1024

_TEXT_DECODE_ERRORS = "replace"

# O_NOFOLLOW closes the gap between resolve_in_workspace() and the actual open:
# if the final path component is swapped for a symlink in that window, the open
# fails outright instead of following the link out of the workspace.
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

# Opening a FIFO blocks until the other end appears. O_NONBLOCK makes the open
# return immediately (it is a no-op for regular files) so a special file in the
# workspace can never wedge the worker thread; S_ISREG then rejects it.
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


class WorkspaceError(Exception):
    """Base class for workspace/fs problems surfaced to the model + the UI."""


class WorkspaceNotSetError(WorkspaceError):
    """No workspace directory has been selected (or it vanished)."""


class PathTraversalError(WorkspaceError):
    """The requested path escapes the workspace root."""


class PathRejectedError(WorkspaceError):
    """The requested path is malformed or not a usable file target."""


def normalise_model_path(raw: str) -> str:
    """Clean up the path string a model emitted inside a tool tag.

    Models frequently wrap paths in quotes or backticks, and sometimes use
    Windows separators. Normalising here (rather than only in the regex) keeps
    traversal detection honest, because `..\\..\\etc\\passwd` becomes
    `../../etc/passwd` *before* it is resolved.
    """
    if not isinstance(raw, str):
        raise PathRejectedError("filename must be a string")
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'`":
        value = value[1:-1].strip()
    value = value.strip("`").strip()
    if "\x00" in value:
        raise PathRejectedError("filename contains a NUL byte")
    value = value.replace("\\", "/")
    if not value:
        raise PathRejectedError("empty filename")
    return value


def workspace_root(workspace: Optional[os.PathLike]) -> Path:
    if workspace is None:
        raise WorkspaceNotSetError("no workspace selected")
    root = Path(workspace).expanduser()
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise WorkspaceNotSetError(f"workspace is not accessible: {exc}") from exc
    if not root.is_dir():
        raise WorkspaceNotSetError(f"workspace is not a directory: {root}")
    return root


def resolve_in_workspace(workspace: Optional[os.PathLike], raw_path: str) -> Path:
    """Resolve a model-supplied path, guaranteeing it lands inside `workspace`.

    Symlinks are resolved with `os.path.realpath`, so a link inside the
    workspace that points outside it is rejected too. The target itself does
    not have to exist yet (writes create files).
    """
    root = workspace_root(workspace)
    value = normalise_model_path(raw_path)

    candidate = Path(value) if os.path.isabs(value) else (root / value)

    # realpath() collapses "..", "~"(already expanded) and symlinks for both
    # existing and not-yet-existing paths, so the comparison below is sound.
    resolved = Path(os.path.realpath(str(candidate)))

    if resolved != root and root not in resolved.parents:
        raise PathTraversalError(
            f"path escapes the workspace: {raw_path!r} -> {resolved}"
        )
    return resolved


def _relative_display(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def read_text_file(workspace: Optional[os.PathLike], raw_path: str) -> Tuple[str, str]:
    """Read a workspace file. Returns (relative_path, contents)."""
    root = workspace_root(workspace)
    target = resolve_in_workspace(root, raw_path)
    rel = _relative_display(root, target)

    # Check before opening: a FIFO or device node must be rejected, not opened.
    try:
        info = os.lstat(str(target))
    except FileNotFoundError as exc:
        raise PathRejectedError(f"file not found: {rel}") from exc
    except OSError as exc:
        raise PathRejectedError(f"cannot access {rel}: {exc}") from exc

    if stat.S_ISDIR(info.st_mode):
        raise PathRejectedError(f"{rel} is a directory")
    if not stat.S_ISREG(info.st_mode):
        raise PathRejectedError(f"{rel} is not a regular file")
    if info.st_size > MAX_READ_BYTES:
        raise PathRejectedError(
            f"file is too large to read ({info.st_size} bytes > {MAX_READ_BYTES})"
        )

    try:
        descriptor = os.open(str(target), os.O_RDONLY | _NOFOLLOW | _NONBLOCK)
    except OSError as exc:
        raise PathRejectedError(f"cannot open {rel}: {exc}") from exc

    with os.fdopen(descriptor, "rb") as handle:
        # Re-check on the descriptor, in case the path was swapped for a
        # symlink or special file between the lstat above and this open.
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise PathRejectedError(f"{rel} is not a regular file")
        # Bounded even if the file grows between the stat above and the read.
        data = handle.read(MAX_READ_BYTES + 1)

    if len(data) > MAX_READ_BYTES:
        raise PathRejectedError(f"{rel} exceeds the {MAX_READ_BYTES} byte read limit")
    return rel, data.decode("utf-8", _TEXT_DECODE_ERRORS)


def write_text_file(workspace: Optional[os.PathLike], raw_path: str, content: str) -> Tuple[str, int]:
    """Write a workspace file, creating parent directories. Returns (rel, bytes)."""
    root = workspace_root(workspace)
    target = resolve_in_workspace(root, raw_path)
    if target == root or target.is_dir():
        raise PathRejectedError(
            f"{_relative_display(root, target)} is a directory, not a file"
        )
    payload = (content or "").encode("utf-8")
    if len(payload) > MAX_WRITE_BYTES:
        raise PathRejectedError(
            f"refusing to write {len(payload)} bytes (> {MAX_WRITE_BYTES})"
        )
    # mkdir on the *resolved* parent is safe: resolve_in_workspace already
    # proved that path is inside the root.
    target.parent.mkdir(parents=True, exist_ok=True)

    # Refuse to clobber a FIFO/device: opening one for writing blocks too.
    try:
        existing = os.lstat(str(target))
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise PathRejectedError(f"cannot access {target}: {exc}") from exc
    if existing is not None and not stat.S_ISREG(existing.st_mode):
        raise PathRejectedError(
            f"{_relative_display(root, target)} exists and is not a regular file"
        )

    try:
        descriptor = os.open(
            str(target),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _NOFOLLOW | _NONBLOCK,
            0o666,  # normal umask semantics, same as Path.write_bytes()
        )
    except OSError as exc:
        raise PathRejectedError(
            f"cannot write {_relative_display(root, target)}: {exc}"
        ) from exc
    with os.fdopen(descriptor, "wb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise PathRejectedError(
                f"{_relative_display(root, target)} is not a regular file"
            )
        handle.write(payload)
    return _relative_display(root, target), len(payload)
