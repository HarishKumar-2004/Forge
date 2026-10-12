"""File-system tools exposed to the chat agent via Ollama tool-calling, sandboxed to the project directory."""

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).parent.resolve()

# Directories no tool may read, write, or list — own conversation memory, audit logs, and tooling cruft.
BLOCKED_DIR_NAMES = {"memory", "logs", ".venv", "__pycache__", ".git", "node_modules"}

MAX_LIST_ENTRIES = 500
MAX_SEARCH_MATCHES = 200
MAX_SEARCH_FILE_BYTES = 2_000_000
MAX_READ_LINES = 1000
MAX_LINE_DISPLAY_CHARS = 300
MAX_WRITE_BYTES = 1_000_000


class ToolError(Exception):
    """Expected, user-facing tool failure such as a bad path, ambiguous edit, or size cap."""


def resolve_sandboxed_path(path_str: str, must_exist: bool = False) -> Path:
    """Resolve a model-supplied relative path to an absolute Path confined to the project sandbox."""
    if not path_str or not path_str.strip():
        raise ToolError("Path must not be empty.")
    raw = Path(path_str)
    if raw.is_absolute():
        raise ToolError(f"Absolute paths are not allowed: '{path_str}'. Use a path relative to the project root.")
    candidate = (PROJECT_ROOT / raw).resolve()
    try:
        rel = candidate.relative_to(PROJECT_ROOT)
    except ValueError:
        raise ToolError(f"Path '{path_str}' resolves outside the project sandbox; rejected.")
    if rel.parts and rel.parts[0] in BLOCKED_DIR_NAMES:
        raise ToolError(f"Path '{path_str}' is inside a restricted directory ('{rel.parts[0]}') and cannot be accessed.")
    if must_exist and not candidate.exists():
        raise ToolError(f"'{path_str}' does not exist.")
    return candidate


LIST_FILES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": (
            "List files and directories within the project sandbox. Returns names, "
            "types (file/dir), and sizes. Paths are relative to the project root; "
            "absolute paths and '..' traversal outside the project are rejected."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list, relative to the project root. Defaults to the project root ('.')."},
                "recursive": {"type": "boolean", "description": "If true, list all files under subdirectories too. Defaults to false (top-level only)."},
            },
            "required": [],
        },
    },
}


def list_files(path: str = ".", recursive: bool = False) -> dict:
    """List files and directories under the sandbox, optionally recursing into subdirectories."""
    base = resolve_sandboxed_path(path, must_exist=True)
    if not base.is_dir():
        raise ToolError(f"'{path}' is not a directory.")
    iterator = sorted(base.rglob("*")) if recursive else sorted(base.iterdir())
    entries = []
    truncated = False
    for p in iterator:
        rel = p.relative_to(base)
        if any(part in BLOCKED_DIR_NAMES or part.startswith(".") for part in rel.parts):
            continue
        if len(entries) >= MAX_LIST_ENTRIES:
            truncated = True
            break
        entries.append({
            "name": str(rel),
            "type": "dir" if p.is_dir() else "file",
            "size": p.stat().st_size if p.is_file() else None,
        })
    return {"path": path, "recursive": recursive, "entries": entries,
            "total_found": len(entries), "truncated": truncated}


SEARCH_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_file",
        "description": (
            "Search for a text pattern inside files under the project sandbox "
            "(grep-style). Searches a single file or recursively under a directory, "
            "skipping binary and oversized files. Returns matching file paths with "
            "line numbers and line text, capped and truncated for safety."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Text or regular expression to search for."},
                "path": {"type": "string", "description": "File or directory to search, relative to the project root. Defaults to the whole project ('.')."},
                "regex": {"type": "boolean", "description": "Treat 'pattern' as a regular expression instead of a literal substring. Defaults to false."},
                "case_sensitive": {"type": "boolean", "description": "Match case exactly. Defaults to false (case-insensitive)."},
            },
            "required": ["pattern"],
        },
    },
}


def search_file(pattern: str, path: str = ".", regex: bool = False, case_sensitive: bool = False) -> dict:
    """Search file contents under the sandbox for a literal or regex pattern, grep-style."""
    base = resolve_sandboxed_path(path, must_exist=True)
    flags = 0 if case_sensitive else re.IGNORECASE

    if regex:
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            raise ToolError(f"Invalid regular expression: {e}")

        def matches_line(line: str) -> bool:
            return compiled.search(line) is not None
    else:
        needle = pattern if case_sensitive else pattern.lower()

        def matches_line(line: str) -> bool:
            return needle in (line if case_sensitive else line.lower())

    candidates = [base] if base.is_file() else sorted(base.rglob("*"))
    matches = []
    truncated = False
    for f in candidates:
        if not f.is_file():
            continue
        rel = f.relative_to(PROJECT_ROOT)
        if any(part in BLOCKED_DIR_NAMES or part.startswith(".") for part in rel.parts[:-1]):
            continue
        if f.stat().st_size > MAX_SEARCH_FILE_BYTES:
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_num, line in enumerate(text.splitlines(), start=1):
            if matches_line(line):
                display = line if len(line) <= MAX_LINE_DISPLAY_CHARS else line[:MAX_LINE_DISPLAY_CHARS] + "..."
                matches.append({"file": str(rel), "line": line_num, "text": display})
                if len(matches) >= MAX_SEARCH_MATCHES:
                    truncated = True
                    break
        if truncated:
            break
    return {"pattern": pattern, "path": path, "matches": matches,
            "match_count": len(matches), "truncated": truncated}


READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read the contents of a file within the project sandbox, optionally "
            "limited to a line range. Returns line-numbered text (cat -n style). "
            "Large files are truncated with a note; pass start_line/end_line to "
            "read a specific range instead of the whole file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File to read, relative to the project root."},
                "start_line": {"type": "integer", "description": "First line to read (1-indexed, inclusive). Omit to start from the beginning."},
                "end_line": {"type": "integer", "description": "Last line to read (1-indexed, inclusive). Omit to read to the end or to the truncation cap."},
            },
            "required": ["path"],
        },
    },
}


def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> dict:
    """Read a file's contents (optionally a line range), returned as cat -n-style numbered text."""
    target = resolve_sandboxed_path(path, must_exist=True)
    if target.is_dir():
        raise ToolError(f"'{path}' is a directory, not a file.")
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolError(f"'{path}' is not a UTF-8 text file; cannot read.")

    lines = text.splitlines()
    total_lines = len(lines)
    start = max(1, start_line) if start_line else 1
    requested_end = min(end_line, total_lines) if end_line else total_lines
    end = min(requested_end, start + MAX_READ_LINES - 1)
    truncated = end < requested_end

    numbered = "\n".join(f"{i:>6}\t{lines[i - 1]}" for i in range(start, end + 1))
    return {"path": path, "start_line": start, "end_line": end,
            "total_lines": total_lines, "truncated": truncated, "content": numbered}


WRITE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Create a new file or overwrite an existing file with the given content, "
            "within the project sandbox. This is a destructive, mutating operation — "
            "the user will be shown a preview and asked to explicitly confirm before "
            "it actually runs; they may decline."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File to write, relative to the project root."},
                "content": {"type": "string", "description": "Full text content to write, replacing any existing content."},
                "create_dirs": {"type": "boolean", "description": "If true, create missing parent directories. Defaults to false."},
            },
            "required": ["path", "content"],
        },
    },
}


def write_file(path: str, content: str, create_dirs: bool = False) -> dict:
    """Create a new file or overwrite an existing one with the given content."""
    target = resolve_sandboxed_path(path)
    size = len(content.encode("utf-8"))
    if size > MAX_WRITE_BYTES:
        raise ToolError(f"Content is {size} bytes, over the {MAX_WRITE_BYTES}-byte write limit.")
    if not target.parent.exists():
        if not create_dirs:
            raise ToolError(f"Parent directory for '{path}' does not exist; pass create_dirs=true to create it.")
        target.parent.mkdir(parents=True, exist_ok=True)
    created_new = not target.exists()
    target.write_text(content, encoding="utf-8")
    return {"path": path, "bytes_written": size, "lines": content.count("\n") + 1, "created_new": created_new}


EDIT_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": (
            "Replace one exact, unique occurrence of old_string with new_string in "
            "an existing file within the project sandbox. old_string must match the "
            "file's current text verbatim (including whitespace) and must be unique "
            "within the file — include enough surrounding context to make it unique "
            "and to make the resulting diff meaningful. The call fails with no changes "
            "made if old_string is not found or is ambiguous. This is a destructive, "
            "mutating operation — the user will be shown the exact diff and asked to "
            "explicitly confirm before it runs; they may decline."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File to edit, relative to the project root."},
                "old_string": {"type": "string", "description": "Exact, unique text currently in the file to replace."},
                "new_string": {"type": "string", "description": "Text to replace old_string with."},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
}


def _validate_edit(path: str, old_string: str, new_string: str) -> tuple[Path, str, str]:
    """Validate an edit_file request and compute the resulting file content without writing it."""
    target = resolve_sandboxed_path(path, must_exist=True)
    if target.is_dir():
        raise ToolError(f"'{path}' is a directory, not a file.")
    try:
        original = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolError(f"'{path}' is not a UTF-8 text file; cannot edit.")

    if old_string == "":
        raise ToolError("old_string must not be empty.")
    if old_string == new_string:
        raise ToolError("old_string and new_string are identical; nothing to change.")

    count = original.count(old_string)
    if count == 0:
        raise ToolError(
            f"old_string not found in '{path}'. No changes made. "
            "Make sure old_string is an exact, verbatim substring of the file, including whitespace/indentation."
        )
    if count > 1:
        raise ToolError(
            f"old_string is ambiguous: it matches {count} locations in '{path}'. "
            "Include more surrounding context to make it unique, then try again. No changes made."
        )
    idx = original.index(old_string)
    new_content = original[:idx] + new_string + original[idx + len(old_string):]
    return target, original, new_content


def edit_file(path: str, old_string: str, new_string: str) -> dict:
    """Replace one exact, unique occurrence of old_string with new_string in a file."""
    target, original, new_content = _validate_edit(path, old_string, new_string)
    target.write_text(new_content, encoding="utf-8")
    diff = "".join(difflib.unified_diff(
        original.splitlines(keepends=True), new_content.splitlines(keepends=True),
        fromfile=path, tofile=path, lineterm="",
    ))
    return {"path": path, "diff": diff}


def preview_edit_file(path: str, old_string: str, new_string: str) -> dict:
    """Validate an edit_file request and return before/after text for display, without writing."""
    _target, original, new_content = _validate_edit(path, old_string, new_string)
    return {"path": path, "old_text": original, "new_text": new_content}


def preview_write_file(path: str, content: str) -> dict:
    """Return the existing file content (if any) and the proposed new content for display."""
    target = resolve_sandboxed_path(path)
    exists = target.exists()
    old_text = None
    if exists:
        try:
            old_text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            old_text = None
    return {"path": path, "old_text": old_text, "new_text": content, "exists": exists}


@dataclass(frozen=True)
class ToolSpec:
    """Bundles a tool's schema, implementation, and safety flags for the registry."""
    name: str
    schema: dict
    func: Callable[..., dict]
    mutating: bool
    requires_confirmation: bool


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "list_files":  ToolSpec("list_files",  LIST_FILES_SCHEMA,  list_files,  mutating=False, requires_confirmation=False),
    "search_file": ToolSpec("search_file", SEARCH_FILE_SCHEMA, search_file, mutating=False, requires_confirmation=False),
    "read_file":   ToolSpec("read_file",   READ_FILE_SCHEMA,   read_file,   mutating=False, requires_confirmation=False),
    "write_file":  ToolSpec("write_file",  WRITE_FILE_SCHEMA,  write_file,  mutating=True,  requires_confirmation=True),
    "edit_file":   ToolSpec("edit_file",   EDIT_FILE_SCHEMA,   edit_file,   mutating=True,  requires_confirmation=True),
}


def tool_schemas() -> list[dict]:
    """Return the list of tool schemas to pass as ollama.chat(tools=...)."""
    return [spec.schema for spec in TOOL_REGISTRY.values()]


def execute_tool(name: str, arguments: dict) -> dict:
    """Run a registered tool by name, catching all errors into a JSON-safe result dict."""
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return {"ok": False, "error": f"Unknown tool '{name}'."}
    try:
        result = spec.func(**arguments)
        result.setdefault("ok", True)
        return result
    except ToolError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"Unexpected error running '{name}': {e}"}
