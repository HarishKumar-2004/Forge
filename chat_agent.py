#!/usr/bin/env python3
"""Terminal chat agent backed by a local Ollama model, with persistent conversation memory."""

import argparse
import difflib
import json
import os
import queue
import re
import select
import shutil
import sys
import textwrap
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import ollama
from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, merge_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.renderer import Renderer
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style

import tools

try:
    import termios  # Unix-only; used to mute the keyboard while the spinner is on screen
except ImportError:
    termios = None

MEMORY_DIR = Path(__file__).parent / "memory"
LOGS_DIR = Path(__file__).parent / "logs"
DEFAULT_MODEL = "qwen3.5:9b"
NUM_CTX = 8192  # model context window (tokens); raise if long sessions/tool schemas run out of generation room
MAX_HISTORY_MESSAGES = 40  # trim to keep requests fast; oldest messages drop first
MAX_TOOL_ROUNDS = 6  # hard cap on tool-call rounds per turn; tools=None past this guarantees termination
LOG_ARG_TRUNCATE_LEN = 500  # cap on argument string length written to the tool-call log
DISPLAY_ARG_TRUNCATE_LEN = 60  # cap on each argument value shown in the tool-call display block

VALID_SESSION_NAME = re.compile(r"^[A-Za-z0-9_-]+$")  # shell-safe: no quoting needed on any OS/shell

BULLET_COLOR = "\033[38;2;111;168;201m"  # soft slate-blue, cool counterpart to the amber "You" label
USER_PROMPT_BG = "\033[48;2;60;60;68m"  # flat neutral grey background for past/submitted user prompts
RECAP_COLOR = "\033[38;2;168;130;201m"  # muted violet, distinct from BULLET_COLOR's slate-blue
RECAP_BULLET = "◆"  # distinct glyph from print_agent_reply's "●"
ITALIC = "\033[3m"

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # pictographs, emoticons, transport, supplemental symbols
    "\U00002600-\U000027BF"  # misc symbols & dingbats
    "\U0001F1E6-\U0001F1FF"  # flag regional indicators
    "\U0000FE0F"            # emoji variation selector
    "\U0000200D"            # zero-width joiner (compound emoji)
    "\U000020E3"            # combining enclosing keycap (e.g. the "1⃣" digit-emoji style)
    "]+",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class SlashCommand:
    name: str   # exact dispatch string used in main(), e.g. "/session"
    usage: str  # completion display text, e.g. "/session [name]"
    help: str   # short description shown in the menu


SLASH_COMMANDS: list[SlashCommand] = [
    SlashCommand("/reset", "/reset", "Clear the active session's memory"),
    SlashCommand("/session", "/session [name]", "Switch or show the active session"),
    SlashCommand("/sessions", "/sessions", "List all sessions"),
    SlashCommand("/rename", "/rename <name>", "Rename the active session"),
    SlashCommand("/model", "/model [name]", "Switch or show the active model"),
    SlashCommand("/models", "/models", "List all installed Ollama models"),
    SlashCommand("/exit", "/exit", "End the session"),
    SlashCommand("/quit", "/quit", "End the session"),
    SlashCommand("/bye", "/bye", "End the session"),
]

EXIT_WORDS = ("/exit", "/quit", "/bye")

MENU_STYLE = Style.from_dict({
    "completion-menu": "bg:#1e2436 #aab0bf",
    "completion-menu.completion": "bg:#1e2436 #aab0bf",
    "completion-menu.completion.current": "bg:#8a7350 #1e2436 noreverse",
    "completion-menu.meta.completion": "bg:#242b40 #8891a3",
    "completion-menu.meta.completion.current": "bg:#7d6a4b #1e2436",
    "prompt.label": "#c9a46a bold",
    "bottom-toolbar": "noreverse",
    "bottom-toolbar.text": "noreverse",
})


class SlashCommandCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for cmd in SLASH_COMMANDS:
            if cmd.name.startswith(text):
                yield Completion(cmd.name, start_position=-len(text),
                                  display=cmd.usage, display_meta=cmd.help)


class DynamicMenuPromptSession(PromptSession):
    """PromptSession that reserves menu height for the current match count, not a fixed maximum."""

    def _get_default_buffer_control_height(self) -> Dimension:
        if self.completer is None or self.complete_style == CompleteStyle.READLINE_LIKE:
            return Dimension()
        max_space = self.reserve_space_for_menu
        if not max_space or get_app().is_done:
            return Dimension()
        buf = self.default_buffer
        if not (buf.complete_while_typing() or buf.complete_state is not None):
            return Dimension()
        # complete_state is None both transiently (completer hasn't run yet this frame) and permanently
        # (it ran and found zero matches, or the menu was dismissed) — on_completions_changed forces a
        # follow-up redraw once the real state is known either way, so there's no need to over-reserve
        # space here on the chance it's the former; Dimension() just means "no extra space yet".
        if buf.complete_state is None:
            return Dimension()
        space = min(len(buf.complete_state.completions), max_space)
        if space == 0:
            return Dimension()
        return Dimension(min=space)


# prompt_toolkit reserves ALL remaining terminal rows below the cursor as a CPR-detected floor on the
# input box's height (by design, so a popup menu never causes a scroll-jump) — harmless in a near-full
# terminal, but a huge empty gap in a tall one with a short banner, and it never decays back down.
# We don't disable CPR outright (some rendering paths gate on the CPR response succeeding) — instead
# we replace the plain _min_available_height attribute with a
# property, computed fresh from the live buffer every time the renderer reads it. A one-shot write (from
# on_text_changed or from DynamicMenuPromptSession, triggered on a text-change/render event) can be stale
# by the time it's actually consumed, since prompt_toolkit's own completer runs asynchronously and the
# height-computation method can be called more than once per render with different completion snapshots.
# Computing on read sidesteps that ordering problem entirely: there is nothing to go stale.
_MIN_AVAILABLE_HEIGHT_BASE = 1
_MENU_ROW_MARGIN = 3  # empirically the minimum slack needed above the raw match count to avoid clipping


def _adaptive_min_available_height(self) -> int:
    """Renderer._min_available_height getter: reserve just enough rows for the menu's current matches.

    complete_state is None both transiently (the async completer hasn't run yet after a keystroke) and
    permanently (the menu was explicitly dismissed, e.g. via Escape) — in both cases there's nothing to
    render yet, so the base height is correct either way; the real size kicks in once completions land.
    """
    try:
        buf = get_app().current_buffer
    except Exception:
        return self.__dict__.get("_min_available_height_stored", _MIN_AVAILABLE_HEIGHT_BASE)
    if not buf.text.startswith("/") or buf.complete_state is None:
        return _MIN_AVAILABLE_HEIGHT_BASE
    return len(buf.complete_state.completions) + _MENU_ROW_MARGIN


def _store_min_available_height(self, value: int) -> None:
    """Renderer._min_available_height setter: keep a fallback value for when no buffer is available."""
    self.__dict__["_min_available_height_stored"] = value


Renderer._min_available_height = property(_adaptive_min_available_height, _store_min_available_height)


def _build_key_bindings() -> KeyBindings:
    """Build custom prompt key bindings: dismiss the menu, Alt+Enter for a newline, Enter to submit (unless blank)."""
    bindings = KeyBindings()

    @bindings.add("escape")
    def _dismiss_menu(event):
        buf = event.current_buffer
        if buf.complete_state:
            buf.cancel_completion()

    @bindings.add("escape", "enter")
    def _insert_newline(event):
        event.current_buffer.insert_text("\n")

    @bindings.add("enter")
    def _submit_unless_blank(event):
        buf = event.current_buffer
        if buf.text.strip():
            buf.validate_and_handle()

    return bindings


@Condition
def _completing_slash_command() -> bool:
    """Only trigger completion (and its reserved menu space) when the buffer looks like a slash command."""
    try:
        return get_app().current_buffer.text.startswith("/")
    except Exception:
        return False


def _request_menu_resize_redraw(*_args) -> None:
    """Clear the cached screen and request a redraw so the input box's height can follow the real menu
    size — needed both on every keystroke and when the async completer resolves (e.g. to zero matches),
    since nothing else prompts a redraw at that point and the old, larger frame would otherwise stick."""
    app = get_app()
    renderer = app.renderer
    if hasattr(renderer, "_last_screen"):
        renderer._last_screen = None
    app.invalidate()


def _on_prompt_text_changed(buf) -> None:
    """Re-trigger completion on delete, then request a resize redraw.

    Buffer._text_changed() always clears complete_state, but only Buffer.insert_text() re-runs the
    completer afterward — backspacing is left with no completions at all unless we restart it here.
    """
    if buf.text.startswith("/"):
        buf.start_completion()
    _request_menu_resize_redraw()


def memory_path(session: str) -> Path:
    MEMORY_DIR.mkdir(exist_ok=True)
    return MEMORY_DIR / f"{session}.json"


def load_history(session: str) -> list[dict]:
    path = memory_path(session)
    if not path.exists():
        return []
    with path.open("r") as f:
        return json.load(f)


def save_history(session: str, history: list[dict]) -> None:
    with memory_path(session).open("w") as f:
        json.dump(history, f, indent=2)


def trim(history: list[dict]) -> list[dict]:
    if len(history) <= MAX_HISTORY_MESSAGES:
        return history
    return history[-MAX_HISTORY_MESSAGES:]


def separator() -> str:
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    return f"\033[2m{'─' * width}\033[0m"


def print_session_banner(session: str) -> None:
    """Print a full-width, dim separator line with the session name centered in it — a clear visual
    boundary marking a session switch. Needed because switching to a session already replayed this
    run shows no replay content at all, leaving the one-line "Switched to session..." text as the
    only sign anything changed, easy to miss once you've scrolled past it."""
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    label = f" Session: {session} "
    if len(label) >= width:
        print(f"\033[2m{label[:width]}\033[0m")
        return
    pad = width - len(label)
    left = pad // 2
    right = pad - left
    print(f"\033[2m{'─' * left}{label}{'─' * right}\033[0m")


def print_user_prompt_block(session: str, text: str) -> None:
    """Print a submitted/past user prompt as a solid, full-width grey bar per line — bold text, no
    other foreground color, no separator rules touching it. Matches Claude Code's own CLI styling for
    past input. Used both for history replay and for a just-submitted live prompt once prompt_toolkit's
    own rendering has been erased (erase_when_done=True)."""
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    label = f"You[{session}]> "
    lines = text.split("\n")
    content_lines = [label + lines[0]] + [" " * len(label) + line for line in lines[1:]]
    for line in content_lines:
        line = line[:width]
        print(f"{USER_PROMPT_BG}\033[1m{line}{' ' * (width - len(line))}\033[0m")
    print()


def is_valid_session_name(name: str) -> bool:
    """Check that a session name is safe to use as a filename and a shell argument without quoting."""
    return bool(VALID_SESSION_NAME.match(name))


def session_names(active: str | None = None) -> list[str]:
    """Return all session names with a memory file, plus the active one even if it has nothing saved yet."""
    names = {p.stem for p in MEMORY_DIR.glob("*.json")} if MEMORY_DIR.exists() else set()
    if active is not None:
        names.add(active)
    return sorted(names)


def installed_models() -> list[str] | None:
    """Return locally installed model names, or None if the Ollama server can't be reached."""
    try:
        return sorted(m.model for m in ollama.list().models)
    except Exception:
        return None


def to_model_message(entry: dict) -> dict:
    """Strip the local 'ts' timestamp field so a stored history entry can be sent to Ollama."""
    return {k: v for k, v in entry.items() if k != "ts"}


def _truncate_for_log(value, limit: int = LOG_ARG_TRUNCATE_LEN):
    """Shorten long strings (recursively through dicts) before writing them to the tool-call log."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"...({len(value)} chars total)"
    if isinstance(value, dict):
        return {k: _truncate_for_log(v, limit) for k, v in value.items()}
    return value


def log_tool_call(session: str, tool: str, arguments: dict, mutating: bool, approval: str,
                   outcome: str, error: str | None, duration_ms: int, round_num: int) -> None:
    """Append one structured JSON record of a tool call to logs/tool_calls.jsonl."""
    LOGS_DIR.mkdir(exist_ok=True)
    entry = {
        "ts": datetime.now().isoformat(),
        "session": session,
        "tool": tool,
        "arguments": _truncate_for_log(dict(arguments)),
        "mutating": mutating,
        "approval": approval,
        "outcome": outcome,
        "error": error,
        "duration_ms": duration_ms,
        "round": round_num,
    }
    with (LOGS_DIR / "tool_calls.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")


def render_diff(old_text: str | None, new_text: str, context: int = 3) -> str:
    """Render a colored, line-numbered diff between old_text and new_text for terminal display."""
    new_lines = new_text.splitlines()
    if old_text is None:
        return "\n".join(f"\033[32m{i:>5} + {line}\033[0m" for i, line in enumerate(new_lines, start=1))

    old_lines = old_text.splitlines()
    opcodes = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False).get_opcodes()
    out = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            idxs = list(range(i1, i2))
            if len(idxs) <= 2 * context:
                head, tail = idxs, []
            else:
                head, tail = idxs[:context], idxs[-context:]
            for k in head:
                out.append(f"\033[2m{k + 1:>5}   {old_lines[k]}\033[0m")
            if tail:
                out.append("\033[2m      ⋮\033[0m")
                for k in tail:
                    out.append(f"\033[2m{k + 1:>5}   {old_lines[k]}\033[0m")
        else:
            if tag in ("delete", "replace"):
                for k in range(i1, i2):
                    out.append(f"\033[31m{k + 1:>5} - {old_lines[k]}\033[0m")
            if tag in ("insert", "replace"):
                for k in range(j1, j2):
                    out.append(f"\033[32m{k + 1:>5} + {new_lines[k]}\033[0m")
    return "\n".join(out)


def _apply_inline_markup(line: str) -> str:
    """Convert inline **bold** and `code` markdown spans into ANSI-styled text."""
    line = re.sub(r"(?<=\S) {2,}(?=\S)", " ", line)
    line = re.sub(r"\*\*(.+?)\*\*", lambda m: f"\033[1m{m.group(1).strip()}\033[0m", line)
    line = re.sub(r"__(.+?)__", lambda m: f"\033[1m{m.group(1).strip()}\033[0m", line)
    line = re.sub(r"`([^`]+)`", lambda m: f"\033[2m{m.group(1).strip()}\033[0m", line)
    return line


def _strip_markup_only(text: str) -> str:
    """Remove nested markdown emphasis/code markers without adding ANSI codes (for use inside headings)."""
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: m.group(1).strip(), text)
    text = re.sub(r"__(.+?)__", lambda m: m.group(1).strip(), text)
    text = re.sub(r"`([^`]+)`", lambda m: m.group(1).strip(), text)
    return text


def _strip_heading(line: str) -> tuple[str, bool]:
    """Strip a markdown '#' heading prefix and bold the heading text, if this line is one."""
    m = re.match(r"^(#{1,6})\s+(.*)", line)
    if m:
        return f"\033[1m{_strip_markup_only(m.group(2))}\033[0m", True
    return line, False


def _normalize_bullet(line: str) -> tuple[str, bool]:
    """Normalize a markdown '-'/'*'/'+' bullet line into a consistent indented bullet, if it is one."""
    m = re.match(r"^\s*[-*+]\s+(.*)", line)
    if m:
        return f"  • {m.group(1)}", True
    return line, False


def _normalize_numbered(line: str) -> tuple[str, bool]:
    """Normalize a markdown numbered-list line into a consistently indented form, if it is one."""
    m = re.match(r"^\s*(\d+)[.)]\s+(.*)", line)
    if m:
        return f"  {m.group(1)}. {m.group(2)}", True
    return line, False


def _format_reply_line(line: str, in_code_block: bool) -> tuple[str | None, bool]:
    """Apply heading/bullet/numbered/code-fence/inline-markup formatting to one already-rstrip()'d line
    of raw reply text. Returns (formatted_line, new_in_code_block_state) — formatted_line is None for a
    fence-toggle (```) line, which emits nothing. Shared by format_agent_reply (whole-text, static
    replies) and the streaming reply path (one line at a time, as text arrives) so both stay in sync."""
    if line.strip().startswith("```"):
        return None, not in_code_block
    if in_code_block:
        return f"    \033[2m{line}\033[0m", in_code_block

    heading, is_heading = _strip_heading(line)
    if is_heading:
        return heading, in_code_block
    bulleted, is_bullet = _normalize_bullet(line)
    if is_bullet:
        return _apply_inline_markup(bulleted), in_code_block
    numbered, is_numbered = _normalize_numbered(line)
    if is_numbered:
        return _apply_inline_markup(numbered), in_code_block
    return _apply_inline_markup(line), in_code_block


def format_agent_reply(text: str) -> str:
    """Strip emoji and normalize common markdown markup into clean, ANSI-styled terminal text."""
    text = EMOJI_PATTERN.sub("", text)
    out_lines = []
    in_code_block = False
    for line in text.split("\n"):
        formatted, in_code_block = _format_reply_line(line.rstrip(), in_code_block)
        if formatted is not None:
            out_lines.append(formatted)

    cleaned = []
    blank_run = 0
    for line in out_lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        cleaned.append(line)
    return "\n".join(cleaned)


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _wrap_ansi_line(line: str, width: int) -> list[str]:
    """Word-wrap a single line to `width` *visible* columns, correctly ignoring embedded ANSI escape
    codes when measuring width (so styled text like bold/dim spans doesn't wrap too early) while
    keeping each code attached to the text it colors across the cut. Never breaks mid-word — without
    this, long reply/recap lines just hit the terminal's own raw-column auto-wrap and can split a word
    across two lines with no indent on the continuation, which is hard to read."""
    visible, raw_positions = [], []
    i = 0
    while i < len(line):
        m = _ANSI_RE.match(line, i)
        if m:
            i = m.end()
            continue
        visible.append(line[i])
        raw_positions.append(i)
        i += 1
    visible_text = "".join(visible)
    if len(visible_text) <= width:
        return [line]
    wrapped = textwrap.wrap(visible_text, width=width, break_long_words=False, break_on_hyphens=False)
    if not wrapped:
        return [line]
    out = []
    pos = 0
    for piece in wrapped:
        idx = visible_text.index(piece, pos)
        raw_start = raw_positions[idx] if idx < len(raw_positions) else len(line)
        end_visible_idx = idx + len(piece) - 1
        raw_end = raw_positions[end_visible_idx] + 1 if end_visible_idx < len(raw_positions) else len(line)
        out.append(line[raw_start:raw_end])
        pos = idx + len(piece)
    return out


def _wrap_for_display(text: str, indent_width: int) -> list[str]:
    """Split `text` on explicit newlines, then word-wrap each resulting line to the terminal width
    minus `indent_width` (the space taken by the bullet/indent prefix each line gets when printed)."""
    width = max(shutil.get_terminal_size(fallback=(80, 24)).columns - indent_width, 20)
    out = []
    for raw_line in text.split("\n"):
        out.extend(_wrap_ansi_line(raw_line, width))
    return out


def print_agent_reply(reply: str, empty_message: str = "model returned an empty response — try again or rephrase") -> None:
    """Print the agent's reply as an indented block prefixed with a round bullet marker. `empty_message`
    lets callers give a more accurate placeholder for the empty-reply case — e.g. distinguishing "the
    model genuinely failed to respond" from "tool calls succeeded but the model gave no closing summary",
    which look identical (empty content) but mean very different things to the user."""
    formatted = format_agent_reply(reply)
    if not formatted.strip():
        print(f"{BULLET_COLOR}●\033[0m \033[2m({empty_message})\033[0m\n")
        return
    lines = _wrap_for_display(formatted, indent_width=2)
    print(f"{BULLET_COLOR}●\033[0m {lines[0]}")
    for line in lines[1:]:
        print(f"  {line}")
    print()


RECAP_INSTRUCTION = (
    "Summarize the conversation above in 1-2 short sentences for someone resuming it. "
    "State what was discussed and any open thread; do not greet the user or ask a question."
)


def generate_recap(model: str, history: list[dict]) -> str | None:
    """Ask the model for a 1-2 line recap of `history`; return None on empty history or any failure
    (unreachable Ollama, bad response) so callers can skip printing it rather than crash.

    Calls ollama.chat() directly rather than via call_with_spinner: this runs in a background thread
    that may well be active *while* the user is already typing in the live prompt box, and
    call_with_spinner's raw termios manipulation + direct \\r-based spinner printing both assume
    exclusive control of the terminal — running them concurrently with prompt_toolkit's own live
    Application rendering corrupts the display. A silent background call has no such conflict."""
    if not history:
        return None
    messages = [to_model_message(e) for e in history] + [{"role": "user", "content": RECAP_INSTRUCTION}]
    try:
        response = ollama.chat(model=model, messages=messages, tools=None, options={"num_ctx": NUM_CTX})
    except Exception:
        return None
    text = (response.get("message") or {}).get("content", "").strip()
    return text or None


def print_recap(text: str) -> None:
    """Print a model-generated resume recap: violet '◆' bullet, italic text — visually distinct from
    print_agent_reply's slate-blue '●'. Skips format_agent_reply's markdown pass deliberately (its
    \033[0m resets would cut the outer italic short); recap text is 1-2 plain sentences, so only
    emoji-stripping is applied."""
    cleaned = EMOJI_PATTERN.sub("", text).strip()
    # Reserve margin for the widest prefix actually printed ("◆ recap: ", 9 visible chars), not just
    # the 2-char continuation indent — otherwise the first line can still overflow the terminal width
    # and get wrapped a second time by the terminal itself, splitting a word across the two pieces.
    lines = _wrap_for_display(cleaned, indent_width=len("◆ recap: "))
    print(f"{RECAP_COLOR}{RECAP_BULLET}\033[0m recap: {ITALIC}{lines[0]}\033[0m")
    for line in lines[1:]:
        print(f"  {ITALIC}{line}\033[0m")
    print()


def format_tool_call(name: str, arguments: dict) -> str:
    """Render a tool call as a one-line 'name(arg=value, ...)' signature with long values shortened."""
    parts = []
    for key, value in arguments.items():
        if isinstance(value, str):
            flat = value.replace("\n", "\\n")
            if len(flat) > DISPLAY_ARG_TRUNCATE_LEN:
                flat = flat[:DISPLAY_ARG_TRUNCATE_LEN] + f"… ({len(value)} chars)"
            parts.append(f'{key}="{flat}"')
        else:
            parts.append(f"{key}={value}")
    return f"\033[1m{name}\033[0m\033[2m({', '.join(parts)})\033[0m"


def _print_tool_block_header(round_num: int, count: int) -> None:
    """Print the header line that opens the display block for one round of tool calls."""
    noun = "tool call" if count == 1 else "tool calls"
    print(f"{BULLET_COLOR}●\033[0m \033[1mTools\033[0m \033[2m(round {round_num}, {count} {noun})\033[0m")


def _print_tool_status(result: dict, duration_ms: int) -> None:
    """Print the indented outcome line that sits under a tool call's bullet."""
    if result.get("declined"):
        print("    \033[33m⊘ declined by user\033[0m")
    elif result.get("ok"):
        print(f"    \033[32m✓\033[0m \033[2mdone ({duration_ms}ms)\033[0m")
    else:
        error = str(result.get("error", "unknown error")).replace("\n", " ")
        if len(error) > 200:
            error = error[:200] + "…"
        print(f"    \033[31m✗ {error}\033[0m")


def _summarize_mutation(old_text: str | None, new_text: str) -> str:
    """Build a short, human-readable summary of what a pending write_file/edit_file change will do —
    new file vs. overwrite, line counts, lines added/removed, byte size."""
    new_lines = new_text.splitlines()
    size = len(new_text.encode("utf-8"))
    if old_text is None:
        return f"create new file · {len(new_lines)} line{'s' if len(new_lines) != 1 else ''} · {size:,} bytes"
    old_lines = old_text.splitlines()
    opcodes = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False).get_opcodes()
    added = sum(j2 - j1 for tag, i1, i2, j1, j2 in opcodes if tag in ("insert", "replace"))
    removed = sum(i2 - i1 for tag, i1, i2, j1, j2 in opcodes if tag in ("delete", "replace"))
    return f"overwrite existing file · {len(old_lines)} → {len(new_lines)} lines · +{added}/-{removed}"


def _print_streaming(text: str, tick: float = 0.02, chars_per_tick: int = 8) -> None:
    """Print `text` (already fully rendered — e.g. a complete diff) at a smooth, evenly-paced reveal
    rate instead of dumping it all at once, for the same reason call_with_streaming_reply paces agent
    replies — a large write_file/edit_file diff otherwise appears instantly, scrolling the terminal
    straight to its end. No background thread/queue needed here (unlike the agent-reply case): the
    text is already fully computed locally, there's no network stream to decouple pacing from. Esc or
    Ctrl+C fast-forwards the rest of the text instantly rather than cancelling anything — there's no
    "decline" semantic for a preview; the y/N prompt right after it is the real decision point."""
    can_poll_esc = termios is not None and sys.stdin.isatty()
    skip = False
    with _muted_keyboard():
        print("\033[?25l", end="", flush=True)
        try:
            pos = 0
            while pos < len(text):
                if not skip:
                    if can_poll_esc:
                        ready, _, _ = select.select([sys.stdin], [], [], tick)
                        if ready and os.read(sys.stdin.fileno(), 1) == b"\x1b":
                            skip = True
                    else:
                        time.sleep(tick)
                step = len(text) if skip else chars_per_tick
                print(text[pos:pos + step], end="", flush=True)
                pos += step
        finally:
            print("\033[?25h", end="", flush=True)
    print()


def _print_mutation_preview(name: str, path: str, old_text: str | None, new_text: str) -> None:
    """Print a colored, line-numbered diff preview for a pending write_file/edit_file call."""
    summary = _summarize_mutation(old_text, new_text)
    print(f"\n{BULLET_COLOR}●\033[0m \033[1mProposed {name}: {path}\033[0m \033[2m({summary})\033[0m")
    _print_streaming(render_diff(old_text, new_text))
    print()


def _ask_confirmation(prompt_text: str) -> bool:
    """Prompt the user for y/n approval, auto-declining when there's no real interactive terminal."""
    if not sys.stdin.isatty():
        print(f"{prompt_text}[auto-declined: non-interactive session]\n")
        return False
    try:
        answer = input(prompt_text).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def run_tool_with_confirmation(session: str, name: str, arguments: dict, round_num: int) -> dict:
    """Execute a tool call, inserting a diff preview and y/n confirmation first if it mutates files."""
    spec = tools.TOOL_REGISTRY.get(name)
    start = time.time()
    print(f"  • {format_tool_call(name, arguments)}")

    if spec is None or not spec.requires_confirmation:
        result = tools.execute_tool(name, arguments)
        duration_ms = int((time.time() - start) * 1000)
        _print_tool_status(result, duration_ms)
        log_tool_call(session, name, arguments, mutating=False, approval="auto",
                      outcome="ok" if result.get("ok") else "error",
                      error=result.get("error"), duration_ms=duration_ms, round_num=round_num)
        return result

    path = arguments.get("path", "?")
    try:
        if name == "edit_file":
            preview = tools.preview_edit_file(**arguments)
        else:
            preview = tools.preview_write_file(path=path, content=arguments.get("content", ""))
    except tools.ToolError as e:
        duration_ms = int((time.time() - start) * 1000)
        log_tool_call(session, name, arguments, mutating=True, approval="auto",
                       outcome="validation_error", error=str(e), duration_ms=duration_ms, round_num=round_num)
        result = {"ok": False, "error": str(e)}
        _print_tool_status(result, duration_ms)
        return result

    _print_mutation_preview(name, path, preview.get("old_text"), preview["new_text"])
    approved = _ask_confirmation(
        f"{BULLET_COLOR}●\033[0m Apply this \033[1m{name}\033[0m to \033[1m{path}\033[0m? \033[2m[y/N]\033[0m: "
    )

    if not approved:
        log_tool_call(session, name, arguments, mutating=True, approval="user_declined",
                       outcome="declined", error=None, duration_ms=0, round_num=round_num)
        result = {"ok": False, "declined": True,
                  "error": "The user declined to approve this action. Do not attempt it again unless asked."}
        _print_tool_status(result, 0)
        return result

    exec_start = time.time()
    result = tools.execute_tool(name, arguments)
    duration_ms = int((time.time() - exec_start) * 1000)
    _print_tool_status(result, duration_ms)
    log_tool_call(session, name, arguments, mutating=True, approval="user_approved",
                  outcome="ok" if result.get("ok") else "error",
                  error=result.get("error"), duration_ms=duration_ms, round_num=round_num)
    return result


@contextmanager
def _muted_keyboard():
    """Turn off terminal echo while the spinner runs so stray keys (e.g. Enter) can't break its line."""
    if termios is None or not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    muted = termios.tcgetattr(fd)
    muted[3] &= ~(termios.ECHO | termios.ICANON)  # ISIG stays on, so Ctrl+C still cancels
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, muted)
        yield
    finally:
        termios.tcflush(fd, termios.TCIFLUSH)  # drop anything typed meanwhile so it doesn't leak into the next prompt
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def call_with_spinner(model: str, messages: list[dict], tool_schemas: list[dict] | None) -> dict:
    """Run ollama.chat() in a background thread while printing a live 'Thinking... (Ns)' timer; Esc or Ctrl+C cancels."""
    result: dict = {}

    def target() -> None:
        try:
            result["response"] = ollama.chat(model=model, messages=messages, tools=tool_schemas,
                                              options={"num_ctx": NUM_CTX})
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    start = time.time()
    thread.start()
    can_poll_esc = termios is not None and sys.stdin.isatty()
    with _muted_keyboard():
        print("\033[?25l", end="", flush=True)  # hide the cursor; it would otherwise sit visibly at end of line
        try:
            while thread.is_alive():
                elapsed = int(time.time() - start)
                print(f"\rThinking... ({elapsed}s)  \033[2m· esc to interrupt\033[0m\033[K", end="", flush=True)
                if can_poll_esc:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if ready and os.read(sys.stdin.fileno(), 1) == b"\x1b":
                        raise KeyboardInterrupt
                else:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\r\033[K", end="", flush=True)
            raise
        finally:
            print("\033[?25h", end="", flush=True)
        print("\r\033[K", end="", flush=True)

    if "error" in result:
        raise result["error"]
    return result["response"]


def call_with_streaming_reply(model: str, messages: list[dict], tool_schemas: list[dict] | None) -> dict:
    """Stream ollama.chat()'s reply, revealing text at a smooth, evenly-paced rate decoupled from the
    model's own (bursty) generation timing — instead of waiting for the full reply and dumping it all
    at once, which on a long reply leaves only the tail end visible, forcing a scroll-up to read from
    the top. A tool-calling round naturally shows nothing here (its chunks carry empty content), so the
    existing 'Thinking...' spinner runs the whole time and the caller's tool-call display flow is
    completely unaffected; this only changes how a genuine final-content reply gets displayed. Esc or
    Ctrl+C cancels, matching call_with_spinner's behavior."""
    display_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()

    def produce() -> None:
        raw_parts: list[str] = []
        in_code_block = False
        line_buf = ""
        held_blank: str | None = None
        first_line_emitted = False
        tool_calls = None
        role = "assistant"

        def emit(raw_line: str) -> None:
            nonlocal in_code_block, held_blank, first_line_emitted
            formatted, in_code_block = _format_reply_line(raw_line, in_code_block)
            if formatted is None:
                return  # fence-toggle line — nothing to display
            if formatted.strip() == "":
                if held_blank is None:  # collapse runs of blanks down to one, like format_agent_reply
                    held_blank = formatted
                return
            if held_blank is not None:
                display_queue.put(("line", held_blank))
                held_blank = None
            width = max(shutil.get_terminal_size(fallback=(80, 24)).columns - 2, 20)
            for wrapped in _wrap_ansi_line(formatted, width):
                prefix = f"{BULLET_COLOR}●\033[0m " if not first_line_emitted else "  "
                first_line_emitted = True
                display_queue.put(("line", prefix + wrapped))

        try:
            for chunk in ollama.chat(model=model, messages=messages, tools=tool_schemas,
                                      stream=True, options={"num_ctx": NUM_CTX}):
                role = chunk.message.role or role
                if chunk.message.tool_calls:
                    tool_calls = chunk.message.tool_calls
                delta = chunk.message.content or ""
                if delta:
                    raw_parts.append(delta)
                    line_buf += delta
                    while "\n" in line_buf:
                        raw_line, line_buf = line_buf.split("\n", 1)
                        emit(raw_line.rstrip())
            if line_buf:  # final line never newline-terminated by the model
                emit(line_buf.rstrip())
            if held_blank is not None:
                display_queue.put(("line", held_blank))
            display_queue.put(("done", {"content": "".join(raw_parts), "tool_calls": tool_calls, "role": role}))
        except Exception as e:
            display_queue.put(("error", e))

    thread = threading.Thread(target=produce, daemon=True)
    start = time.time()
    thread.start()
    can_poll_esc = termios is not None and sys.stdin.isatty()
    final_payload: dict = {}

    with _muted_keyboard():
        print("\033[?25l", end="", flush=True)  # hide the cursor; it would otherwise sit visibly at end of line
        in_spinner_phase = True
        mid_line = False
        try:
            # Phase 1: spinner until the first queued item of any kind arrives (mirrors call_with_spinner).
            pending_item = None
            while pending_item is None:
                try:
                    pending_item = display_queue.get(timeout=0.1)
                except queue.Empty:
                    elapsed = int(time.time() - start)
                    print(f"\rThinking... ({elapsed}s)  \033[2m· esc to interrupt\033[0m\033[K", end="", flush=True)
                    if can_poll_esc:
                        ready, _, _ = select.select([sys.stdin], [], [], 0)
                        if ready and os.read(sys.stdin.fileno(), 1) == b"\x1b":
                            raise KeyboardInterrupt
            print("\r\033[K", end="", flush=True)
            in_spinner_phase = False

            # Phase 2: paced reveal — ~150 chars/sec, decoupled from raw chunk arrival timing. The same
            # select.select() call paces each tick AND polls for Esc, exactly like call_with_spinner.
            tick = 0.001
            chars_per_tick = 1
            pos = 0
            while True:
                if can_poll_esc:
                    ready, _, _ = select.select([sys.stdin], [], [], tick)
                    if ready and os.read(sys.stdin.fileno(), 1) == b"\x1b":
                        raise KeyboardInterrupt
                else:
                    time.sleep(tick)

                if pending_item is None:
                    try:
                        pending_item = display_queue.get_nowait()
                        pos = 0
                    except queue.Empty:
                        continue  # model slower than display right now — idle this tick, not an error

                tag, payload = pending_item
                if tag == "error":
                    raise payload
                if tag == "done":
                    final_payload = payload
                    break

                line = payload
                print(line[pos:pos + chars_per_tick], end="", flush=True)
                mid_line = True
                pos += chars_per_tick
                if pos >= len(line):
                    print()
                    mid_line = False
                    pending_item = None
        except KeyboardInterrupt:
            if in_spinner_phase:
                print("\r\033[K", end="", flush=True)
            elif mid_line:
                print()  # don't glue '[cancelled]' onto the cut-off text
            raise
        finally:
            print("\033[?25h", end="", flush=True)

    msg = ollama.Message(role=final_payload.get("role", "assistant"),
                          content=final_payload.get("content") or None,
                          tool_calls=final_payload.get("tool_calls"))
    return {"message": msg}


def replay_history(session: str, history: list[dict]) -> None:
    """Print the full past conversation for `session` — user turns as grey bars, each followed by
    whatever that turn actually produced. Pure-bookkeeping entries (tool results, and intermediate
    assistant entries that only carry tool_calls) stay skipped, same as before — but every user turn
    now always gets SOME assistant-side line, mirroring exactly what main()'s live round loop shows:
    the real final content if there is any, else a "no closing summary, but the action(s) completed"
    placeholder if tool calls happened that turn with no final text, else the generic empty-response
    placeholder. Previously a turn whose last round had no content (e.g. a tool call with no closing
    summary from the model) silently vanished on replay — the user prompt showed with nothing after
    it, looking like the request had been dropped, even though the tool call itself had succeeded.
    Shared by startup resume and in-chat '/session' switch. No-ops on empty history. Does NOT generate
    the recap — see _recap_worker, which runs in the background so this doesn't block the live prompt
    box from appearing."""
    i, n = 0, len(history)
    while i < n:
        entry = history[i]
        if entry.get("role") != "user":
            i += 1  # orphaned assistant/tool entry with no preceding user turn — shouldn't normally happen
            continue

        print_user_prompt_block(session, entry["content"])
        i += 1

        any_tool_calls = False
        last_assistant_content = None
        while i < n and history[i].get("role") in ("assistant", "tool"):
            e = history[i]
            if e.get("role") == "assistant":
                if e.get("tool_calls"):
                    any_tool_calls = True
                last_assistant_content = e.get("content")  # overwritten each round; ends up being the LAST one
            i += 1

        if last_assistant_content:
            print_agent_reply(last_assistant_content)
        elif any_tool_calls:
            print_agent_reply("", empty_message="no closing summary from the model, "
                                                 "but the action(s) above completed")
        else:
            print_agent_reply("")


def _recap_worker(model: str, history: list[dict], result_queue: "queue.Queue[str]") -> None:
    """Background-thread target: generate the resume recap (a blocking ollama.chat() call) and put it
    on `result_queue` once ready. Deliberately does NOT print — the main loop drains the queue and
    prints it at the top of each iteration, between prompt() calls, where a plain print() is always
    safe since nothing else is being rendered at that moment. Live mid-air injection while the prompt
    box is up (via patch_stdout, then via run_in_terminal from a pre_run-started background task) was
    tried and worked technically, but visually it looked like the recap was intruding on the live box
    rather than reading as part of the conversation — simpler and calmer to just let it land on the
    next natural redraw instead."""
    recap = generate_recap(model, history)
    if recap is not None:
        result_queue.put(recap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--resume", metavar="NAME", default=None,
                        help="Resume an existing session by name (default: the 'default' session); "
                             "start new sessions with '/session <name>' inside the chat")
    parser.add_argument("--reset", action="store_true", help="Clear this session's memory before starting")
    args = parser.parse_args()

    session = args.resume if args.resume is not None else "default"
    if args.resume is not None:
        if not is_valid_session_name(session):
            print(f"\nSession name '{session}' is invalid — use only letters, numbers, '-', and '_'")
            print("(this keeps session names safe to pass on the command line without quoting).\n")
            sys.exit(1)
        if not memory_path(session).exists():
            existing = session_names()
            print(f"\nNo session named '{session}' to resume.")
            print(f"Existing sessions: {', '.join(existing) if existing else '(none yet)'}")
            print("Start a new one with '/session <name>' inside the chat.\n")
            sys.exit(1)

    models = installed_models()
    if models is not None and args.model not in models:
        print(f"\nModel '{args.model}' is not installed locally.")
        print(f"Installed models: {', '.join(models) if models else '(none — pull one first)'}")
        print(f"Pull it with: ollama pull {args.model}\n")
        sys.exit(1)

    if args.reset:
        memory_path(session).unlink(missing_ok=True)
        print(f"Cleared memory for session '{session}'.")

    model = args.model
    history = load_history(session)
    print(f"\n{BULLET_COLOR}●\033[0m Chatting with {model} (session: {session}, {len(history)} messages in memory)")
    print("   - Type '/exit', '/quit', or '/bye' to stop;")
    print("   - '/reset' wipes the active session; '/session [name]' switches or shows the active session; '/sessions' lists them;")
    print("   - '/rename <name>' renames the active session;")
    print("   - '/model [name]' switches or shows the active model; '/models' lists installed models;\n")

    replayed_sessions = {session}
    replay_history(session, history)

    prompt_session = (
        DynamicMenuPromptSession(completer=SlashCommandCompleter(),
                      complete_while_typing=_completing_slash_command,
                      key_bindings=_build_key_bindings(),
                      style=MENU_STYLE,
                      bottom_toolbar=lambda: ANSI(separator()),
                      reserve_space_for_menu=len(SLASH_COMMANDS),
                      erase_when_done=True)
        if sys.stdin.isatty() else None
    )
    if prompt_session is not None:
        prompt_session.default_buffer.on_text_changed += _on_prompt_text_changed
        prompt_session.default_buffer.on_completions_changed += _request_menu_resize_redraw

    pending_default = ""
    pending_recap: "queue.Queue[str]" = queue.Queue()
    if history:
        # Pass a snapshot, not the live list — the main thread may mutate/reassign `history` (e.g. if
        # the user sends another message) while this background thread is still iterating it.
        threading.Thread(target=_recap_worker, args=(model, list(history), pending_recap), daemon=True).start()

    while True:
        # Drain any ready recap here, between prompt() calls — nothing else is being rendered at this
        # point, so a plain print() is always safe, and it reads as a normal, sequential part of the
        # conversation rather than intruding on the live box.
        while True:
            try:
                print_recap(pending_recap.get_nowait())
            except queue.Empty:
                break
        try:
            if prompt_session is not None:
                # The boundary separator is rendered as part of the prompt's own first line (not a
                # separate print()), so erase_when_done wipes it along with the rest of the box on
                # submit — nothing is left behind once the grey block takes over that screen position,
                # matching how replayed history has no separators between entries either.
                prompt_fragments = merge_formatted_text([
                    ANSI(separator() + "\n"),
                    [("class:prompt.label", f"You[{session}]"), ("", "> ")],
                ])
                user_input = prompt_session.prompt(prompt_fragments, default=pending_default).strip()
            else:
                print(separator())
                user_input = input(f"You[{session}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        pending_default = ""

        if not user_input:
            continue

        # Any non-empty submission — chat message or slash command alike — gets the grey block, so
        # there's always a visible record of what you typed once it's no longer the live input. Only
        # a genuine chat message additionally gets persisted to history (and so reappears on replay);
        # slash commands don't persist, but still deserve to be visible in the moment they're typed.
        print_user_prompt_block(session, user_input)

        if user_input.lower() in EXIT_WORDS:
            print(separator())
            print(f"{BULLET_COLOR}●\033[0m Goodbye! See you next time.")
            print(separator())
            break
        if user_input == "/reset":
            history = []
            save_history(session, history)
            print(f"{BULLET_COLOR}●\033[0m Memory cleared.\n")
            continue
        if user_input == "/sessions":
            names = [f"{n} (active)" if n == session else n for n in session_names(active=session)]
            print(f"{BULLET_COLOR}●\033[0m Sessions: {', '.join(names)}\n")
            continue
        if user_input == "/rename" or user_input.startswith("/rename "):
            new_name = user_input[len("/rename"):].strip()
            if not new_name:
                print(f"{BULLET_COLOR}●\033[0m Usage: /rename <new_session_name>\n")
            elif not is_valid_session_name(new_name):
                print(f"{BULLET_COLOR}●\033[0m Session name '{new_name}' is invalid — use only letters, numbers, '-', and '_'.\n")
            elif new_name == session:
                print(f"{BULLET_COLOR}●\033[0m Session is already named '{session}'.\n")
            elif memory_path(new_name).exists():
                print(f"{BULLET_COLOR}●\033[0m A session named '{new_name}' already exists — pick a different name.\n")
            else:
                if memory_path(session).exists():
                    memory_path(session).rename(memory_path(new_name))
                print(f"{BULLET_COLOR}●\033[0m Renamed session '{session}' to '{new_name}'.\n")
                session = new_name
            continue
        if user_input == "/session" or user_input.startswith("/session "):
            new_session = user_input[len("/session"):].strip()
            if not new_session:
                print(f"{BULLET_COLOR}●\033[0m Current session: '{session}' ({len(history)} messages)\n")
            elif not is_valid_session_name(new_session):
                print(f"{BULLET_COLOR}●\033[0m Session name '{new_session}' is invalid — use only letters, numbers, '-', and '_'.\n")
            else:
                session = new_session
                history = load_history(session)
                print(f"{BULLET_COLOR}●\033[0m Switched to session '{session}' ({len(history)} messages in memory)\n")
                print_session_banner(session)
                print()
                if session not in replayed_sessions:
                    replay_history(session, history)
                    if history:
                        threading.Thread(target=_recap_worker, args=(model, list(history), pending_recap),
                                          daemon=True).start()
                    replayed_sessions.add(session)
            continue
        if user_input == "/models":
            available = installed_models()
            if available is None:
                print(f"{BULLET_COLOR}●\033[0m Could not reach the Ollama server to list models.\n")
            else:
                print(f"{BULLET_COLOR}●\033[0m Available Models: {', '.join(available) if available else '(none installed)'}\n")
            continue
        if user_input == "/model" or user_input.startswith("/model "):
            new_model = user_input[len("/model"):].strip()
            if not new_model:
                print(f"{BULLET_COLOR}●\033[0m Current model: '{model}'\n")
            else:
                available = installed_models()
                if available is not None and new_model not in available:
                    print(f"{BULLET_COLOR}●\033[0m Model '{new_model}' is not installed locally.")
                    print(f"  - Installed models: {', '.join(available) if available else '(none — pull one first)'}")
                    print(f"  - Pull it with: ollama pull {new_model}\n")
                else:
                    model = new_model
                    print(f"{BULLET_COLOR}●\033[0m Switched to model '{model}'.\n")
            continue

        history_before_turn = list(history)  # snapshot so an Esc/Ctrl+C interrupt can roll this whole turn back
        history.append({"role": "user", "content": user_input, "ts": datetime.now().isoformat()})
        history = trim(history)
        save_history(session, history)

        round_num = 0
        final_reply = None
        interrupted = False
        errored = False
        any_tool_calls = False

        while True:
            round_num += 1
            force_final = round_num > MAX_TOOL_ROUNDS
            messages = [to_model_message(e) for e in history]
            try:
                response = call_with_streaming_reply(model, messages, None if force_final else tools.tool_schemas())
            except KeyboardInterrupt:
                print("[cancelled]\n")
                interrupted = True
                break
            except Exception as e:
                print(f"[error talking to ollama: {e}]")
                errored = True
                break

            msg = response["message"]
            msg_entry = msg.model_dump(exclude_none=True)
            msg_entry["ts"] = datetime.now().isoformat()
            history.append(msg_entry)

            tool_calls = msg.tool_calls or []
            if not tool_calls or force_final:
                final_reply = msg.content
                break

            any_tool_calls = True
            _print_tool_block_header(round_num, len(tool_calls))
            for call in tool_calls:
                tool_name = call.function.name
                tool_args = dict(call.function.arguments)
                result = run_tool_with_confirmation(session, tool_name, tool_args, round_num)
                history.append({
                    "role": "tool", "tool_name": tool_name,
                    "content": json.dumps(result),
                    "ts": datetime.now().isoformat(),
                })
            print()

            history = trim(history)
            save_history(session, history)

        if interrupted:
            history = history_before_turn
            save_history(session, history)
            pending_default = user_input
        else:
            if not errored:
                if not final_reply:
                    # Nothing was ever streamed. Distinguish "the model genuinely failed to respond"
                    # from "tool calls succeeded but the model gave no closing summary" — these look
                    # identical (empty content) but mean very different things to the user; the latter
                    # isn't a failure, so "try again or rephrase" would be actively misleading there.
                    if any_tool_calls:
                        print_agent_reply("", empty_message="no closing summary from the model, "
                                                              "but the action(s) above completed")
                    else:
                        print_agent_reply("")
                else:
                    print()  # trailing blank line, matching print_agent_reply's own trailing print()
            history = trim(history)
            save_history(session, history)


if __name__ == "__main__":
    main()
