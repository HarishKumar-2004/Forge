"""Terminal rendering: markdown → ANSI formatting, ANSI-aware wrapping, diff previews, display blocks."""

import re

import pytest

import chat_agent as ca
from tests.helpers import known_bug, strip_ansi

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


# ------------------------------------------------------------------------------------------------
# Markdown formatting
# ------------------------------------------------------------------------------------------------
class TestInlineMarkup:
    @pytest.mark.parametrize("src,expected", [
        ("**bold**", f"{BOLD}bold{RESET}"),
        ("__bold__", f"{BOLD}bold{RESET}"),
        ("`code`", f"{DIM}code{RESET}"),
        ("a **b** c `d` e", f"a {BOLD}b{RESET} c {DIM}d{RESET} e"),
        ("** padded **", f"{BOLD}padded{RESET}"),
        ("plain text", "plain text"),
        ("two  spaces   collapse", "two spaces collapse"),
    ])
    def test_inline(self, src, expected):
        assert ca._apply_inline_markup(src) == expected

    def test_leading_indentation_is_preserved(self):
        assert ca._apply_inline_markup("    indented") == "    indented"

    def test_unclosed_markers_left_alone(self):
        assert ca._apply_inline_markup("a ** b and ` c") == "a ** b and ` c"

    @known_bug("'__' is treated as bold even inside words/code spans, so '__init__.py' renders as a bold "
               "'init.py' — dunder names (very common in a coding assistant's replies) are mangled.")
    def test_python_dunder_names_survive(self):
        assert strip_ansi(ca.format_agent_reply("Edit `__init__.py` and __main__")) == "Edit __init__.py and __main__"


class TestLineFormatting:
    @pytest.mark.parametrize("src,expected", [
        ("# Title", f"{BOLD}Title{RESET}"),
        ("###### Deep", f"{BOLD}Deep{RESET}"),
        ("## **Bold** `x`", f"{BOLD}Bold x{RESET}"),
        ("#NoSpace", "#NoSpace"),
        ("####### seven", "####### seven"),
    ])
    def test_headings(self, src, expected):
        assert ca._format_reply_line(src, False)[0] == expected

    @pytest.mark.parametrize("src", ["- item", "* item", "+ item", "   - item"])
    def test_bullets_normalized(self, src):
        assert ca._format_reply_line(src, False)[0] == "  • item"

    def test_emphasis_at_line_start_is_not_a_bullet(self):
        assert ca._format_reply_line("*not* a bullet", False)[0] == "*not* a bullet"

    @pytest.mark.parametrize("src,expected", [("1. one", "  1. one"), ("  12) twelve", "  12. twelve")])
    def test_numbered(self, src, expected):
        assert ca._format_reply_line(src, False)[0] == expected

    def test_bullet_content_gets_inline_markup(self):
        assert ca._format_reply_line("- **x**", False)[0] == f"  • {BOLD}x{RESET}"

    def test_code_fence_toggles_and_emits_nothing(self):
        assert ca._format_reply_line("```python", False) == (None, True)
        assert ca._format_reply_line("```", True) == (None, False)

    def test_code_lines_are_verbatim_indented_dim(self):
        line, state = ca._format_reply_line("# not a heading **raw**", True)
        assert line == f"    {DIM}# not a heading **raw**{RESET}" and state is True


class TestFormatAgentReply:
    def test_strips_emoji(self):
        out = ca.format_agent_reply("Done ✅ 🚀 ok")
        assert "✅" not in out and "🚀" not in out and "Done" in out and "ok" in out

    def test_strips_flags_keycaps_and_zwj_sequences(self):
        out = ca.format_agent_reply("a🇮🇳b 1️⃣ c👨‍👩‍👧d")
        assert not ca.EMOJI_PATTERN.search(out)

    def test_keeps_non_emoji_unicode(self):
        assert ca.format_agent_reply("café — 世界 → λ") == "café — 世界 → λ"

    def test_collapses_blank_runs_to_one(self):
        assert ca.format_agent_reply("a\n\n\n\nb") == "a\n\nb"

    def test_code_block_contents_not_markdown_processed(self):
        out = ca.format_agent_reply("```\n- not bullet\n# not heading\n```\nafter")
        lines = out.split("\n")
        assert lines == [f"    {DIM}- not bullet{RESET}", f"    {DIM}# not heading{RESET}", "after"]

    def test_unterminated_code_fence_stays_code(self):
        out = ca.format_agent_reply("```\ncode line")
        assert out == f"    {DIM}code line{RESET}"

    def test_trailing_whitespace_stripped(self):
        assert ca.format_agent_reply("hello   \nworld\t") == "hello\nworld"

    def test_full_document(self):
        md = "# Plan\n\n1. Read **file**\n2. Edit it\n\n- note `x`\n\n```\nprint(1)\n```"
        assert strip_ansi(ca.format_agent_reply(md)) == "Plan\n\n  1. Read file\n  2. Edit it\n\n  • note x\n\n    print(1)"

    def test_streaming_and_static_formatting_agree(self):
        """call_with_streaming_reply and format_agent_reply share _format_reply_line — they must agree."""
        md = "# H\ntext **b**\n\n\n- x\n```\ncode\n```\n1. y"
        state, streamed = False, []
        for line in md.split("\n"):
            out, state = ca._format_reply_line(line.rstrip(), state)
            if out is not None:
                streamed.append(out)
        collapsed = re.sub(r"\n{3,}", "\n\n", "\n".join(streamed))
        assert collapsed == ca.format_agent_reply(md)


# ------------------------------------------------------------------------------------------------
# ANSI-aware wrapping
# ------------------------------------------------------------------------------------------------
class TestWrapAnsiLine:
    def test_short_line_untouched(self):
        assert ca._wrap_ansi_line("short", 20) == ["short"]

    def test_ansi_codes_do_not_count_toward_width(self):
        styled = f"{BOLD}{'x' * 18}{RESET}"
        assert ca._wrap_ansi_line(styled, 20) == [styled]

    def test_wraps_on_word_boundaries_within_width(self):
        text = "the quick brown fox jumps over the lazy dog " * 4
        out = ca._wrap_ansi_line(text.strip(), 20)
        assert all(len(strip_ansi(l)) <= 20 for l in out)
        assert " ".join(strip_ansi(l).strip() for l in out).split() == text.split()

    def test_never_breaks_inside_a_word(self):
        out = ca._wrap_ansi_line("aaa " + "b" * 30 + " ccc", 10)
        assert "b" * 30 in [strip_ansi(l).strip() for l in out]

    def test_styling_kept_attached_across_cut(self):
        line = f"plain {BOLD}bold words that wrap{RESET} tail"
        out = ca._wrap_ansi_line(line, 12)
        joined = "".join(out)
        assert BOLD in joined and RESET in joined
        assert strip_ansi(" ".join(out)).split() == strip_ansi(line).split()

    @known_bug("textwrap expands tabs to spaces, so visible_text.index(piece) raises ValueError on any long "
               "line containing a tab — e.g. a Go/Makefile code block in a reply crashes rendering.",
               raises=ValueError)
    def test_long_line_with_tab_does_not_crash(self):
        out = ca._wrap_ansi_line("\t" + "word " * 30, 40)
        assert out

    def test_wrap_for_display_respects_terminal_width_and_indent(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "30")
        out = ca._wrap_for_display("word " * 20 + "\nsecond", indent_width=2)
        assert all(len(strip_ansi(l)) <= 28 for l in out)
        assert out[-1] == "second"

    def test_wrap_for_display_has_a_minimum_width(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "5")
        out = ca._wrap_for_display("word " * 10, indent_width=2)
        assert all(len(l) <= 20 for l in out)


# ------------------------------------------------------------------------------------------------
# Diff rendering & mutation summary (what the user sees before approving a write)
# ------------------------------------------------------------------------------------------------
class TestRenderDiff:
    def test_new_file_all_additions_numbered(self):
        out = strip_ansi(ca.render_diff(None, "a\nb"))
        assert out.split("\n") == ["    1 + a", "    2 + b"]

    def test_replacement_shows_minus_then_plus(self):
        out = strip_ansi(ca.render_diff("a\nb\nc", "a\nB\nc")).split("\n")
        assert out == ["    1   a", "    2 - b", "    2 + B", "    3   c"]

    def test_colors(self):
        out = ca.render_diff("old", "new")
        assert "\033[31m" in out and "\033[32m" in out

    def test_long_unchanged_runs_are_elided_with_context(self):
        old = "\n".join(f"l{i}" for i in range(1, 21))
        new = old.replace("l10", "L10")
        out = strip_ansi(ca.render_diff(old, new, context=2)).split("\n")
        assert "⋮" in out[2]
        assert "l1" in out[0] and "l2" in out[1]
        assert "l8" in out[3] and "l9" in out[4] and "- l10" in out[5] and "+ L10" in out[6]

    def test_identical_text_has_no_changes(self):
        out = strip_ansi(ca.render_diff("a\nb", "a\nb"))
        assert "+" not in out and "-" not in out

    def test_pure_insertion_and_deletion(self):
        assert " + new" in strip_ansi(ca.render_diff("a", "a\nnew"))
        assert " - gone" in strip_ansi(ca.render_diff("a\ngone", "a"))

    def test_empty_new_file(self):
        assert ca.render_diff(None, "") == ""


class TestSummarizeMutation:
    def test_new_file(self):
        assert ca._summarize_mutation(None, "a\nb\n") == "create new file · 2 lines · 4 bytes"

    def test_new_single_line_file_is_singular(self):
        assert ca._summarize_mutation(None, "x") == "create new file · 1 line · 1 bytes"

    def test_bytes_use_thousands_separator(self):
        assert "1,200 bytes" in ca._summarize_mutation(None, "x" * 1200)

    def test_overwrite_counts(self):
        summary = ca._summarize_mutation("a\nb\nc", "a\nB\nc\nd")
        assert summary == "overwrite existing file · 3 → 4 lines · +2/-1"

    def test_no_op_overwrite(self):
        assert ca._summarize_mutation("a", "a").endswith("+0/-0")


# ------------------------------------------------------------------------------------------------
# Display blocks
# ------------------------------------------------------------------------------------------------
class TestDisplayBlocks:
    def test_separator_spans_terminal_width(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "40")
        assert strip_ansi(ca.separator()) == "─" * 40

    def test_session_banner_centered(self, capsys, monkeypatch):
        monkeypatch.setenv("COLUMNS", "40")
        ca.print_session_banner("work")
        line = strip_ansi(capsys.readouterr().out).rstrip("\n")
        assert len(line) == 40 and " Session: work " in line
        left, right = line.split(" Session: work ")
        assert abs(len(left) - len(right)) <= 1 and set(left + right) == {"─"}

    def test_session_banner_narrow_terminal(self, capsys, monkeypatch):
        monkeypatch.setenv("COLUMNS", "8")
        ca.print_session_banner("a-very-long-session-name")
        assert len(strip_ansi(capsys.readouterr().out).rstrip("\n")) == 8

    def test_user_prompt_block_full_width_grey_bar(self, capsys, monkeypatch):
        monkeypatch.setenv("COLUMNS", "40")
        ca.print_user_prompt_block("s", "hi\nthere")
        raw = capsys.readouterr().out
        lines = raw.split("\n")
        assert all(l.startswith(ca.USER_PROMPT_BG) for l in lines[:2])
        plain = [strip_ansi(l) for l in lines[:2]]
        assert plain[0] == "You[s]> hi".ljust(40)
        assert plain[1] == "        there".ljust(40)  # continuation aligned under the label
        assert lines[2] == ""

    @known_bug("print_user_prompt_block hard-truncates each line to the terminal width, so the visible "
               "record of a long prompt silently loses everything past the first ~80 characters.")
    def test_user_prompt_block_does_not_lose_long_prompt_text(self, capsys, monkeypatch):
        monkeypatch.setenv("COLUMNS", "40")
        ca.print_user_prompt_block("s", "start " + "x" * 60 + " END")
        assert "END" in strip_ansi(capsys.readouterr().out)

    def test_agent_reply_bullet_and_indentation(self, capsys, monkeypatch):
        monkeypatch.setenv("COLUMNS", "30")
        ca.print_agent_reply("word " * 15)
        lines = strip_ansi(capsys.readouterr().out).split("\n")
        assert lines[0].startswith("● ")
        assert all(l.startswith("  ") for l in lines[1:-2]) and lines[-1] == "" and lines[-2] == ""

    def test_agent_reply_empty_placeholders(self, capsys):
        ca.print_agent_reply("   ")
        ca.print_agent_reply("", empty_message="custom")
        ca.print_agent_reply("🚀")  # emoji-only reply is empty after cleaning
        out = strip_ansi(capsys.readouterr().out)
        assert out.count("model returned an empty response") == 2 and "(custom)" in out

    def test_recap_style(self, capsys):
        ca.print_recap("  You fixed the bug ✅  ")
        raw = capsys.readouterr().out
        assert ca.RECAP_COLOR in raw and ca.ITALIC in raw
        assert strip_ansi(raw).startswith("◆ recap: You fixed the bug")
        assert "✅" not in raw

    def test_recap_first_line_fits_terminal(self, capsys, monkeypatch):
        monkeypatch.setenv("COLUMNS", "40")
        ca.print_recap("word " * 30)
        for line in strip_ansi(capsys.readouterr().out).split("\n"):
            assert len(line) <= 40

    @pytest.mark.parametrize("args,expected", [
        ({"path": "a.txt"}, 'read_file(path="a.txt")'),
        ({"path": "a", "recursive": True}, 'read_file(path="a", recursive=True)'),
        ({"content": "l1\nl2"}, 'read_file(content="l1\\nl2")'),
        ({}, "read_file()"),
    ])
    def test_format_tool_call(self, args, expected):
        assert strip_ansi(ca.format_tool_call("read_file", args)) == expected

    def test_format_tool_call_truncates_long_values(self):
        out = strip_ansi(ca.format_tool_call("write_file", {"content": "x" * 500}))
        assert f'{"x" * ca.DISPLAY_ARG_TRUNCATE_LEN}… (500 chars)' in out

    @pytest.mark.parametrize("result,expected", [
        ({"ok": True}, "✓ done (12ms)"),
        ({"ok": False, "declined": True, "error": "x"}, "⊘ declined by user"),
        ({"ok": False, "error": "bad\nthing"}, "✗ bad thing"),
        ({"ok": False}, "✗ unknown error"),
    ])
    def test_tool_status(self, capsys, result, expected):
        ca._print_tool_status(result, 12)
        assert strip_ansi(capsys.readouterr().out).strip() == expected

    def test_tool_status_truncates_long_errors(self, capsys):
        ca._print_tool_status({"ok": False, "error": "e" * 500}, 0)
        assert strip_ansi(capsys.readouterr().out).strip() == "✗ " + "e" * 200 + "…"

    @pytest.mark.parametrize("count,noun", [(1, "1 tool call"), (3, "3 tool calls")])
    def test_tool_block_header(self, capsys, count, noun):
        ca._print_tool_block_header(2, count)
        assert strip_ansi(capsys.readouterr().out).strip() == f"● Tools (round 2, {noun})"
