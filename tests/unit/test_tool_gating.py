"""Human-in-the-loop gating: run_tool_with_confirmation, the y/N prompt, and the streamed diff preview."""

import builtins
import io
import json
import sys

import pytest

import chat_agent as ca
from tests.helpers import known_bug, strip_ansi, write


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(ca.time, "sleep", lambda _s: None)


@pytest.fixture
def confirm(monkeypatch):
    """Script the answer to the y/N prompt and record what was asked."""
    asked = []

    def set_answer(answer: bool):
        def fake(prompt_text):
            asked.append(strip_ansi(prompt_text))
            return answer
        monkeypatch.setattr(ca, "_ask_confirmation", fake)
        return asked
    return set_answer


def log_entries(project):
    path = project / "logs" / "tool_calls.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


class TestAskConfirmation:
    def test_non_interactive_auto_declines_without_reading(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
        monkeypatch.setattr(builtins, "input", lambda *_: pytest.fail("must not prompt"))
        assert ca._ask_confirmation("Apply? ") is False
        assert "auto-declined: non-interactive session" in capsys.readouterr().out

    @pytest.mark.parametrize("answer,expected", [
        ("y", True), ("yes", True), ("Y", True), (" YES ", True),
        ("n", False), ("", False), ("no", False), ("yep", False), ("sure", False),
    ])
    def test_interactive_answers(self, monkeypatch, answer, expected):
        monkeypatch.setattr(sys, "stdin", FakeTTY())
        monkeypatch.setattr(builtins, "input", lambda *_: answer)
        assert ca._ask_confirmation("? ") is expected

    @pytest.mark.parametrize("exc", [EOFError, KeyboardInterrupt])
    def test_eof_and_ctrl_c_decline(self, monkeypatch, exc):
        monkeypatch.setattr(sys, "stdin", FakeTTY())

        def boom(*_):
            raise exc
        monkeypatch.setattr(builtins, "input", boom)
        assert ca._ask_confirmation("? ") is False


class TestReadOnlyTools:
    def test_auto_approved_executed_and_logged(self, project, capsys, monkeypatch):
        write(project, "a.txt", "hi")
        monkeypatch.setattr(ca, "_ask_confirmation", lambda *_: pytest.fail("read-only tools must not prompt"))
        result = ca.run_tool_with_confirmation("s", "read_file", {"path": "a.txt"}, round_num=3)
        assert result["ok"] is True
        out = strip_ansi(capsys.readouterr().out)
        assert 'read_file(path="a.txt")' in out and "✓ done" in out
        (entry,) = log_entries(project)
        assert entry["tool"] == "read_file" and entry["approval"] == "auto" and entry["outcome"] == "ok"
        assert entry["mutating"] is False and entry["round"] == 3 and entry["session"] == "s"

    def test_failure_logged_as_error(self, project, capsys):
        result = ca.run_tool_with_confirmation("s", "read_file", {"path": "missing"}, 1)
        assert result["ok"] is False
        assert "✗" in strip_ansi(capsys.readouterr().out)
        assert log_entries(project)[0]["outcome"] == "error"
        assert "does not exist" in log_entries(project)[0]["error"]

    def test_unknown_tool_is_reported_not_raised(self, project, capsys):
        result = ca.run_tool_with_confirmation("s", "delete_everything", {}, 1)
        assert result == {"ok": False, "error": "Unknown tool 'delete_everything'."}
        assert log_entries(project)[0]["outcome"] == "error"


class TestMutatingTools:
    def test_write_approved_shows_preview_then_writes(self, project, capsys, confirm, no_sleep):
        asked = confirm(True)
        result = ca.run_tool_with_confirmation("s", "write_file", {"path": "new.py", "content": "print(1)\n"}, 1)
        assert result["ok"] is True and (project / "new.py").read_text() == "print(1)\n"
        out = strip_ansi(capsys.readouterr().out)
        assert "Proposed write_file: new.py (create new file · 1 line · 9 bytes)" in out
        assert "1 + print(1)" in out
        assert out.index("Proposed") < out.index("✓ done")
        assert asked == ["● Apply this write_file to new.py? [y/N]: "]
        (entry,) = log_entries(project)
        assert (entry["approval"], entry["outcome"], entry["mutating"]) == ("user_approved", "ok", True)

    def test_write_declined_leaves_disk_untouched(self, project, capsys, confirm, no_sleep):
        write(project, "keep.txt", "original")
        confirm(False)
        result = ca.run_tool_with_confirmation("s", "write_file", {"path": "keep.txt", "content": "clobbered"}, 2)
        assert (project / "keep.txt").read_text() == "original"
        assert result["ok"] is False and result["declined"] is True
        assert "Do not attempt it again" in result["error"]
        out = strip_ansi(capsys.readouterr().out)
        assert "overwrite existing file" in out and "⊘ declined by user" in out
        (entry,) = log_entries(project)
        assert (entry["approval"], entry["outcome"], entry["duration_ms"]) == ("user_declined", "declined", 0)

    def test_edit_approved_shows_exact_diff(self, project, capsys, confirm, no_sleep):
        write(project, "f.py", "x = 1\ny = 2\n")
        confirm(True)
        ca.run_tool_with_confirmation("s", "edit_file", {"path": "f.py", "old_string": "y = 2", "new_string": "y = 3"}, 1)
        assert (project / "f.py").read_text() == "x = 1\ny = 3\n"
        out = strip_ansi(capsys.readouterr().out)
        assert "2 - y = 2" in out and "2 + y = 3" in out and "+1/-1" in out

    def test_edit_declined(self, project, confirm, no_sleep):
        write(project, "f.py", "a")
        confirm(False)
        ca.run_tool_with_confirmation("s", "edit_file", {"path": "f.py", "old_string": "a", "new_string": "b"}, 1)
        assert (project / "f.py").read_text() == "a"

    @pytest.mark.parametrize("name,args,err", [
        ("edit_file", {"path": "f.py", "old_string": "nope", "new_string": "x"}, "not found"),
        ("edit_file", {"path": "../f.py", "old_string": "a", "new_string": "b"}, "outside"),
        ("write_file", {"path": "/abs.txt", "content": "x"}, "Absolute"),
        ("write_file", {"path": "memory/default.json", "content": "[]"}, "restricted"),
    ])
    def test_invalid_mutation_never_reaches_the_user(self, project, capsys, monkeypatch, name, args, err):
        write(project, "f.py", "a")
        monkeypatch.setattr(ca, "_ask_confirmation", lambda *_: pytest.fail("must not ask about an invalid edit"))
        result = ca.run_tool_with_confirmation("s", name, args, 1)
        assert result["ok"] is False and err in result["error"]
        assert "Proposed" not in capsys.readouterr().out
        (entry,) = log_entries(project)
        assert (entry["outcome"], entry["approval"], entry["mutating"]) == ("validation_error", "auto", True)

    def test_approved_but_execution_fails_is_logged_as_error(self, project, confirm, no_sleep):
        confirm(True)
        result = ca.run_tool_with_confirmation("s", "write_file", {"path": "a/b/c.txt", "content": "x"}, 1)
        assert result["ok"] is False and "create_dirs" in result["error"]
        (entry,) = log_entries(project)
        assert (entry["approval"], entry["outcome"]) == ("user_approved", "error")

    def test_file_changed_between_preview_and_approval_does_not_apply_stale_edit(self, project, monkeypatch, no_sleep):
        """edit_file re-validates on execution: if the target changed while the user was deciding, nothing
        is silently overwritten."""
        target = write(project, "f.txt", "alpha")

        def concurrent_edit(_prompt):
            target.write_text("someone else rewrote this")
            return True
        monkeypatch.setattr(ca, "_ask_confirmation", concurrent_edit)
        result = ca.run_tool_with_confirmation("s", "edit_file",
                                               {"path": "f.txt", "old_string": "alpha", "new_string": "beta"}, 1)
        assert result["ok"] is False and target.read_text() == "someone else rewrote this"

    def test_large_content_is_truncated_in_log(self, project, confirm, no_sleep):
        confirm(True)
        ca.run_tool_with_confirmation("s", "write_file", {"path": "big.txt", "content": "z" * 5000}, 1)
        assert log_entries(project)[0]["arguments"]["content"].endswith("(5000 chars total)")

    # -- confirmed crash bugs: model-supplied arguments that blow up the preview step ---------------
    @known_bug("preview_write_file raises IsADirectoryError for a directory path; run_tool_with_confirmation "
               "only catches ToolError, so the exception escapes and crashes the whole chat loop.",
               raises=IsADirectoryError)
    def test_write_onto_directory_is_reported_not_raised(self, project, no_sleep):
        (project / "src").mkdir()
        result = ca.run_tool_with_confirmation("s", "write_file", {"path": "src", "content": "x"}, 1)
        assert result["ok"] is False

    @known_bug("A NUL byte in a model-supplied path raises ValueError from Path.resolve() during preview, "
               "which escapes run_tool_with_confirmation and crashes the chat loop.",
               raises=ValueError)
    def test_nul_byte_path_is_reported_not_raised(self, no_sleep):
        result = ca.run_tool_with_confirmation("s", "write_file", {"path": "a\x00b", "content": "x"}, 1)
        assert result["ok"] is False

    @known_bug("Unexpected/missing arguments to edit_file (e.g. a hallucinated replace_all=true) raise "
               "TypeError from preview_edit_file(**arguments), crashing the chat loop.",
               raises=TypeError)
    def test_edit_with_unexpected_argument_is_reported_not_raised(self, project, no_sleep):
        write(project, "f.txt", "a")
        result = ca.run_tool_with_confirmation(
            "s", "edit_file", {"path": "f.txt", "old_string": "a", "new_string": "b", "replace_all": True}, 1)
        assert result["ok"] is False

    @known_bug("Non-string write_file content (models often pass a JSON object for .json files) makes "
               "_summarize_mutation call .splitlines() on a dict → AttributeError crashes the chat loop.",
               raises=AttributeError)
    def test_write_with_non_string_content_is_reported_not_raised(self, no_sleep):
        result = ca.run_tool_with_confirmation("s", "write_file", {"path": "c.json", "content": {"a": 1}}, 1)
        assert result["ok"] is False

    @known_bug("Overwriting an existing non-UTF-8 file is previewed as 'create new file' (old_text=None), "
               "misrepresenting a destructive overwrite in the confirmation the user approves.")
    def test_overwrite_of_binary_file_is_not_presented_as_new_file(self, project, capsys, confirm, no_sleep):
        write(project, "data.bin", b"\xff\xfe\x00\x01")
        confirm(False)
        ca.run_tool_with_confirmation("s", "write_file", {"path": "data.bin", "content": "text"}, 1)
        assert "create new file" not in strip_ansi(capsys.readouterr().out)


class TestPrintStreaming:
    def test_prints_entire_text_then_newline(self, capsys, no_sleep):
        ca._print_streaming("abcdefghijklmnopqrstuvwxyz", chars_per_tick=5)
        out = capsys.readouterr().out
        assert strip_ansi(out) == "abcdefghijklmnopqrstuvwxyz\n"

    def test_cursor_hidden_then_restored(self, capsys, no_sleep):
        ca._print_streaming("x")
        out = capsys.readouterr().out
        assert out.index("\033[?25l") < out.index("x") < out.index("\033[?25h")

    def test_paced_in_ticks(self, monkeypatch, capsys):
        sleeps = []
        monkeypatch.setattr(ca.time, "sleep", sleeps.append)
        ca._print_streaming("x" * 20, tick=0.5, chars_per_tick=8)
        assert sleeps == [0.5, 0.5, 0.5]

    def test_empty_text(self, capsys, no_sleep):
        ca._print_streaming("")
        assert strip_ansi(capsys.readouterr().out) == "\n"

    @known_bug("_print_streaming's docstring promises Ctrl+C fast-forwards the diff preview, but only Esc is "
               "handled; KeyboardInterrupt escapes and crashes the app mid-confirmation.")
    def test_ctrl_c_fast_forwards_instead_of_raising(self, monkeypatch, capsys):
        def interrupt(_s):
            raise KeyboardInterrupt
        monkeypatch.setattr(ca.time, "sleep", interrupt)
        try:
            ca._print_streaming("full diff text")
        except KeyboardInterrupt:  # re-raised as a normal failure: a bare KeyboardInterrupt aborts pytest
            pytest.fail("KeyboardInterrupt escaped _print_streaming")
        assert "full diff text" in strip_ansi(capsys.readouterr().out)
