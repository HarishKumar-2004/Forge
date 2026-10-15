"""Session memory on disk, history trimming, session naming, model discovery, and the audit log."""

import json
from datetime import datetime

import pytest

import chat_agent as ca
from tests.helpers import known_bug


class TestMemoryFiles:
    def test_memory_path_creates_dir_lazily(self, project):
        assert not (project / "memory").exists()
        assert ca.memory_path("s") == project / "memory" / "s.json"
        assert (project / "memory").is_dir()

    def test_load_missing_session_is_empty(self):
        assert ca.load_history("nobody") == []

    def test_save_load_roundtrip(self):
        history = [{"role": "user", "content": "héllo 世界", "ts": "t"},
                   {"role": "assistant", "content": "x", "tool_calls": [{"function": {"name": "a", "arguments": {}}}]}]
        ca.save_history("s", history)
        assert ca.load_history("s") == history

    def test_saved_file_is_readable_json(self, memory_file):
        ca.save_history("s", [{"role": "user", "content": "a"}])
        assert json.loads(memory_file("s").read_text()) == [{"role": "user", "content": "a"}]

    def test_save_overwrites(self):
        ca.save_history("s", [{"role": "user", "content": "1"}] * 5)
        ca.save_history("s", [])
        assert ca.load_history("s") == []

    def test_sessions_are_isolated(self):
        ca.save_history("a", [{"role": "user", "content": "A"}])
        ca.save_history("b", [{"role": "user", "content": "B"}])
        assert ca.load_history("a")[0]["content"] == "A" and ca.load_history("b")[0]["content"] == "B"


class TestTrim:
    def test_under_limit_is_same_list(self):
        h = [{"i": i} for i in range(ca.MAX_HISTORY_MESSAGES)]
        assert ca.trim(h) is h

    def test_over_limit_keeps_newest(self):
        h = [{"i": i} for i in range(ca.MAX_HISTORY_MESSAGES + 7)]
        out = ca.trim(h)
        assert len(out) == ca.MAX_HISTORY_MESSAGES and out[0] == {"i": 7} and out[-1] == h[-1]

    @known_bug("trim() cuts at a fixed message count, so the retained history can begin with an orphaned "
               "'tool' result (or tool-calling assistant entry) whose originating user turn was dropped.")
    def test_trimmed_history_starts_at_a_user_turn(self):
        h = []
        for i in range(15):
            h += [{"role": "user", "content": f"q{i}"},
                  {"role": "assistant", "content": "", "tool_calls": [{}]},
                  {"role": "tool", "content": "{}"}]
        assert ca.trim(h)[0]["role"] == "user"


class TestSessionNames:
    @pytest.mark.parametrize("name", ["default", "a", "Work_2", "my-session", "A1-b_2"])
    def test_valid(self, name):
        assert ca.is_valid_session_name(name)

    @pytest.mark.parametrize("name", ["", "has space", "../etc", "a/b", "a.b", "semi;colon", "$(rm)", "é"])
    def test_invalid(self, name):
        assert not ca.is_valid_session_name(name)

    @known_bug("VALID_SESSION_NAME uses '$', which also matches before a trailing newline, so a name like "
               "'a\\n' (reachable via --resume) passes validation and becomes a filename containing a newline.")
    def test_trailing_newline_invalid(self):
        assert not ca.is_valid_session_name("a\n")

    def test_session_names_lists_memory_files_sorted(self):
        for n in ["zeta", "alpha", "mid"]:
            ca.save_history(n, [])
        assert ca.session_names() == ["alpha", "mid", "zeta"]

    def test_session_names_includes_unsaved_active(self):
        ca.save_history("saved", [])
        assert ca.session_names(active="fresh") == ["fresh", "saved"]

    def test_session_names_without_memory_dir(self, project):
        assert ca.session_names() == [] and not (project / "memory").exists()

    def test_session_names_ignores_non_json(self, project):
        ca.save_history("real", [])
        (project / "memory" / "notes.txt").write_text("x")
        assert ca.session_names() == ["real"]


class TestModelMessages:
    def test_to_model_message_strips_only_ts(self):
        entry = {"role": "tool", "content": "{}", "tool_name": "x", "ts": "now"}
        assert ca.to_model_message(entry) == {"role": "tool", "content": "{}", "tool_name": "x"}
        assert "ts" in entry  # original not mutated

    def test_installed_models_sorted(self, fake_ollama):
        fake_ollama.models = ["b:latest", "a:7b"]
        assert ca.installed_models() == ["a:7b", "b:latest"]

    def test_installed_models_unreachable_is_none(self, fake_ollama):
        fake_ollama.list_error = ConnectionError("refused")
        assert ca.installed_models() is None


class TestToolCallLog:
    def _entries(self, project):
        return [json.loads(l) for l in (project / "logs" / "tool_calls.jsonl").read_text().splitlines()]

    def test_appends_structured_jsonl_records(self, project):
        ca.log_tool_call("s", "read_file", {"path": "a"}, False, "auto", "ok", None, 5, 1)
        ca.log_tool_call("s", "write_file", {"path": "b", "content": "x"}, True, "user_declined", "declined", None, 0, 2)
        first, second = self._entries(project)
        assert {k: first[k] for k in first if k != "ts"} == {
            "session": "s", "tool": "read_file", "arguments": {"path": "a"}, "mutating": False,
            "approval": "auto", "outcome": "ok", "error": None, "duration_ms": 5, "round": 1}
        datetime.fromisoformat(first["ts"])
        assert second["approval"] == "user_declined" and second["round"] == 2

    def test_long_arguments_are_truncated_recursively(self, project):
        ca.log_tool_call("s", "write_file", {"content": "x" * 2000, "nested": {"deep": "y" * 900}, "n": 7},
                         True, "user_approved", "ok", None, 1, 1)
        args = self._entries(project)[0]["arguments"]
        assert args["content"] == "x" * ca.LOG_ARG_TRUNCATE_LEN + "...(2000 chars total)"
        assert args["nested"]["deep"].endswith("...(900 chars total)")
        assert args["n"] == 7

    def test_logging_does_not_mutate_caller_arguments(self):
        args = {"content": "x" * 2000}
        ca.log_tool_call("s", "write_file", args, True, "auto", "ok", None, 1, 1)
        assert len(args["content"]) == 2000

    def test_each_record_is_one_line_even_with_newlines(self, project):
        ca.log_tool_call("s", "write_file", {"content": "a\nb\nc"}, True, "auto", "error", "boom\nline2", 1, 1)
        assert len((project / "logs" / "tool_calls.jsonl").read_text().splitlines()) == 1
