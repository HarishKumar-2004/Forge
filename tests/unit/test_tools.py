"""Behavior, limits, and error handling of each sandboxed file tool and the tool registry."""

import json

import pytest

import tools
from tests.helpers import known_bug, write


# ------------------------------------------------------------------------------------------------
# list_files
# ------------------------------------------------------------------------------------------------
class TestListFiles:
    def test_lists_top_level_sorted_with_types_and_sizes(self, project):
        write(project, "b.txt", "hello")
        write(project, "a.py", "")
        (project / "dir").mkdir()
        result = tools.list_files()
        assert [e["name"] for e in result["entries"]] == ["a.py", "b.txt", "dir"]
        by_name = {e["name"]: e for e in result["entries"]}
        assert by_name["b.txt"] == {"name": "b.txt", "type": "file", "size": 5}
        assert by_name["dir"] == {"name": "dir", "type": "dir", "size": None}
        assert result["truncated"] is False and result["total_found"] == 3

    def test_non_recursive_does_not_descend(self, project):
        write(project, "dir/nested.txt", "x")
        names = [e["name"] for e in tools.list_files(".")["entries"]]
        assert names == ["dir"]

    def test_recursive_descends(self, project):
        write(project, "dir/sub/nested.txt", "x")
        names = [e["name"] for e in tools.list_files(".", recursive=True)["entries"]]
        assert names == ["dir", "dir/sub", "dir/sub/nested.txt"]

    def test_subdirectory_names_are_relative_to_listed_dir(self, project):
        write(project, "dir/inner.txt", "x")
        assert tools.list_files("dir")["entries"][0]["name"] == "inner.txt"

    @pytest.mark.parametrize("hidden", sorted(tools.BLOCKED_DIR_NAMES) + [".env", ".hidden_dir"])
    def test_hides_blocked_and_dot_entries_at_any_depth(self, project, hidden):
        write(project, f"{hidden}/x", "x") if not hidden == ".env" else write(project, ".env", "SECRET=1")
        write(project, f"pkg/{hidden}/y", "y")
        write(project, "visible.txt", "v")
        names = [e["name"] for e in tools.list_files(".", recursive=True)["entries"]]
        assert all(hidden not in n.split("/") for n in names)
        assert "visible.txt" in names

    def test_truncates_at_cap(self, project, monkeypatch):
        monkeypatch.setattr(tools, "MAX_LIST_ENTRIES", 5)
        for i in range(8):
            write(project, f"f{i}.txt", "")
        result = tools.list_files()
        assert len(result["entries"]) == 5
        assert result["truncated"] is True

    def test_exactly_at_cap_is_not_truncated(self, project, monkeypatch):
        monkeypatch.setattr(tools, "MAX_LIST_ENTRIES", 3)
        for i in range(3):
            write(project, f"f{i}.txt", "")
        assert tools.list_files()["truncated"] is False

    def test_empty_directory(self, project):
        (project / "empty").mkdir()
        assert tools.list_files("empty")["entries"] == []

    def test_errors(self, project):
        write(project, "file.txt", "x")
        with pytest.raises(tools.ToolError, match="not a directory"):
            tools.list_files("file.txt")
        with pytest.raises(tools.ToolError, match="does not exist"):
            tools.list_files("nope")


# ------------------------------------------------------------------------------------------------
# search_file
# ------------------------------------------------------------------------------------------------
class TestSearchFile:
    @pytest.fixture
    def corpus(self, project):
        write(project, "a.py", "import os\nprint('Hello')\nHELLO = 1\n")
        write(project, "sub/b.md", "say hello\nnothing here\n")
        return project

    def test_literal_case_insensitive_default(self, corpus):
        result = tools.search_file("hello")
        assert [(m["file"], m["line"]) for m in result["matches"]] == [("a.py", 2), ("a.py", 3), ("sub/b.md", 1)]
        assert result["match_count"] == 3

    def test_case_sensitive(self, corpus):
        result = tools.search_file("HELLO", case_sensitive=True)
        assert [(m["file"], m["line"], m["text"]) for m in result["matches"]] == [("a.py", 3, "HELLO = 1")]

    def test_literal_mode_treats_regex_metachars_literally(self, project):
        write(project, "r.txt", "a.b\naxb\n(x)\n")
        assert [m["text"] for m in tools.search_file("a.b")["matches"]] == ["a.b"]
        assert [m["text"] for m in tools.search_file("(x)")["matches"]] == ["(x)"]

    def test_regex_mode(self, corpus):
        result = tools.search_file(r"^hel+o\b", regex=True)
        assert [(m["file"], m["line"]) for m in result["matches"]] == [("a.py", 3)]

    def test_invalid_regex_is_a_tool_error(self, corpus):
        with pytest.raises(tools.ToolError, match="Invalid regular expression"):
            tools.search_file("([", regex=True)

    def test_single_file_target(self, corpus):
        result = tools.search_file("hello", path="sub/b.md")
        assert [m["file"] for m in result["matches"]] == ["sub/b.md"]

    def test_file_paths_reported_relative_to_project_root(self, corpus):
        assert tools.search_file("hello", path="sub")["matches"][0]["file"] == "sub/b.md"

    def test_skips_binary_files(self, project):
        write(project, "bin.dat", b"\xff\xfe\x00hello\x00")
        write(project, "t.txt", "hello")
        assert [m["file"] for m in tools.search_file("hello")["matches"]] == ["t.txt"]

    def test_skips_oversized_files(self, project, monkeypatch):
        monkeypatch.setattr(tools, "MAX_SEARCH_FILE_BYTES", 10)
        write(project, "big.txt", "hello " * 10)
        write(project, "small.txt", "hello")
        assert [m["file"] for m in tools.search_file("hello")["matches"]] == ["small.txt"]

    def test_skips_blocked_and_hidden_directories(self, project):
        for d in ["memory", "logs", ".git", "node_modules", "pkg/node_modules", ".cache"]:
            write(project, f"{d}/f.txt", "needle")
        write(project, "ok.txt", "needle")
        assert [m["file"] for m in tools.search_file("needle")["matches"]] == ["ok.txt"]

    def test_match_cap_truncates(self, project, monkeypatch):
        monkeypatch.setattr(tools, "MAX_SEARCH_MATCHES", 3)
        write(project, "many.txt", "x\n" * 10)
        write(project, "zzz.txt", "x\n")
        result = tools.search_file("x")
        assert result["match_count"] == 3 and result["truncated"] is True
        assert {m["file"] for m in result["matches"]} == {"many.txt"}

    def test_long_lines_shortened_for_display(self, project):
        write(project, "long.txt", "needle" + "a" * 1000)
        text = tools.search_file("needle")["matches"][0]["text"]
        assert len(text) == tools.MAX_LINE_DISPLAY_CHARS + 3 and text.endswith("...")

    def test_no_matches(self, corpus):
        result = tools.search_file("zzz-not-there")
        assert result["matches"] == [] and result["truncated"] is False

    def test_missing_path(self, project):
        with pytest.raises(tools.ToolError, match="does not exist"):
            tools.search_file("x", path="missing")


# ------------------------------------------------------------------------------------------------
# read_file
# ------------------------------------------------------------------------------------------------
class TestReadFile:
    @pytest.fixture
    def ten_lines(self, project):
        write(project, "ten.txt", "".join(f"line{i}\n" for i in range(1, 11)))

    def test_whole_file_numbered_cat_n_style(self, ten_lines):
        r = tools.read_file("ten.txt")
        assert r["content"].splitlines()[0] == "     1\tline1"
        assert r["content"].splitlines()[-1] == "    10\tline10"
        assert (r["start_line"], r["end_line"], r["total_lines"], r["truncated"]) == (1, 10, 10, False)

    def test_line_range(self, ten_lines):
        r = tools.read_file("ten.txt", start_line=3, end_line=5)
        assert r["content"] == "     3\tline3\n     4\tline4\n     5\tline5"

    def test_end_beyond_eof_clamped(self, ten_lines):
        r = tools.read_file("ten.txt", start_line=9, end_line=500)
        assert r["end_line"] == 10 and r["truncated"] is False

    @pytest.mark.parametrize("start", [0, -5])
    def test_nonpositive_start_clamped_to_one(self, ten_lines, start):
        assert tools.read_file("ten.txt", start_line=start)["start_line"] == 1

    def test_start_beyond_eof_returns_empty(self, ten_lines):
        assert tools.read_file("ten.txt", start_line=50)["content"] == ""

    def test_truncates_at_line_cap(self, project, monkeypatch):
        monkeypatch.setattr(tools, "MAX_READ_LINES", 4)
        write(project, "ten.txt", "x\n" * 10)
        r = tools.read_file("ten.txt")
        assert r["end_line"] == 4 and r["truncated"] is True
        r2 = tools.read_file("ten.txt", start_line=5)
        assert (r2["start_line"], r2["end_line"]) == (5, 8)

    def test_empty_file(self, project):
        write(project, "empty.txt", "")
        r = tools.read_file("empty.txt")
        assert r["content"] == "" and r["total_lines"] == 0

    def test_unicode_content(self, project):
        write(project, "u.txt", "héllo — 世界\n")
        assert "héllo — 世界" in tools.read_file("u.txt")["content"]

    def test_binary_file_is_a_tool_error(self, project):
        write(project, "img.png", b"\x89PNG\r\n\x1a\n\xff\xff")
        with pytest.raises(tools.ToolError, match="not a UTF-8 text file"):
            tools.read_file("img.png")

    def test_directory_is_a_tool_error(self, project):
        (project / "d").mkdir()
        with pytest.raises(tools.ToolError, match="is a directory"):
            tools.read_file("d")

    def test_missing_is_a_tool_error(self):
        with pytest.raises(tools.ToolError, match="does not exist"):
            tools.read_file("nope.txt")

    def test_string_line_numbers_from_model_do_not_crash_dispatcher(self, ten_lines):
        """Models sometimes send integers as strings; the dispatcher must still return a JSON-safe result."""
        result = tools.execute_tool("read_file", {"path": "ten.txt", "start_line": "2", "end_line": "3"})
        json.dumps(result)
        assert "ok" in result


# ------------------------------------------------------------------------------------------------
# write_file
# ------------------------------------------------------------------------------------------------
class TestWriteFile:
    def test_creates_new_file(self, project):
        r = tools.write_file("new.txt", "hello\n")
        assert (project / "new.txt").read_text() == "hello\n"
        assert r["created_new"] is True and r["bytes_written"] == 6

    def test_overwrites_existing(self, project):
        write(project, "f.txt", "old")
        r = tools.write_file("f.txt", "new")
        assert (project / "f.txt").read_text() == "new" and r["created_new"] is False

    def test_bytes_counted_as_utf8(self, project):
        assert tools.write_file("u.txt", "é")["bytes_written"] == 2

    def test_missing_parent_rejected_without_create_dirs(self, project):
        with pytest.raises(tools.ToolError, match="create_dirs"):
            tools.write_file("a/b/c.txt", "x")
        assert not (project / "a").exists()

    def test_create_dirs(self, project):
        tools.write_file("a/b/c.txt", "x", create_dirs=True)
        assert (project / "a/b/c.txt").read_text() == "x"

    def test_size_cap_enforced_and_nothing_written(self, project, monkeypatch):
        monkeypatch.setattr(tools, "MAX_WRITE_BYTES", 10)
        with pytest.raises(tools.ToolError, match="write limit"):
            tools.write_file("big.txt", "x" * 11)
        assert not (project / "big.txt").exists()
        tools.write_file("ok.txt", "x" * 10)

    def test_empty_content(self, project):
        tools.write_file("e.txt", "")
        assert (project / "e.txt").read_text() == ""

    def test_writing_onto_directory_fails_without_raising_from_dispatcher(self, project):
        (project / "d").mkdir()
        result = tools.execute_tool("write_file", {"path": "d", "content": "x"})
        assert result["ok"] is False

    @known_bug("write_file reports lines = count('\\n') + 1, so 'a\\n' is reported as 2 lines (and '' as 1), "
               "disagreeing with the 1-line count shown in the confirmation summary.")
    def test_line_count_matches_actual_lines(self, project):
        assert tools.write_file("one.txt", "a\n")["lines"] == 1


# ------------------------------------------------------------------------------------------------
# edit_file
# ------------------------------------------------------------------------------------------------
class TestEditFile:
    def test_replaces_unique_occurrence_and_returns_diff(self, project):
        write(project, "f.py", "a = 1\nb = 2\nc = 3\n")
        r = tools.edit_file("f.py", "b = 2", "b = 20")
        assert (project / "f.py").read_text() == "a = 1\nb = 20\nc = 3\n"
        assert "-b = 2" in r["diff"] and "+b = 20" in r["diff"]

    def test_multiline_replacement_preserves_rest(self, project):
        write(project, "f.txt", "head\nold1\nold2\ntail\n")
        tools.edit_file("f.txt", "old1\nold2\n", "new\n")
        assert (project / "f.txt").read_text() == "head\nnew\ntail\n"

    def test_deletion_via_empty_new_string(self, project):
        write(project, "f.txt", "keep REMOVE keep")
        tools.edit_file("f.txt", " REMOVE", "")
        assert (project / "f.txt").read_text() == "keep keep"

    def test_whitespace_must_match_exactly(self, project):
        write(project, "f.py", "    indented\n")
        with pytest.raises(tools.ToolError, match="not found"):
            tools.edit_file("f.py", "indented\n  ", "x")

    @pytest.mark.parametrize("old,new,msg", [
        ("", "x", "must not be empty"),
        ("same", "same", "identical"),
        ("absent", "x", "not found"),
        ("dup", "x", "ambiguous: it matches 2"),
    ])
    def test_rejections_leave_file_untouched(self, project, old, new, msg):
        write(project, "f.txt", "dup same dup")
        with pytest.raises(tools.ToolError, match=msg):
            tools.edit_file("f.txt", old, new)
        assert (project / "f.txt").read_text() == "dup same dup"

    def test_binary_and_directory_and_missing(self, project):
        write(project, "b.bin", b"\xff\xfe")
        (project / "d").mkdir()
        with pytest.raises(tools.ToolError, match="UTF-8"):
            tools.edit_file("b.bin", "a", "b")
        with pytest.raises(tools.ToolError, match="directory"):
            tools.edit_file("d", "a", "b")
        with pytest.raises(tools.ToolError, match="does not exist"):
            tools.edit_file("nope", "a", "b")

    def test_regex_special_chars_are_literal(self, project):
        write(project, "f.txt", "x = a.*b\n")
        tools.edit_file("f.txt", "a.*b", "c")
        assert (project / "f.txt").read_text() == "x = c\n"

    @known_bug("edit_file reads with universal newlines and writes back '\\n', silently converting every "
               "CRLF line ending in the file to LF on any edit.")
    def test_preserves_crlf_line_endings(self, project):
        (project / "win.txt").write_bytes(b"one\r\ntwo\r\nthree\r\n")
        tools.edit_file("win.txt", "two", "TWO")
        assert (project / "win.txt").read_bytes() == b"one\r\nTWO\r\nthree\r\n"


# ------------------------------------------------------------------------------------------------
# previews (used to render the confirmation diff before a mutation runs)
# ------------------------------------------------------------------------------------------------
class TestPreviews:
    def test_preview_write_new_file(self, project):
        p = tools.preview_write_file("n.txt", "x")
        assert p == {"path": "n.txt", "old_text": None, "new_text": "x", "exists": False}
        assert not (project / "n.txt").exists()

    def test_preview_write_existing_file(self, project):
        write(project, "f.txt", "old")
        p = tools.preview_write_file("f.txt", "new")
        assert (p["old_text"], p["exists"]) == ("old", True)
        assert (project / "f.txt").read_text() == "old"

    def test_preview_edit_does_not_write(self, project):
        write(project, "f.txt", "a b c")
        p = tools.preview_edit_file("f.txt", "b", "B")
        assert (p["old_text"], p["new_text"]) == ("a b c", "a B c")
        assert (project / "f.txt").read_text() == "a b c"

    def test_preview_edit_validates_like_edit(self, project):
        write(project, "f.txt", "x x")
        with pytest.raises(tools.ToolError, match="ambiguous"):
            tools.preview_edit_file("f.txt", "x", "y")

    def test_preview_respects_sandbox(self):
        with pytest.raises(tools.ToolError):
            tools.preview_write_file("../x", "y")
        with pytest.raises(tools.ToolError):
            tools.preview_edit_file("/etc/hosts", "a", "b")

    def test_preview_of_existing_binary_file_reports_it_exists(self, project):
        write(project, "b.bin", b"\xff\xfe")
        p = tools.preview_write_file("b.bin", "text")
        assert p["exists"] is True


# ------------------------------------------------------------------------------------------------
# registry + dispatcher
# ------------------------------------------------------------------------------------------------
class TestRegistryAndDispatch:
    def test_registry_names_match_schemas(self):
        for name, spec in tools.TOOL_REGISTRY.items():
            assert spec.name == name == spec.schema["function"]["name"]

    def test_only_mutating_tools_require_confirmation(self):
        flags = {n: (s.mutating, s.requires_confirmation) for n, s in tools.TOOL_REGISTRY.items()}
        assert flags == {
            "list_files": (False, False), "search_file": (False, False), "read_file": (False, False),
            "write_file": (True, True), "edit_file": (True, True),
        }

    @pytest.mark.parametrize("name", list(tools.TOOL_REGISTRY))
    def test_schema_is_well_formed_and_matches_signature(self, name):
        import inspect
        spec = tools.TOOL_REGISTRY[name]
        fn = spec.schema["function"]
        assert spec.schema["type"] == "function" and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        sig = inspect.signature(spec.func).parameters
        assert set(params["properties"]) == set(sig), "schema properties drift from function signature"
        for req in params["required"]:
            assert sig[req].default is inspect.Parameter.empty, f"{req} is required in schema but has a default"
        for pname, prop in params["properties"].items():
            assert prop["type"] in {"string", "boolean", "integer"} and prop["description"]

    def test_tool_schemas_is_json_serializable(self):
        schemas = tools.tool_schemas()
        assert len(schemas) == len(tools.TOOL_REGISTRY)
        json.dumps(schemas)

    def test_success_gets_ok_true(self, project):
        write(project, "f.txt", "x")
        assert tools.execute_tool("read_file", {"path": "f.txt"})["ok"] is True

    def test_unknown_tool(self):
        assert tools.execute_tool("rm_rf", {}) == {"ok": False, "error": "Unknown tool 'rm_rf'."}

    def test_tool_error_is_captured(self):
        r = tools.execute_tool("read_file", {"path": "missing"})
        assert r["ok"] is False and "does not exist" in r["error"]

    @pytest.mark.parametrize("args", [{}, {"path": "f.txt", "bogus": 1}, {"path": None}])
    def test_bad_arguments_from_model_are_captured_not_raised(self, project, args):
        write(project, "f.txt", "x")
        r = tools.execute_tool("read_file", args)
        assert r["ok"] is False and r["error"]
        json.dumps(r)

    @pytest.mark.parametrize("name", list(tools.TOOL_REGISTRY))
    def test_every_result_is_json_serializable(self, project, name):
        write(project, "f.txt", "alpha beta\n")
        args = {
            "list_files": {}, "search_file": {"pattern": "alpha"}, "read_file": {"path": "f.txt"},
            "write_file": {"path": "g.txt", "content": "x"},
            "edit_file": {"path": "f.txt", "old_string": "alpha", "new_string": "gamma"},
        }[name]
        result = tools.execute_tool(name, args)
        assert result["ok"] is True
        json.dumps(result)
