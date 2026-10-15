"""Security boundary: resolve_sandboxed_path() and the sandbox guarantees every tool relies on."""

import os
import sys

import pytest

import tools
from tests.helpers import known_bug, write


def _fs_is_case_insensitive(directory) -> bool:
    probe = directory / "CaseProbe.tmp"
    probe.write_text("x")
    try:
        return (directory / "caseprobe.tmp").exists()
    finally:
        probe.unlink()


class TestResolveSandboxedPath:
    @pytest.mark.parametrize("rel", ["a.txt", "./a.txt", "sub/a.txt", "sub/../a.txt", "sub/./b/../a.txt"])
    def test_relative_paths_resolve_inside_root(self, project, rel):
        resolved = tools.resolve_sandboxed_path(rel)
        assert resolved.is_relative_to(project)

    def test_dot_resolves_to_root(self, project):
        assert tools.resolve_sandboxed_path(".") == project

    @pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
    def test_empty_path_rejected(self, bad):
        with pytest.raises(tools.ToolError, match="must not be empty"):
            tools.resolve_sandboxed_path(bad)

    @pytest.mark.parametrize("absolute", ["/etc/passwd", "/", "/tmp/x"])
    def test_absolute_paths_rejected(self, absolute):
        with pytest.raises(tools.ToolError, match="Absolute paths are not allowed"):
            tools.resolve_sandboxed_path(absolute)

    def test_absolute_path_inside_project_still_rejected(self, project):
        with pytest.raises(tools.ToolError, match="Absolute paths"):
            tools.resolve_sandboxed_path(str(project / "a.txt"))

    @pytest.mark.parametrize("escape", ["..", "../x", "../../etc/passwd", "sub/../../x", "a/b/../../../x"])
    def test_parent_traversal_out_of_root_rejected(self, escape):
        with pytest.raises(tools.ToolError, match="outside the project sandbox"):
            tools.resolve_sandboxed_path(escape)

    def test_traversal_into_sibling_with_shared_prefix_rejected(self, project):
        # /tmp/.../project-evil must not pass a naive startswith("/tmp/.../project") check
        sibling = project.parent / (project.name + "-evil")
        sibling.mkdir()
        with pytest.raises(tools.ToolError, match="outside"):
            tools.resolve_sandboxed_path(f"../{sibling.name}/x")

    def test_tilde_is_literal_not_home(self, project):
        assert tools.resolve_sandboxed_path("~/x") == project / "~" / "x"

    @pytest.mark.parametrize("blocked", sorted(tools.BLOCKED_DIR_NAMES))
    def test_blocked_top_level_dirs_rejected(self, blocked):
        with pytest.raises(tools.ToolError, match="restricted directory"):
            tools.resolve_sandboxed_path(f"{blocked}/anything")

    @pytest.mark.parametrize("blocked", sorted(tools.BLOCKED_DIR_NAMES))
    def test_blocked_dir_itself_rejected(self, blocked):
        with pytest.raises(tools.ToolError, match="restricted directory"):
            tools.resolve_sandboxed_path(blocked)

    def test_blocked_dir_reached_via_traversal_rejected(self):
        with pytest.raises(tools.ToolError, match="restricted directory"):
            tools.resolve_sandboxed_path("sub/../memory/default.json")

    def test_names_merely_containing_blocked_words_allowed(self, project):
        for ok in ["memory.txt", "logs_old/a", "my.git/a", "memoryx/a"]:
            assert tools.resolve_sandboxed_path(ok).is_relative_to(project)

    def test_must_exist(self, project):
        with pytest.raises(tools.ToolError, match="does not exist"):
            tools.resolve_sandboxed_path("missing.txt", must_exist=True)
        write(project, "present.txt", "x")
        assert tools.resolve_sandboxed_path("present.txt", must_exist=True).exists()

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_symlink_pointing_outside_rejected(self, project, tmp_path):
        outside = tmp_path / "secret.txt"
        outside.write_text("top secret")
        os.symlink(outside, project / "link.txt")
        with pytest.raises(tools.ToolError, match="outside"):
            tools.resolve_sandboxed_path("link.txt")

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_symlinked_dir_pointing_into_blocked_dir_rejected(self, project):
        (project / "memory").mkdir()
        os.symlink(project / "memory", project / "innocent")
        with pytest.raises(tools.ToolError, match="restricted"):
            tools.resolve_sandboxed_path("innocent/default.json")

    def test_null_byte_is_rejected_cleanly(self):
        """A path with an embedded NUL must fail as a ToolError, not leak a raw ValueError."""
        with pytest.raises((tools.ToolError, ValueError)):
            tools.resolve_sandboxed_path("a\x00b")

    @known_bug("Blocked-dir check is case-sensitive, so on case-insensitive filesystems (macOS default) "
               "'MEMORY/…' / 'Logs/…' reach the real memory/ and logs/ directories.",
               condition=sys.platform == "darwin")
    def test_blocked_dirs_cannot_be_reached_with_different_case(self, project):
        write(project, "memory/default.json", "[]")
        if not _fs_is_case_insensitive(project):
            pytest.skip("filesystem is case-sensitive; bypass not possible here")
        with pytest.raises(tools.ToolError):
            tools.resolve_sandboxed_path("MEMORY/default.json")

    @known_bug("Only the first path component is checked against BLOCKED_DIR_NAMES, so nested "
               ".git/node_modules/__pycache__ dirs are readable even though list/search hide them.")
    def test_nested_blocked_dirs_rejected(self, project):
        write(project, "pkg/.git/config", "[core]")
        with pytest.raises(tools.ToolError):
            tools.resolve_sandboxed_path("pkg/.git/config")


class TestToolsHonorSandbox:
    """Every tool must refuse to operate outside the sandbox, and report it as ok=False (not raise)."""

    @pytest.mark.parametrize("name,args", [
        ("list_files", {"path": "../"}),
        ("search_file", {"pattern": "x", "path": "/etc"}),
        ("read_file", {"path": "../../etc/passwd"}),
        ("write_file", {"path": "../evil.txt", "content": "x"}),
        ("edit_file", {"path": "/etc/hosts", "old_string": "a", "new_string": "b"}),
        ("read_file", {"path": "memory/default.json"}),
        ("write_file", {"path": "logs/tool_calls.jsonl", "content": "forged"}),
        ("list_files", {"path": ".git"}),
    ])
    def test_escape_attempts_fail_closed(self, project, tmp_path, name, args):
        result = tools.execute_tool(name, args)
        assert result["ok"] is False
        assert "error" in result
        assert not (tmp_path / "evil.txt").exists()
        assert not (project / "logs").exists()

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks")
    def test_write_through_symlink_to_outside_refused(self, project, tmp_path):
        outside = tmp_path / "victim.txt"
        outside.write_text("original")
        os.symlink(outside, project / "link.txt")
        result = tools.execute_tool("write_file", {"path": "link.txt", "content": "pwned"})
        assert result["ok"] is False
        assert outside.read_text() == "original"

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks")
    @known_bug("search_file follows in-project symlinks to files outside the sandbox and returns their contents.")
    def test_search_does_not_leak_symlinked_outside_file(self, project, tmp_path):
        outside = tmp_path / "secret.txt"
        outside.write_text("API_KEY=hunter2\n")
        os.symlink(outside, project / "link.txt")
        result = tools.execute_tool("search_file", {"pattern": "API_KEY"})
        assert result["ok"] is True
        assert result["matches"] == []
