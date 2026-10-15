"""prompt_toolkit integration: slash-command completion, key bindings, adaptive menu height."""

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.keys import Keys

import chat_agent as ca


def complete(text: str) -> list[str]:
    return [c.text for c in ca.SlashCommandCompleter().get_completions(Document(text), None)]


class TestSlashCommandCompleter:
    def test_slash_alone_lists_every_command(self):
        assert complete("/") == [c.name for c in ca.SLASH_COMMANDS]

    @pytest.mark.parametrize("prefix,expected", [
        ("/se", ["/session", "/sessions"]),
        ("/sessions", ["/sessions"]),
        ("/mod", ["/model", "/models"]),
        ("/q", ["/quit"]),
        ("/zzz", []),
    ])
    def test_prefix_narrowing(self, prefix, expected):
        assert complete(prefix) == expected

    @pytest.mark.parametrize("text", ["", "hello", "/session name", " /reset", "x/"])
    def test_no_completions_outside_a_bare_slash_command(self, text):
        assert complete(text) == []

    def test_completion_replaces_typed_prefix_and_shows_usage_and_help(self):
        c = next(ca.SlashCommandCompleter().get_completions(Document("/ren"), None))
        assert c.start_position == -4
        assert c.display[0][1] == "/rename <name>"
        assert c.display_meta[0][1] == "Rename the active session"

    def test_command_table_is_consistent(self):
        names = [c.name for c in ca.SLASH_COMMANDS]
        assert len(names) == len(set(names))
        assert all(c.usage.startswith(c.name) and c.help for c in ca.SLASH_COMMANDS)
        assert set(ca.EXIT_WORDS) <= set(names)


class TestKeyBindings:
    @pytest.fixture
    def bindings(self):
        return ca._build_key_bindings()

    def _handler(self, bindings, *keys):
        matches = bindings.get_bindings_for_keys(keys)
        assert matches, f"no binding for {keys}"
        return matches[-1].handler

    class FakeBuffer:
        def __init__(self, text="", complete_state=None):
            self.text, self.complete_state = text, complete_state
            self.submitted = self.cancelled = False

        def validate_and_handle(self):
            self.submitted = True

        def insert_text(self, t):
            self.text += t

        def cancel_completion(self):
            self.cancelled = True
            self.complete_state = None

    def _event(self, buf):
        return type("E", (), {"current_buffer": buf})()

    def test_enter_submits_non_blank(self, bindings):
        buf = self.FakeBuffer("hi")
        self._handler(bindings, Keys.Enter)(self._event(buf))
        assert buf.submitted

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
    def test_enter_ignored_on_blank(self, bindings, blank):
        buf = self.FakeBuffer(blank)
        self._handler(bindings, Keys.Enter)(self._event(buf))
        assert not buf.submitted

    def test_alt_enter_inserts_newline(self, bindings):
        buf = self.FakeBuffer("line1")
        self._handler(bindings, Keys.Escape, Keys.Enter)(self._event(buf))
        assert buf.text == "line1\n" and not buf.submitted

    def test_escape_dismisses_open_menu(self, bindings):
        buf = self.FakeBuffer("/se", complete_state=object())
        self._handler(bindings, Keys.Escape)(self._event(buf))
        assert buf.cancelled

    def test_escape_without_menu_is_noop(self, bindings):
        buf = self.FakeBuffer("text")
        self._handler(bindings, Keys.Escape)(self._event(buf))
        assert not buf.cancelled and buf.text == "text"


class TestAdaptiveMenuHeight:
    """Renderer._min_available_height is replaced by a property computed from the live buffer."""

    class FakeApp:
        def __init__(self, buf):
            self.current_buffer = buf

    def _renderer(self):
        return type("R", (), {"_min_available_height": ca.Renderer._min_available_height})()

    def test_installed_on_renderer(self):
        assert isinstance(ca.Renderer.__dict__["_min_available_height"], property)

    def test_reserves_match_count_plus_margin(self, monkeypatch):
        state = type("S", (), {"completions": [1, 2, 3]})()
        buf = type("B", (), {"text": "/se", "complete_state": state})()
        monkeypatch.setattr(ca, "get_app", lambda: self.FakeApp(buf))
        assert self._renderer()._min_available_height == 3 + ca._MENU_ROW_MARGIN

    @pytest.mark.parametrize("text,state", [("hello", object()), ("/se", None)])
    def test_base_height_when_no_menu(self, monkeypatch, text, state):
        buf = type("B", (), {"text": text, "complete_state": state})()
        monkeypatch.setattr(ca, "get_app", lambda: self.FakeApp(buf))
        assert self._renderer()._min_available_height == ca._MIN_AVAILABLE_HEIGHT_BASE

    def test_falls_back_to_stored_value_without_an_app(self, monkeypatch):
        def no_app():
            raise RuntimeError("no app")
        monkeypatch.setattr(ca, "get_app", no_app)
        r = self._renderer()
        assert r._min_available_height == ca._MIN_AVAILABLE_HEIGHT_BASE
        r._min_available_height = 17
        assert r._min_available_height == 17

    def test_completing_condition_only_for_slash(self, monkeypatch):
        for text, expected in [("/x", True), ("x", False)]:
            buf = type("B", (), {"text": text})()
            monkeypatch.setattr(ca, "get_app", lambda b=buf: self.FakeApp(b))
            assert ca._completing_slash_command() is expected

    def test_text_change_restarts_completion_only_for_slash(self, monkeypatch):
        calls = []
        monkeypatch.setattr(ca, "_request_menu_resize_redraw", lambda *a: calls.append("redraw"))
        for text in ["/se", "hello"]:
            buf = type("B", (), {"text": text, "start_completion": lambda self: calls.append("complete")})()
            ca._on_prompt_text_changed(buf)
        assert calls == ["complete", "redraw", "redraw"]
