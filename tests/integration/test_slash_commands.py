"""In-chat slash commands: /reset, /session(s), /rename, /model(s) — and session switching/replay."""

import json

import pytest

from tests.fakes import reply


def seed(memory_file, session, history):
    path = memory_file(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history))
    return path


def exchange(q, a):
    return [{"role": "user", "content": q, "ts": "t"}, {"role": "assistant", "content": a, "ts": "t"}]


def after_startup(monkeypatch, fake, models_or_error):
    """Let startup's model check see the default install, then change what ollama.list() returns."""
    from ollama import ListResponse
    calls = {"n": 0}
    real_list = fake.list

    def later_list():
        calls["n"] += 1
        if calls["n"] == 1:
            return real_list()
        if isinstance(models_or_error, Exception):
            raise models_or_error
        return ListResponse(models=[ListResponse.Model(model=m) for m in models_or_error])
    monkeypatch.setattr("ollama.list", later_list)


SLASH_INPUTS = ["/sessions", "/session", "/model", "/models", "/rename", "/reset"]


class TestSlashCommandsAreLocal:
    @pytest.mark.parametrize("cmd", SLASH_INPUTS)
    def test_never_sent_to_model_nor_persisted(self, run_app, fake_ollama, memory_file, cmd):
        seed(memory_file, "default", exchange("q", "a"))
        run = run_app([cmd])
        assert fake_ollama.chat_calls == []
        assert f"You[default]> {cmd}" in run.out  # still echoed as a grey block
        if cmd != "/reset":
            assert json.loads(memory_file("default").read_text()) == exchange("q", "a")


class TestReset:
    def test_reset_clears_memory_and_context(self, run_app, fake_ollama, memory_file):
        seed(memory_file, "default", exchange("old q", "old a"))
        fake_ollama.script(reply("fresh"))
        run = run_app(["/reset", "new question"])
        assert "● Memory cleared." in run.out
        assert fake_ollama.chat_calls[0]["messages"] == [{"role": "user", "content": "new question"}]
        assert [e["content"] for e in json.loads(memory_file("default").read_text())] == ["new question", "fresh"]

    def test_reset_only_affects_active_session(self, run_app, memory_file):
        other = seed(memory_file, "other", exchange("q", "a"))
        run_app(["/reset"])
        assert json.loads(other.read_text()) == exchange("q", "a")


class TestSessions:
    def test_lists_sessions_marking_active(self, run_app, memory_file):
        seed(memory_file, "alpha", [])
        seed(memory_file, "beta", [])
        run = run_app(["/sessions"])
        assert "● Sessions: alpha, beta, default (active)" in run.out

    def test_show_current_session(self, run_app, memory_file):
        seed(memory_file, "default", exchange("q", "a"))
        assert "● Current session: 'default' (2 messages)" in run_app(["/session"]).out

    def test_switch_to_new_session(self, run_app, fake_ollama, memory_file):
        fake_ollama.script(reply("hello from new"))
        run = run_app(["/session project-x", "hi"])
        assert "● Switched to session 'project-x' (0 messages in memory)" in run.out
        assert "Session: project-x" in run.out
        assert run.prompts[1:] == ["You[project-x]> ", "You[project-x]> "]
        assert [e["content"] for e in json.loads(memory_file("project-x").read_text())] == ["hi", "hello from new"]
        assert not memory_file("default").exists()

    def test_switch_isolates_context(self, run_app, fake_ollama, memory_file):
        seed(memory_file, "default", exchange("default q", "default a"))
        seed(memory_file, "other", exchange("other q", "other a"))
        fake_ollama.script(reply("x"))
        run_app(["/session other", "continue"])
        sent = [m["content"] for m in fake_ollama.chat_calls[0]["messages"]]
        assert sent == ["other q", "other a", "continue"]

    def test_switch_replays_history_and_recaps_once_per_run(self, run_app, fake_ollama, memory_file):
        seed(memory_file, "other", exchange("other question", "other answer"))
        run = run_app(["/session other", "/session default", "/session other"])
        assert run.out.count("You[other]> other question") == 1
        assert run.out.count("◆ recap:") == 1
        assert len(fake_ollama.recap_calls) == 1
        assert run.out.count("Session: other") == 2  # banner shown on every switch, even without replay

    def test_switching_to_empty_session_makes_no_recap_call(self, run_app, fake_ollama):
        run_app(["/session empty"])
        assert fake_ollama.recap_calls == []

    def test_switch_to_current_startup_session_does_not_replay_again(self, run_app, memory_file, fake_ollama):
        seed(memory_file, "default", exchange("q", "a"))
        run = run_app(["/session default"])
        assert run.out.count("You[default]> q") == 1
        assert len(fake_ollama.recap_calls) == 1

    @pytest.mark.parametrize("bad", ["../x", "has space", "a.b", "semi;colon"])
    def test_invalid_session_names_rejected(self, run_app, project, bad):
        run = run_app([f"/session {bad}"])
        assert f"Session name '{bad}' is invalid" in run.out
        assert run.prompts[-1] == "You[default]> "


class TestRename:
    def test_usage(self, run_app):
        assert "Usage: /rename <new_session_name>" in run_app(["/rename"]).out

    def test_renames_file_and_continues_under_new_name(self, run_app, fake_ollama, memory_file):
        seed(memory_file, "default", exchange("q", "a"))
        fake_ollama.script(reply("still here"))
        run = run_app(["/rename research", "next"])
        assert "● Renamed session 'default' to 'research'." in run.out
        assert not memory_file("default").exists()
        history = json.loads(memory_file("research").read_text())
        assert [e["content"] for e in history] == ["q", "a", "next", "still here"]
        assert run.prompts[-1] == "You[research]> "

    def test_rename_unsaved_session(self, run_app, memory_file):
        run = run_app(["/rename fresh"])
        assert "Renamed session 'default' to 'fresh'" in run.out
        assert not memory_file("fresh").exists()

    def test_refuses_to_clobber_existing_session(self, run_app, memory_file):
        seed(memory_file, "default", exchange("mine", "a"))
        taken = seed(memory_file, "taken", exchange("theirs", "b"))
        run = run_app(["/rename taken"])
        assert "A session named 'taken' already exists" in run.out
        assert json.loads(taken.read_text())[0]["content"] == "theirs"
        assert memory_file("default").exists()

    def test_same_name(self, run_app):
        assert "Session is already named 'default'." in run_app(["/rename default"]).out

    @pytest.mark.parametrize("bad", ["../escape", "with space", "a/b"])
    def test_invalid(self, run_app, project, bad):
        run = run_app([f"/rename {bad}"])
        assert "is invalid" in run.out
        assert not (project / "escape.json").exists()


class TestModelCommands:
    def test_show_current_model(self, run_app):
        assert "● Current model: 'qwen3.5:9b'" in run_app(["/model"]).out

    def test_list_models(self, run_app, fake_ollama):
        fake_ollama.models = ["b:1b", "a:7b"]
        fake_ollama.models.append("qwen3.5:9b")
        assert "● Available Models: a:7b, b:1b, qwen3.5:9b" in run_app(["/models"]).out

    def test_list_models_none_installed(self, run_app, fake_ollama, monkeypatch):
        after_startup(monkeypatch, fake_ollama, [])
        assert "● Available Models: (none installed)" in run_app(["/models"]).out

    def test_list_models_unreachable(self, run_app, fake_ollama, monkeypatch):
        after_startup(monkeypatch, fake_ollama, ConnectionError("gone"))
        assert "Could not reach the Ollama server to list models." in run_app(["/models"]).out

    def test_switch_model_applies_to_next_request(self, run_app, fake_ollama):
        fake_ollama.models = ["qwen3.5:9b", "llama3.2:3b"]
        fake_ollama.script(reply("from llama"))
        run = run_app(["/model llama3.2:3b", "hi"])
        assert "● Switched to model 'llama3.2:3b'." in run.out
        assert fake_ollama.chat_calls[0]["model"] == "llama3.2:3b"

    def test_switch_to_missing_model_is_refused(self, run_app, fake_ollama):
        fake_ollama.script(reply("x"))
        run = run_app(["/model ghost:1b", "hi"])
        assert "Model 'ghost:1b' is not installed locally." in run.out
        assert "ollama pull ghost:1b" in run.out
        assert fake_ollama.chat_calls[0]["model"] == "qwen3.5:9b"

    def test_switch_allowed_when_server_unreachable(self, run_app, fake_ollama, monkeypatch):
        after_startup(monkeypatch, fake_ollama, ConnectionError("gone"))
        assert "Switched to model 'anything'" in run_app(["/model anything"]).out

    def test_recap_uses_active_model(self, run_app, fake_ollama, memory_file):
        fake_ollama.models = ["qwen3.5:9b", "llama3.2:3b"]
        seed(memory_file, "other", exchange("q", "a"))
        run_app(["/model llama3.2:3b", "/session other"])
        assert fake_ollama.recap_calls[0]["model"] == "llama3.2:3b"
