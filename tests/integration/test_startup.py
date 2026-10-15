"""Command-line startup: argument validation, model checks, --resume, --reset."""

import json

import pytest

from tests.fakes import reply
from tests.helpers import known_bug


def seed(memory_file, session, history):
    path = memory_file(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history))
    return path


class TestDefaultStartup:
    def test_banner_and_clean_exit_on_eof(self, run_app, fake_ollama):
        run = run_app([])
        assert run.exit_code is None
        assert "Chatting with qwen3.5:9b (session: default, 0 messages in memory)" in run.out
        assert "/exit" in run.out and "/session [name]" in run.out and "/model [name]" in run.out
        assert run.prompts == ["You[default]> "]
        assert fake_ollama.chat_calls == [] and fake_ollama.recap_calls == []

    def test_fresh_default_session_has_no_replay_or_recap(self, run_app, fake_ollama, project):
        run = run_app([])
        assert "recap" not in run.out
        assert not (project / "memory" / "default.json").exists()

    def test_existing_default_session_is_replayed_on_plain_start(self, run_app, memory_file, fake_ollama):
        seed(memory_file, "default", [{"role": "user", "content": "earlier q"},
                                      {"role": "assistant", "content": "earlier a"}])
        run = run_app([])
        assert "2 messages in memory" in run.out
        assert "You[default]> earlier q" in run.out and "● earlier a" in run.out
        assert "◆ recap:" in run.out


class TestModelValidation:
    def test_unknown_model_exits_with_guidance(self, run_app, fake_ollama):
        fake_ollama.models = ["llama3.2:3b", "qwen3.5:9b"]
        run = run_app([], argv=["--model", "mistral:7b"])
        assert run.exit_code == 1
        assert "Model 'mistral:7b' is not installed locally." in run.out
        assert "Installed models: llama3.2:3b, qwen3.5:9b" in run.out
        assert "ollama pull mistral:7b" in run.out
        assert run.prompts == []

    def test_no_models_installed(self, run_app, fake_ollama):
        fake_ollama.models = []
        run = run_app([], argv=["--model", "x"])
        assert run.exit_code == 1 and "(none — pull one first)" in run.out

    def test_installed_custom_model_used_for_requests(self, run_app, fake_ollama):
        fake_ollama.models = ["llama3.2:3b"]
        fake_ollama.script(reply("hi"))
        run = run_app(["hello"], argv=["--model", "llama3.2:3b"])
        assert "Chatting with llama3.2:3b" in run.out
        assert fake_ollama.chat_calls[0]["model"] == "llama3.2:3b"

    def test_unreachable_server_does_not_block_startup(self, run_app, fake_ollama):
        fake_ollama.list_error = ConnectionError("refused")
        run = run_app([], argv=["--model", "anything"])
        assert run.exit_code is None and "Chatting with anything" in run.out

    @known_bug("Model validation compares exact names, so '--model llama3.2' is rejected when 'llama3.2:latest' "
               "is installed even though Ollama itself resolves the implicit ':latest' tag.")
    def test_implicit_latest_tag_accepted(self, run_app, fake_ollama):
        fake_ollama.models = ["llama3.2:latest"]
        run = run_app([], argv=["--model", "llama3.2"])
        assert run.exit_code is None


class TestResume:
    def test_resume_existing_session_replays_and_recaps(self, run_app, memory_file, fake_ollama):
        seed(memory_file, "work", [{"role": "user", "content": "fix the parser", "ts": "t"},
                                   {"role": "assistant", "content": "Parser fixed.", "ts": "t"}])
        run = run_app([], argv=["--resume", "work"])
        assert "(session: work, 2 messages in memory)" in run.out
        assert run.out.index("You[work]> fix the parser") < run.out.index("● Parser fixed.") < run.out.index("◆ recap:")
        assert run.prompts == ["You[work]> "]
        (recap_req,) = fake_ollama.recap_calls
        assert all("ts" not in m for m in recap_req["messages"])

    def test_resume_missing_session_lists_existing(self, run_app, memory_file):
        seed(memory_file, "alpha", [])
        seed(memory_file, "beta", [])
        run = run_app([], argv=["--resume", "gamma"])
        assert run.exit_code == 1
        assert "No session named 'gamma' to resume." in run.out
        assert "Existing sessions: alpha, beta" in run.out

    def test_resume_with_no_sessions_at_all(self, run_app):
        run = run_app([], argv=["--resume", "gamma"])
        assert run.exit_code == 1 and "(none yet)" in run.out

    @pytest.mark.parametrize("bad", ["../etc/passwd", "has space", "a/b", "x;rm -rf ~"])
    def test_resume_rejects_unsafe_names_before_touching_disk(self, run_app, project, bad):
        run = run_app([], argv=["--resume", bad])
        assert run.exit_code == 1 and "is invalid" in run.out
        assert not (project / "memory").exists() or list((project / "memory").iterdir()) == []

    def test_recap_failure_is_silent(self, run_app, memory_file, fake_ollama):
        seed(memory_file, "work", [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
        fake_ollama.recap = ConnectionError("down")
        run = run_app([], argv=["--resume", "work"])
        assert run.exit_code is None and "recap" not in run.out and "Traceback" not in run.out

    @known_bug("A corrupt/truncated memory file (e.g. after a crash mid-save) raises JSONDecodeError at "
               "startup, so the app cannot be started for that session at all.",
               raises=json.JSONDecodeError)
    def test_corrupt_memory_file_does_not_crash(self, run_app, memory_file):
        memory_file("work").parent.mkdir(parents=True)
        memory_file("work").write_text('[{"role": "user", "content": "trunc')
        run_app([], argv=["--resume", "work"])


class TestReset:
    def test_reset_clears_default_session(self, run_app, memory_file):
        path = seed(memory_file, "default", [{"role": "user", "content": "old"}])
        run = run_app([], argv=["--reset"])
        assert "Cleared memory for session 'default'." in run.out
        assert "0 messages in memory" in run.out and not path.exists()

    def test_reset_with_resume_clears_only_that_session(self, run_app, memory_file):
        keep = seed(memory_file, "keep", [{"role": "user", "content": "k"}])
        wipe = seed(memory_file, "wipe", [{"role": "user", "content": "w"}])
        run = run_app([], argv=["--resume", "wipe", "--reset"])
        assert not wipe.exists() and keep.exists()
        assert "recap" not in run.out

    def test_reset_with_nothing_to_clear(self, run_app):
        assert run_app([], argv=["--reset"]).exit_code is None

    def test_unknown_flag_is_rejected(self, run_app):
        assert run_app([], argv=["--bogus"]).exit_code == 2
