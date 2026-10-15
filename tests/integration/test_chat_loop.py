"""The agent loop end to end (in-process): turns, persistence, tool rounds, approvals, errors, interrupts."""

import json

import pytest

import chat_agent
import tools
from tests.fakes import Turn, call, reply, tool_round
from tests.helpers import known_bug, write


def saved(memory_file, session="default"):
    return json.loads(memory_file(session).read_text())


def roles(history):
    return [e["role"] for e in history]


class TestBasicTurns:
    def test_single_turn_streams_reply_and_persists(self, run_app, fake_ollama, memory_file):
        fake_ollama.script(reply("Hi! How can I help?"))
        run = run_app(["hello"])
        assert "You[default]> hello" in run.out and "● Hi! How can I help?" in run.out
        history = saved(memory_file)
        assert [(e["role"], e["content"]) for e in history] == [("user", "hello"), ("assistant", "Hi! How can I help?")]
        assert all("ts" in e for e in history)

    def test_request_carries_tools_and_strips_local_fields(self, run_app, fake_ollama):
        fake_ollama.script(reply("ok"))
        run_app(["hello"])
        (req,) = fake_ollama.chat_calls
        assert req["tools"] == tools.tool_schemas()
        assert req["messages"] == [{"role": "user", "content": "hello"}]
        assert req["stream"] is True and req["options"] == {"num_ctx": chat_agent.NUM_CTX}

    def test_context_accumulates_across_turns(self, run_app, fake_ollama):
        fake_ollama.script(reply("one"), reply("two"))
        run_app(["first", "second"])
        second = fake_ollama.chat_calls[1]["messages"]
        assert [(m["role"], m["content"]) for m in second] == [
            ("user", "first"), ("assistant", "one"), ("user", "second")]

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_input_is_ignored(self, run_app, fake_ollama, blank):
        run = run_app([blank])
        assert fake_ollama.chat_calls == [] and len(run.prompts) == 2

    def test_input_is_trimmed(self, run_app, fake_ollama, memory_file):
        fake_ollama.script(reply("ok"))
        run_app(["   padded   "])
        assert saved(memory_file)[0]["content"] == "padded"

    def test_empty_model_reply_gets_placeholder(self, run_app, fake_ollama):
        fake_ollama.script(reply(""))
        assert "model returned an empty response — try again or rephrase" in run_app(["hi"]).out

    def test_history_file_is_trimmed_to_cap(self, run_app, fake_ollama, memory_file):
        n = chat_agent.MAX_HISTORY_MESSAGES
        fake_ollama.script(*[reply(f"a{i}") for i in range(n)])
        run_app([f"q{i}" for i in range(n)])
        history = saved(memory_file)
        assert len(history) == n and history[-1]["content"] == f"a{n - 1}"
        assert len(fake_ollama.chat_calls[-1]["messages"]) <= n


class TestExit:
    @pytest.mark.parametrize("word", ["/exit", "/quit", "/bye", "/EXIT", "/Bye"])
    def test_exit_commands(self, run_app, fake_ollama, memory_file, word):
        run = run_app([word, "never reached"])
        assert "Goodbye! See you next time." in run.out
        assert len(run.prompts) == 1 and fake_ollama.chat_calls == []
        assert not memory_file("default").exists()

    @pytest.mark.parametrize("word", ["exit", "quit", "bye", "Bye!"])
    def test_bare_words_are_ordinary_chat(self, run_app, fake_ollama, word):
        fake_ollama.script(reply("Goodbye to you too"))
        run = run_app([word])
        assert "Goodbye! See you next time." not in run.out
        assert fake_ollama.chat_calls[0]["messages"][-1]["content"] == word

    def test_ctrl_c_at_prompt_exits_cleanly(self, run_app, fake_ollama):
        run = run_app([KeyboardInterrupt, "never reached"])
        assert run.exit_code is None and len(run.prompts) == 1 and fake_ollama.chat_calls == []


class TestToolLoop:
    def test_read_only_tool_round_then_answer(self, run_app, fake_ollama, project, memory_file):
        write(project, "notes.txt", "the secret word is pineapple\n")
        fake_ollama.script(tool_round(call("read_file", path="notes.txt")), reply("The word is pineapple."))
        run = run_app(["what's in notes.txt?"])

        assert "● Tools (round 1, 1 tool call)" in run.out
        assert 'read_file(path="notes.txt")' in run.out and "✓ done" in run.out
        assert "● The word is pineapple." in run.out

        second_request = fake_ollama.chat_calls[1]["messages"]
        assert roles(second_request) == ["user", "assistant", "tool"]
        tool_msg = second_request[2]
        assert tool_msg["tool_name"] == "read_file"
        assert "pineapple" in json.loads(tool_msg["content"])["content"]
        assert roles(saved(memory_file)) == ["user", "assistant", "tool", "assistant"]

        log = [json.loads(l) for l in (project / "logs" / "tool_calls.jsonl").read_text().splitlines()]
        assert [(e["tool"], e["approval"], e["outcome"]) for e in log] == [("read_file", "auto", "ok")]

    def test_multiple_calls_in_one_round_run_in_order(self, run_app, fake_ollama, project):
        write(project, "a.txt", "A")
        write(project, "b.txt", "B")
        fake_ollama.script(tool_round(call("read_file", path="a.txt"), call("read_file", path="b.txt")),
                           reply("both read"))
        run = run_app(["read both"])
        assert "(round 1, 2 tool calls)" in run.out
        tool_msgs = [m for m in fake_ollama.chat_calls[1]["messages"] if m["role"] == "tool"]
        assert [json.loads(m["content"])["content"].split("\t")[1] for m in tool_msgs] == ["A", "B"]

    def test_tool_errors_are_fed_back_to_the_model(self, run_app, fake_ollama):
        fake_ollama.script(tool_round(call("read_file", path="../../etc/passwd"), call("teleport")),
                           reply("I can't do that."))
        run = run_app(["read /etc/passwd"])
        results = [json.loads(m["content"]) for m in fake_ollama.chat_calls[1]["messages"] if m["role"] == "tool"]
        assert all(r["ok"] is False for r in results)
        assert "outside the project sandbox" in results[0]["error"]
        assert results[1]["error"] == "Unknown tool 'teleport'."
        assert run.out.count("✗") == 2

    def test_approved_write_creates_file(self, run_app, fake_ollama, project):
        fake_ollama.script(tool_round(call("write_file", path="hello.py", content="print('hi')\n")),
                           reply("Created hello.py."))
        run = run_app(["make hello.py"], approve=[True])
        assert (project / "hello.py").read_text() == "print('hi')\n"
        assert run.confirmations == ["● Apply this write_file to hello.py? [y/N]: "]
        assert "Proposed write_file: hello.py (create new file · 1 line · 12 bytes)" in run.out
        result = json.loads(fake_ollama.chat_calls[1]["messages"][-1]["content"])
        assert result["ok"] is True and result["created_new"] is True

    def test_declined_edit_leaves_file_and_tells_model(self, run_app, fake_ollama, project):
        write(project, "cfg.ini", "debug = false\n")
        fake_ollama.script(tool_round(call("edit_file", path="cfg.ini", old_string="false", new_string="true")),
                           reply("Okay, I left it alone."))
        run = run_app(["enable debug"], approve=[False])
        assert (project / "cfg.ini").read_text() == "debug = false\n"
        assert "⊘ declined by user" in run.out
        result = json.loads(fake_ollama.chat_calls[1]["messages"][-1]["content"])
        assert result["declined"] is True and "Do not attempt it again" in result["error"]

    def test_each_mutation_is_confirmed_individually(self, run_app, fake_ollama, project):
        fake_ollama.script(tool_round(call("write_file", path="a.txt", content="a"),
                                      call("write_file", path="b.txt", content="b")),
                           reply("done"))
        run = run_app(["write two files"], approve=[True, False])
        assert len(run.confirmations) == 2
        assert (project / "a.txt").exists() and not (project / "b.txt").exists()

    def test_multi_round_agentic_task(self, run_app, fake_ollama, project):
        write(project, "app.py", "VERSION = '1.0'\n")
        fake_ollama.script(
            tool_round(call("search_file", pattern="VERSION")),
            tool_round(call("read_file", path="app.py")),
            tool_round(call("edit_file", path="app.py", old_string="'1.0'", new_string="'1.1'")),
            reply("Bumped VERSION to 1.1."),
        )
        run = run_app(["bump the version"], approve=[True])
        assert (project / "app.py").read_text() == "VERSION = '1.1'\n"
        assert [f"(round {i}," in run.out for i in (1, 2, 3)] == [True] * 3
        assert len(fake_ollama.chat_calls) == 4

    def test_tool_round_without_closing_summary(self, run_app, fake_ollama, project):
        fake_ollama.script(tool_round(call("list_files")), reply(""))
        run = run_app(["list"])
        assert "no closing summary from the model, but the action(s) above completed" in run.out
        assert "try again or rephrase" not in run.out

    def test_round_cap_forces_a_final_answer(self, run_app, fake_ollama):
        cap = chat_agent.MAX_TOOL_ROUNDS
        fake_ollama.script(*[tool_round(call("list_files")) for _ in range(cap)],
                           tool_round(call("list_files"), content="Final answer anyway."))
        run = run_app(["loop forever"])
        assert len(fake_ollama.chat_calls) == cap + 1
        assert all(c["tools"] is not None for c in fake_ollama.chat_calls[:cap])
        assert fake_ollama.chat_calls[cap]["tools"] is None
        assert f"(round {cap + 1}," not in run.out  # tool calls on the forced round are not executed
        assert "Final answer anyway." in run.out

    def test_history_saved_after_each_tool_round(self, run_app, fake_ollama, memory_file, monkeypatch):
        """A crash between rounds must not lose tool results already obtained."""
        snapshots = []
        real_save = chat_agent.save_history
        monkeypatch.setattr(chat_agent, "save_history",
                            lambda s, h: (snapshots.append(roles(h)), real_save(s, h)))
        fake_ollama.script(tool_round(call("list_files")), reply("done"))
        run_app(["go"])
        assert ["user", "assistant", "tool"] in snapshots

    @known_bug("A model tool call whose preview raises a non-ToolError (here: write_file onto a directory) "
               "propagates out of main() and kills the session with a traceback.",
               raises=IsADirectoryError)
    def test_session_survives_bad_tool_arguments(self, run_app, fake_ollama, project):
        (project / "src").mkdir()
        fake_ollama.script(tool_round(call("write_file", path="src", content="x")), reply("sorry"))
        run = run_app(["overwrite src"])
        assert "sorry" in run.out


class TestErrorsAndInterrupts:
    def test_connection_error_is_reported_and_session_continues(self, run_app, fake_ollama, memory_file):
        fake_ollama.script(Turn(error=ConnectionError("Failed to connect to Ollama")), reply("back online"))
        run = run_app(["first", "second"])
        assert "[error talking to ollama: Failed to connect to Ollama]" in run.out
        assert "● back online" in run.out
        assert len(run.prompts) == 3

    def test_mid_stream_failure_keeps_partial_on_screen(self, run_app, fake_ollama):
        fake_ollama.script(Turn(content="partial line\nrest\n", error=RuntimeError("EOF"), error_after_chunks=1))
        run = run_app(["go"])
        assert "partial line" in run.out and "[error talking to ollama: EOF]" in run.out

    def test_interrupt_rolls_back_entire_turn(self, run_app, fake_ollama, memory_file):
        fake_ollama.script(reply("kept"), Turn(interrupt=True))
        run = run_app(["keep me", "cancel me"])
        assert "[cancelled]" in run.out
        assert [e["content"] for e in saved(memory_file)] == ["keep me", "kept"]

    def test_interrupt_after_tool_round_rolls_back_history_but_not_disk(self, run_app, fake_ollama, project, memory_file):
        fake_ollama.script(tool_round(call("write_file", path="done.txt", content="x")), Turn(interrupt=True))
        run_app(["write then cancel"], approve=[True])
        assert (project / "done.txt").exists()  # the approved side effect already happened
        assert not memory_file("default").exists() or saved(memory_file) == []

    def test_conversation_continues_after_interrupt(self, run_app, fake_ollama):
        fake_ollama.script(Turn(interrupt=True), reply("fresh answer"))
        run = run_app(["cancel me", "try again"])
        assert "● fresh answer" in run.out
        assert [m["content"] for m in fake_ollama.chat_calls[1]["messages"]] == ["try again"]
