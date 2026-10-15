"""End-to-end: the real app in a real pseudo-terminal, talking to a fake Ollama over real HTTP.

These cover what in-process tests cannot: prompt_toolkit's raw-mode input, key bindings, the live
Esc/Ctrl+C handling during streaming, the y/N prompt on a real TTY, and process exit behavior.
"""

import json
import socket
import time

from tests.e2e.pty_app import KEY_ALT_ENTER, KEY_CTRL_C, KEY_ENTER, KEY_ESC
from tests.fakes import Turn, call, reply, tool_round
from tests.helpers import known_bug


def memory(app_dir, session="default"):
    return json.loads((app_dir / "memory" / f"{session}.json").read_text())


def log(app_dir):
    return [json.loads(l) for l in (app_dir / "logs" / "tool_calls.jsonl").read_text().splitlines()]


class TestConversation:
    def test_chat_turn_streams_persists_and_exits(self, launch, server, app_dir):
        server.script(reply("Hi there! I can help with your project.", chunk_size=4))
        app = launch()
        app.expect("Chatting with qwen3.5:9b (session: default, 0 messages in memory)")
        app.expect_prompt()
        app.submit("hello")
        app.expect("● Hi there! I can help with your project.")
        app.expect_prompt()
        assert [(e["role"], e["content"]) for e in memory(app_dir)] == [
            ("user", "hello"), ("assistant", "Hi there! I can help with your project.")]
        app.submit("/exit")
        app.expect("Goodbye! See you next time.")
        assert app.wait_exit() == 0
        assert "Traceback" not in app.transcript

    def test_submitted_prompt_becomes_grey_block(self, launch, server):
        server.script(reply("ok"))
        app = launch()
        app.expect_prompt()
        app.submit("remember me")
        app.expect("● ok")
        screen = app.wait_screen(lambda s: "You[default]> remember me" in s)
        assert "You[default]> remember me" in screen

    def test_alt_enter_inserts_newline_instead_of_submitting(self, launch, server):
        server.script(reply("got both lines"))
        app = launch()
        app.expect_prompt()
        app.type("line one")
        app.send(KEY_ALT_ENTER, delay=0.1)
        app.type("line two")
        app.send(KEY_ENTER)
        app.expect("● got both lines")
        (req,) = server.chat_requests
        assert req["messages"][-1]["content"] == "line one\nline two"

    def test_enter_on_blank_prompt_does_nothing(self, launch, server):
        server.script(reply("only once"))
        app = launch()
        app.expect_prompt()
        app.send(KEY_ENTER, delay=0.3)
        app.send("   " + KEY_ENTER, delay=0.3)
        assert server.chat_requests == []
        app.submit("now")
        app.expect("● only once")
        assert len(server.chat_requests) == 1

    def test_ollama_unreachable_is_reported_and_app_keeps_running(self, launch):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            dead_port = s.getsockname()[1]
        app = launch(host=f"http://127.0.0.1:{dead_port}")
        app.expect_prompt()
        app.submit("anyone there?")
        app.expect("[error talking to ollama:")
        app.expect_prompt()
        app.submit("/bye")
        assert app.wait_exit() == 0


class TestStreamingDisplay:
    def test_multi_line_reply_appears_progressively(self, launch, server):
        server.script(reply("".join(f"line {i}\n" for i in range(40)), chunk_delay=0.1))
        app = launch()
        app.expect_prompt()
        app.submit("count")
        t0 = time.monotonic()
        app.expect("line 0")
        assert time.monotonic() - t0 < 2.0, "first line should show long before the 4s stream finishes"
        app.expect("line 39", timeout=15)

    @known_bug("Streaming is line-buffered: a reply paragraph without a newline (the common case for prose) "
               "shows only the 'Thinking...' spinner until the whole paragraph has been generated.")
    def test_single_long_paragraph_appears_progressively(self, launch, server):
        server.script(reply("word " * 120, chunk_size=10, chunk_delay=0.05))  # ~3s to generate
        app = launch()
        app.expect_prompt()
        app.submit("talk")
        app.expect("● word", timeout=1.5)


class TestInterrupts:
    def test_esc_mid_stream_cancels_rolls_back_and_restores_input(self, launch, server, app_dir):
        story = "Once upon a time\n" + "and then something else happened\n" * 60
        server.script(reply(story, chunk_delay=0.05), reply("short answer"))
        app = launch()
        app.expect_prompt()
        app.submit("tell me a story")
        app.expect("Once upon")
        app.send(KEY_ESC)
        app.expect("[cancelled]")
        app.expect_prompt()
        # The cancelled message is restored into the input box, ready to edit/resubmit.
        app.wait_screen(lambda s: "[cancelled]" in s
                        and "You[default]> tell me a story" in s.rsplit("[cancelled]", 1)[1])
        assert not (app_dir / "memory" / "default.json").exists() or memory(app_dir) == []
        app.send(KEY_ENTER)
        app.expect("● short answer")
        second = server.wait_for_requests(2)[1]
        assert [m["content"] for m in second["messages"]] == ["tell me a story"]

    @known_bug("Cancelling only stops the *display*: the producer thread keeps reading the HTTP stream to the "
               "end, so Ollama keeps generating the abandoned reply (burning GPU, and with OLLAMA_NUM_PARALLEL=1 "
               "the user's next message queues behind it).")
    def test_cancel_closes_the_model_stream(self, launch, server):
        server.script(reply("generating\n" * 100, chunk_delay=0.04))  # ~4s if read to the end
        app = launch()
        app.expect_prompt()
        app.submit("go")
        app.expect("generating")
        app.send(KEY_ESC)
        app.expect("[cancelled]")
        time.sleep(1.0)
        (stream,) = server.streams
        # The server notices a closed connection on its next write, so a closed stream ends right away;
        # one still being read would still be mid-flight here with ~2-3s left to go.
        assert stream["finished"] is not None and stream["chunks_sent"] < 100, stream

    def test_esc_while_thinking_cancels(self, launch, server):
        server.script(Turn(content="late reply", chunk_delay=3.0))
        app = launch()
        app.expect_prompt()
        app.submit("slow question")
        app.expect("Thinking...")
        app.send(KEY_ESC)
        app.expect("[cancelled]", timeout=3)
        app.expect_prompt()

    def test_ctrl_c_mid_stream_cancels_without_exiting(self, launch, server):
        server.script(reply("word word word\n" * 100, chunk_delay=0.05), reply("still alive"))
        app = launch()
        app.expect_prompt()
        app.submit("go")
        app.expect("word word")
        app.send(KEY_CTRL_C)
        app.expect("[cancelled]")
        app.expect_prompt()
        app.send(KEY_ENTER)
        app.expect("● still alive")
        assert app.proc.poll() is None

    def test_ctrl_c_at_idle_prompt_exits_cleanly(self, launch):
        app = launch()
        app.expect_prompt()
        app.send(KEY_CTRL_C)
        assert app.wait_exit() == 0
        assert "Traceback" not in app.transcript


class TestToolConfirmation:
    def test_approve_write_on_real_tty(self, launch, server, app_dir):
        server.script(tool_round(call("write_file", path="hello.py", content="print('hello')\n")),
                      reply("Created hello.py for you."))
        app = launch()
        app.expect_prompt()
        app.submit("create hello.py")
        app.expect("Proposed write_file: hello.py (create new file · 1 line · 15 bytes)")
        app.expect("1 + print('hello')")
        app.expect("Apply this write_file to hello.py? [y/N]:")
        app.send("y" + KEY_ENTER)
        app.expect("✓ done")
        app.expect("● Created hello.py for you.")
        assert (app_dir / "hello.py").read_text() == "print('hello')\n"
        assert [(e["tool"], e["approval"], e["outcome"]) for e in log(app_dir)] == [
            ("write_file", "user_approved", "ok")]

    def test_enter_alone_declines(self, launch, server, app_dir):
        (app_dir / "config.txt").write_text("safe\n")
        server.script(tool_round(call("edit_file", path="config.txt", old_string="safe", new_string="unsafe")),
                      reply("Understood, no changes."))
        app = launch()
        app.expect_prompt()
        app.submit("make it unsafe")
        app.expect("[y/N]:")
        app.send(KEY_ENTER)
        app.expect("⊘ declined by user")
        app.expect("● Understood, no changes.")
        assert (app_dir / "config.txt").read_text() == "safe\n"
        result = json.loads(server.wait_for_requests(2)[1]["messages"][-1]["content"])
        assert result["declined"] is True

    def test_keys_typed_during_preview_do_not_answer_the_prompt(self, launch, server, app_dir):
        """Type-ahead during the diff stream is flushed, so a stray 'y' can't pre-approve a write."""
        content = "".join(f"line {i}\n" for i in range(60))
        server.script(tool_round(call("write_file", path="big.txt", content=content)), reply("ok"))
        app = launch()
        app.expect_prompt()
        app.submit("write big.txt")
        app.expect("Proposed write_file")
        app.send("y" + KEY_ENTER)  # typed while the preview is still streaming
        app.expect("[y/N]:", timeout=30)
        app.send(KEY_ENTER)  # the real answer: decline
        app.expect("⊘ declined by user")
        assert not (app_dir / "big.txt").exists()

    def test_esc_fast_forwards_long_diff_preview(self, launch, server, app_dir):
        content = "".join(f"generated line number {i}\n" for i in range(400))  # ~12s at normal pace
        server.script(tool_round(call("write_file", path="gen.txt", content=content)), reply("done"))
        app = launch()
        app.expect_prompt()
        app.submit("generate")
        app.expect("Proposed write_file")
        app.send(KEY_ESC)
        app.expect("[y/N]:", timeout=3)
        app.send("yes" + KEY_ENTER)
        app.expect("✓ done")
        assert (app_dir / "gen.txt").read_text() == content

    @known_bug("Ctrl+C while a diff preview is streaming raises KeyboardInterrupt out of _print_streaming "
               "and kills the app with a traceback instead of fast-forwarding / declining.")
    def test_ctrl_c_during_diff_preview_does_not_crash(self, launch, server):
        content = "".join(f"generated line number {i}\n" for i in range(400))
        server.script(tool_round(call("write_file", path="gen.txt", content=content)), reply("ok"))
        app = launch()
        app.expect_prompt()
        app.submit("generate")
        app.expect("Proposed write_file")
        app.send(KEY_CTRL_C)
        app.expect("[y/N]:", timeout=5)
        assert "Traceback" not in app.transcript


class TestSessionsOnTerminal:
    def test_resume_replays_history_then_recap_lands_at_next_prompt(self, launch, server, app_dir):
        (app_dir / "memory").mkdir()
        (app_dir / "memory" / "work.json").write_text(json.dumps([
            {"role": "user", "content": "refactor the parser"},
            {"role": "assistant", "content": "Parser refactored into three functions."}]))
        app = launch("--resume", "work")
        app.expect("You[work]> refactor the parser")
        app.expect("● Parser refactored into three functions.")
        app.expect_prompt("work")
        app.submit("/sessions")
        app.expect("Sessions: work (active)")
        app.expect("◆ recap: You discussed the test project")  # drained at the loop boundary, never mid-box
        assert len(server.recap_requests) == 1

    def test_slash_menu_shows_matching_commands(self, launch):
        app = launch()
        app.expect_prompt()
        app.type("/ses")
        screen = app.wait_screen(lambda s: "List all sessions" in s and "Switch or show the active session" in s)
        assert "Rename the active session" not in screen
        app.send(KEY_ESC, delay=0.6)  # dismiss menu (Esc waits for a possible escape sequence)
        app.wait_screen(lambda s: "List all sessions" not in s)
