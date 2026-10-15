"""Talking to the model: streamed reveal, spinner calls, resume recap, and history replay.

These run with real background threads (only Ollama itself is faked) because thread/queue hand-off is
exactly what call_with_streaming_reply and call_with_spinner exist to get right.
"""

import queue
import time

import pytest
from ollama import Message

import chat_agent as ca
from tests.fakes import RECAP_TEXT, Turn, call, reply, tool_round
from tests.helpers import known_bug, strip_ansi


def assistant(content="", tool_calls=None):
    e = {"role": "assistant", "content": content}
    if tool_calls:
        e["tool_calls"] = tool_calls
    return e


def user(content):
    return {"role": "user", "content": content, "ts": "2026-01-01T00:00:00"}


# ------------------------------------------------------------------------------------------------
# call_with_streaming_reply
# ------------------------------------------------------------------------------------------------
class TestStreamingReply:
    def test_returns_full_message_and_prints_formatted_reply(self, fake_ollama, capsys):
        fake_ollama.script(reply("Hello **there**\nsecond line\n", chunk_size=3))
        result = ca.call_with_streaming_reply("m", [user("hi")], None)
        msg = result["message"]
        assert isinstance(msg, Message)
        assert (msg.role, msg.content, msg.tool_calls) == ("assistant", "Hello **there**\nsecond line\n", None)
        out = strip_ansi(capsys.readouterr().out)
        assert "● Hello there\n  second line\n" in out

    def test_request_shape(self, fake_ollama, capsys):
        fake_ollama.script(reply("ok"))
        schemas = [{"type": "function", "function": {"name": "x"}}]
        ca.call_with_streaming_reply("my-model", [user("hi")], schemas)
        (req,) = fake_ollama.chat_calls
        assert req["model"] == "my-model" and req["stream"] is True and req["tools"] == schemas
        assert req["options"] == {"num_ctx": ca.NUM_CTX}

    def test_markup_split_across_chunks_still_formats(self, fake_ollama, capsys):
        fake_ollama.script(reply("a **bold** b `code` c", chunk_size=1))
        ca.call_with_streaming_reply("m", [user("x")], None)
        raw = capsys.readouterr().out
        assert "\033[1mbold\033[0m" in raw and "**" not in raw

    def test_final_line_without_newline_is_shown(self, fake_ollama, capsys):
        fake_ollama.script(reply("no trailing newline", chunk_size=4))
        ca.call_with_streaming_reply("m", [user("x")], None)
        assert "● no trailing newline" in strip_ansi(capsys.readouterr().out)

    def test_code_block_and_blank_line_collapse(self, fake_ollama, capsys):
        fake_ollama.script(reply("intro\n\n\n\n```\ncode()\n```\nend"))
        ca.call_with_streaming_reply("m", [user("x")], None)
        assert strip_ansi(capsys.readouterr().out).split("\n")[:4] == ["● intro", "", "      code()", "  end"]

    def test_long_lines_are_word_wrapped_with_hanging_indent(self, fake_ollama, capsys, monkeypatch):
        monkeypatch.setenv("COLUMNS", "30")
        fake_ollama.script(reply("word " * 20))
        ca.call_with_streaming_reply("m", [user("x")], None)
        lines = [l for l in strip_ansi(capsys.readouterr().out).split("\n") if l.strip()]
        assert lines[0].startswith("● ") and all(l.startswith("  ") for l in lines[1:])
        assert all(len(l) <= 30 for l in lines)

    def test_tool_call_round_prints_no_reply_text(self, fake_ollama, capsys):
        fake_ollama.script(tool_round(call("list_files"), call("read_file", path="a")))
        msg = ca.call_with_streaming_reply("m", [user("x")], [])["message"]
        assert [c.function.name for c in msg.tool_calls] == ["list_files", "read_file"]
        assert msg.content is None
        assert "●" not in strip_ansi(capsys.readouterr().out)

    def test_text_and_tool_calls_in_same_round(self, fake_ollama, capsys):
        fake_ollama.script(tool_round(call("list_files"), content="Let me look.\n"))
        msg = ca.call_with_streaming_reply("m", [user("x")], [])["message"]
        assert msg.content == "Let me look.\n" and len(msg.tool_calls) == 1
        assert "● Let me look." in strip_ansi(capsys.readouterr().out)

    @known_bug("Each streamed chunk's tool_calls *replace* the previous ones instead of accumulating, so if "
               "Ollama streams parallel tool calls in separate chunks only the last call survives.")
    def test_tool_calls_split_across_chunks_are_all_kept(self, fake_ollama, capsys):
        fake_ollama.script(tool_round(call("list_files"), call("read_file", path="a"), split_tool_calls=True))
        msg = ca.call_with_streaming_reply("m", [user("x")], [])["message"]
        assert [c.function.name for c in msg.tool_calls] == ["list_files", "read_file"]

    @known_bug("The live streaming path never applies EMOJI_PATTERN, so emoji appear in live replies but are "
               "stripped from the same reply on session replay (format_agent_reply).")
    def test_live_stream_strips_emoji_like_replay(self, fake_ollama, capsys):
        fake_ollama.script(reply("Done ✅ ship it 🚀"))
        ca.call_with_streaming_reply("m", [user("x")], None)
        out = capsys.readouterr().out
        assert "✅" not in out and "🚀" not in out

    def test_error_before_any_output_is_raised(self, fake_ollama, capsys):
        fake_ollama.script(Turn(error=ConnectionError("connection refused")))
        with pytest.raises(ConnectionError, match="refused"):
            ca.call_with_streaming_reply("m", [user("x")], None)
        assert "\033[?25h" in capsys.readouterr().out  # cursor restored

    def test_error_mid_stream_shows_partial_then_raises(self, fake_ollama, capsys):
        fake_ollama.script(Turn(content="first line\nsecond line\n", error=RuntimeError("stream dropped"),
                                error_after_chunks=1))
        with pytest.raises(RuntimeError, match="stream dropped"):
            ca.call_with_streaming_reply("m", [user("x")], None)
        out = strip_ansi(capsys.readouterr().out)
        assert "first line" in out and "second line" not in out

    def test_empty_reply(self, fake_ollama, capsys):
        fake_ollama.script(reply(""))
        msg = ca.call_with_streaming_reply("m", [user("x")], None)["message"]
        assert msg.content is None and not msg.tool_calls

    def test_spinner_shown_while_waiting_for_first_token(self, monkeypatch, capsys):
        def slow_chat(**kw):
            time.sleep(0.35)
            yield from ()
        monkeypatch.setattr(ca.ollama, "chat", lambda **kw: slow_chat(**kw))
        ca.call_with_streaming_reply("m", [user("x")], None)
        assert "Thinking... (0s)" in capsys.readouterr().out

    def test_interrupt_during_reveal_restores_terminal_state(self, fake_ollama, capsys, monkeypatch):
        """Esc/Ctrl+C mid-reveal: the partial line is ended cleanly and the cursor is re-shown."""
        fake_ollama.script(reply("x" * 200))
        real_sleep = time.sleep
        ticks = {"n": 0}

        def sleep(s):
            ticks["n"] += 1
            if ticks["n"] == 80:  # past the ~30 raw chars of the colored "● " prefix
                raise KeyboardInterrupt
            real_sleep(s)
        monkeypatch.setattr(ca.time, "sleep", sleep)
        interrupted = False
        try:
            ca.call_with_streaming_reply("m", [user("x")], None)
        except KeyboardInterrupt:
            interrupted = True
        assert interrupted, "interrupt must propagate so main() can roll the turn back"
        out = capsys.readouterr().out
        assert out.endswith("\n\033[?25h")
        assert 0 < strip_ansi(out).count("x") < 200

    def test_reveal_is_paced_not_dumped(self, fake_ollama, capsys, monkeypatch):
        """~1 char per tick: the reveal must take one sleep per visible character."""
        fake_ollama.script(reply("abcdefghij"))
        sleeps = []
        monkeypatch.setattr(ca.time, "sleep", lambda s: sleeps.append(s))
        ca.call_with_streaming_reply("m", [user("x")], None)
        assert len(sleeps) >= len("● abcdefghij")


# ------------------------------------------------------------------------------------------------
# call_with_spinner (non-streaming)
# ------------------------------------------------------------------------------------------------
class TestSpinnerCall:
    def test_returns_response(self, fake_ollama, capsys):
        fake_ollama.script(tool_round(call("list_files"), content="hi"))
        resp = ca.call_with_spinner("m", [user("x")], [{"t": 1}])
        assert resp["message"].content == "hi"
        req = fake_ollama.chat_calls[0]
        assert req["stream"] is False and req["tools"] == [{"t": 1}] and req["options"] == {"num_ctx": ca.NUM_CTX}

    def test_error_reraised_and_line_cleared(self, fake_ollama, capsys):
        fake_ollama.script(Turn(error=TimeoutError("slow")))
        with pytest.raises(TimeoutError):
            ca.call_with_spinner("m", [user("x")], None)
        out = capsys.readouterr().out
        assert out.endswith("\r\033[K") and "\033[?25h" in out

    def test_elapsed_timer_ticks(self, monkeypatch, capsys):
        def slow(**kw):
            time.sleep(0.3)
            return {"message": Message(role="assistant", content="ok")}
        monkeypatch.setattr(ca.ollama, "chat", slow)
        ca.call_with_spinner("m", [], None)
        assert capsys.readouterr().out.count("Thinking... (0s)") >= 2


# ------------------------------------------------------------------------------------------------
# Resume recap
# ------------------------------------------------------------------------------------------------
class TestRecap:
    def test_empty_history_makes_no_call(self, fake_ollama):
        assert ca.generate_recap("m", []) is None
        assert fake_ollama.recap_calls == []

    def test_sends_history_without_timestamps_plus_instruction(self, fake_ollama):
        fake_ollama.recap = f"  {RECAP_TEXT}\n"
        assert ca.generate_recap("m", [user("hello"), assistant("hi")]) == RECAP_TEXT
        (req,) = fake_ollama.recap_calls
        assert req["tools"] is None and req["stream"] is False
        assert req["messages"] == [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"},
                                   {"role": "user", "content": ca.RECAP_INSTRUCTION}]

    def test_failure_returns_none(self, fake_ollama):
        fake_ollama.recap = ConnectionError("down")
        assert ca.generate_recap("m", [user("x")]) is None

    def test_blank_recap_returns_none(self, fake_ollama):
        fake_ollama.recap = "   "
        assert ca.generate_recap("m", [user("x")]) is None

    @known_bug("If the model returns content=None, generate_recap calls None.strip() outside its try block; "
               "the AttributeError kills the background recap thread with a traceback on the user's terminal.",
               raises=AttributeError)
    def test_none_content_returns_none(self, fake_ollama):
        fake_ollama.recap = None
        assert ca.generate_recap("m", [user("x")]) is None

    def test_worker_queues_recap(self, fake_ollama):
        q = queue.Queue()
        ca._recap_worker("m", [user("x")], q)
        assert q.get_nowait() == RECAP_TEXT

    def test_worker_queues_nothing_on_failure(self, fake_ollama):
        fake_ollama.recap = ConnectionError()
        q = queue.Queue()
        ca._recap_worker("m", [user("x")], q)
        assert q.empty()


# ------------------------------------------------------------------------------------------------
# Session replay
# ------------------------------------------------------------------------------------------------
class TestReplayHistory:
    def _replay(self, capsys, history, session="s"):
        ca.replay_history(session, history)
        return strip_ansi(capsys.readouterr().out)

    def test_empty_history_prints_nothing(self, capsys):
        assert self._replay(capsys, []) == ""

    def test_plain_exchange(self, capsys):
        out = self._replay(capsys, [user("hello"), assistant("Hi **you**")])
        assert "You[s]> hello" in out and "● Hi you" in out
        assert out.index("hello") < out.index("Hi you")

    def test_multiple_turns_in_order(self, capsys):
        out = self._replay(capsys, [user("q1"), assistant("a1"), user("q2"), assistant("a2")])
        assert [out.index(t) for t in ["q1", "a1", "q2", "a2"]] == sorted(out.index(t) for t in ["q1", "a1", "q2", "a2"])

    def test_tool_turn_shows_only_final_reply(self, capsys):
        history = [user("read it"), assistant("", [{"function": {"name": "read_file", "arguments": {}}}]),
                   {"role": "tool", "tool_name": "read_file", "content": '{"ok": true, "content": "SECRET-DETAIL"}'},
                   assistant("The file says hello.")]
        out = self._replay(capsys, history)
        assert "● The file says hello." in out
        assert "SECRET-DETAIL" not in out and "read_file" not in out

    def test_tool_turn_without_closing_summary_gets_placeholder(self, capsys):
        history = [user("write it"), assistant("", [{"function": {"name": "write_file", "arguments": {}}}]),
                   {"role": "tool", "content": "{}"}, assistant("")]
        assert "no closing summary from the model, but the action(s) above completed" in self._replay(capsys, history)

    def test_unanswered_turn_gets_empty_placeholder(self, capsys):
        out = self._replay(capsys, [user("lost"), user("next"), assistant("answer")])
        assert out.count("model returned an empty response") == 1
        assert out.index("lost") < out.index("empty response") < out.index("next")

    def test_orphaned_leading_entries_skipped(self, capsys):
        history = [{"role": "tool", "content": "{}"}, assistant("orphan"), user("q"), assistant("a")]
        out = self._replay(capsys, history)
        assert "orphan" not in out and "● a" in out

    def test_multiline_user_prompt(self, capsys):
        out = self._replay(capsys, [user("line one\nline two"), assistant("ok")])
        assert "You[s]> line one" in out and "        line two" in out

    def test_replay_is_offline(self, capsys, fake_ollama):
        self._replay(capsys, [user("q"), assistant("a")])
        assert fake_ollama.chat_calls == [] and fake_ollama.recap_calls == []
