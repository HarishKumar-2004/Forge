"""Scriptable stand-ins for the Ollama server, shared by the in-process and end-to-end test layers.

A test describes what the "model" should do on each successive chat request as a list of `Turn`s.
`FakeOllama` replays them in-process (patched over `ollama.chat` / `ollama.list`); the e2e layer's
HTTP server (tests/e2e/fake_ollama_server.py) replays the very same `Turn`s over real HTTP.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ollama import ChatResponse, ListResponse, Message

DEFAULT_TEST_MODEL = "qwen3.5:9b"
RECAP_TEXT = "You discussed the test project and nothing is left open."


@dataclass
class ToolCallSpec:
    name: str
    arguments: dict[str, Any]

    def to_message(self) -> Message.ToolCall:
        return Message.ToolCall(function=Message.ToolCall.Function(name=self.name, arguments=self.arguments))

    def to_json(self) -> dict:
        return {"function": {"name": self.name, "arguments": self.arguments}}


def call(name: str, **arguments: Any) -> ToolCallSpec:
    """Shorthand for one model-issued tool call: call("read_file", path="a.txt")."""
    return ToolCallSpec(name, arguments)


@dataclass
class Turn:
    """What the fake model returns for ONE chat request.

    content        reply text; streamed in `chunk_size`-char pieces (None = one chunk per line)
    tool_calls     tool calls attached to the response
    split_tool_calls  stream each tool call in its own chunk instead of all in the final chunk
    error          raise this exception (in-process) / return HTTP 500 (e2e) instead of replying
    error_after_chunks  stream this many content chunks first, then fail mid-stream
    interrupt      in-process only: raise KeyboardInterrupt, simulating Esc/Ctrl+C during generation
    chunk_delay    e2e only: seconds to sleep between streamed chunks
    """
    content: str = ""
    tool_calls: list[ToolCallSpec] = field(default_factory=list)
    chunk_size: int | None = None
    split_tool_calls: bool = False
    error: Exception | None = None
    error_after_chunks: int | None = None
    interrupt: bool = False
    chunk_delay: float = 0.0

    def content_chunks(self) -> list[str]:
        if not self.content:
            return []
        if self.chunk_size:
            return [self.content[i:i + self.chunk_size] for i in range(0, len(self.content), self.chunk_size)]
        return self.content.splitlines(keepends=True)


def reply(content: str, **kw: Any) -> Turn:
    return Turn(content=content, **kw)


def tool_round(*calls: ToolCallSpec, content: str = "", **kw: Any) -> Turn:
    return Turn(content=content, tool_calls=list(calls), **kw)


def is_recap_request(messages: list[dict]) -> bool:
    import chat_agent
    return bool(messages) and messages[-1].get("role") == "user" and messages[-1].get("content") == chat_agent.RECAP_INSTRUCTION


class FakeOllama:
    """In-process replacement for the `ollama` module's chat()/list() functions.

    Chat requests consume `Turn`s in order. Recap requests (the background resume summary) are served
    separately from `recap`, so they never race with the scripted conversation turns.
    """

    def __init__(self, models: list[str] | None = None):
        self.turns: deque[Turn] = deque()
        self.models: list[str] | None = [DEFAULT_TEST_MODEL] if models is None else models
        self.list_error: Exception | None = None
        self.recap: str | None | Exception = RECAP_TEXT
        self.chat_calls: list[dict] = []
        self.recap_calls: list[dict] = []
        self._lock = threading.Lock()

    # -- scripting ------------------------------------------------------------------------------
    def script(self, *turns: Turn) -> "FakeOllama":
        self.turns.extend(turns)
        return self

    @property
    def remaining(self) -> int:
        return len(self.turns)

    # -- ollama.list ----------------------------------------------------------------------------
    def list(self) -> ListResponse:
        if self.list_error is not None:
            raise self.list_error
        return ListResponse(models=[ListResponse.Model(model=m) for m in (self.models or [])])

    # -- ollama.chat ----------------------------------------------------------------------------
    def chat(self, model: str, messages: list[dict], tools: list[dict] | None = None,
             stream: bool = False, options: dict | None = None, **kwargs: Any):
        record = {"model": model, "messages": [dict(m) for m in messages], "tools": tools,
                  "stream": stream, "options": options}
        if not stream and is_recap_request(messages):
            with self._lock:
                self.recap_calls.append(record)
            if isinstance(self.recap, Exception):
                raise self.recap
            return ChatResponse(model=model, done=True,
                                message=Message(role="assistant", content=self.recap))

        with self._lock:
            self.chat_calls.append(record)
            if not self.turns:
                raise AssertionError(f"FakeOllama: unexpected chat request #{len(self.chat_calls)} (script exhausted)")
            turn = self.turns.popleft()

        if turn.interrupt:
            raise KeyboardInterrupt
        if turn.error is not None and turn.error_after_chunks is None:
            raise turn.error
        if not stream:
            return ChatResponse(model=model, done=True, message=Message(
                role="assistant", content=turn.content,
                tool_calls=[c.to_message() for c in turn.tool_calls] or None))
        return self._stream(model, turn)

    @staticmethod
    def _stream(model: str, turn: Turn):
        for i, piece in enumerate(turn.content_chunks()):
            if turn.error_after_chunks is not None and i >= turn.error_after_chunks:
                raise turn.error
            yield ChatResponse(model=model, done=False, message=Message(role="assistant", content=piece))
        if turn.error_after_chunks is not None:
            raise turn.error
        if turn.split_tool_calls:
            for c in turn.tool_calls:
                yield ChatResponse(model=model, done=False,
                                   message=Message(role="assistant", content="", tool_calls=[c.to_message()]))
            final_calls = None
        else:
            final_calls = [c.to_message() for c in turn.tool_calls] or None
        yield ChatResponse(model=model, done=True, done_reason="stop",
                           message=Message(role="assistant", content="", tool_calls=final_calls))
