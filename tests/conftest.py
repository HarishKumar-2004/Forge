"""Shared fixtures: per-test sandboxing of every on-disk path, a fake Ollama, and an in-process app runner.

Isolation is enforced, not optional: every test runs with tools.PROJECT_ROOT, chat_agent.MEMORY_DIR,
and chat_agent.LOGS_DIR pointed at a fresh temp directory, real Ollama calls are blocked, and a
session-level guard fails the run if the real memory/ or logs/ directories change at all.
"""

from __future__ import annotations

import builtins
import io
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import ollama
import pytest

import chat_agent
import tools
from tests.fakes import FakeOllama
from tests.helpers import strip_ansi

REAL_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------------------------------------
# Guard: the user's real conversation memory and audit log must never be touched by the suite.
# ------------------------------------------------------------------------------------------------
def _snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    if not directory.exists():
        return {}
    return {str(p.relative_to(directory)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in directory.rglob("*") if p.is_file()}


@pytest.fixture(scope="session", autouse=True)
def _real_user_data_untouched():
    watched = [REAL_PROJECT_ROOT / "memory", REAL_PROJECT_ROOT / "logs"]
    before = {d: _snapshot(d) for d in watched}
    yield
    for d in watched:
        assert _snapshot(d) == before[d], f"Test suite modified real user data under {d}"


# ------------------------------------------------------------------------------------------------
# Per-test sandbox
# ------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch) -> Path:
    """A throwaway project root. All tool paths, memory files and logs resolve inside it."""
    root = (tmp_path / "project").resolve()
    root.mkdir()
    monkeypatch.setattr(tools, "PROJECT_ROOT", root)
    monkeypatch.setattr(chat_agent, "MEMORY_DIR", root / "memory")
    monkeypatch.setattr(chat_agent, "LOGS_DIR", root / "logs")
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("LINES", "24")
    return root


@pytest.fixture(autouse=True)
def _no_real_ollama(monkeypatch):
    """Any test that reaches the network without installing `fake_ollama` fails loudly."""
    def refuse(*_a, **_kw):
        raise AssertionError("Test attempted a real Ollama call; use the fake_ollama fixture")
    monkeypatch.setattr(ollama, "chat", refuse)
    monkeypatch.setattr(ollama, "list", refuse)


@pytest.fixture
def fake_ollama(monkeypatch) -> FakeOllama:
    fake = FakeOllama()
    monkeypatch.setattr(ollama, "chat", fake.chat)
    monkeypatch.setattr(ollama, "list", fake.list)
    return fake


@pytest.fixture
def sync_threads(monkeypatch):
    """Run chat_agent's background threads inline, so recap/streaming ordering is deterministic."""
    class InlineThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            self._target, self._args, self._kwargs = target, args, kwargs or {}

        def start(self):
            self._target(*self._args, **self._kwargs)

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    shim = types.SimpleNamespace(Thread=InlineThread)
    monkeypatch.setattr(chat_agent, "threading", shim)



# ------------------------------------------------------------------------------------------------
# In-process app runner: drives chat_agent.main() through its non-TTY input() path.
# ------------------------------------------------------------------------------------------------
@dataclass
class AppRun:
    raw: str
    prompts: list[str]
    exit_code: int | None
    confirmations: list[str] = field(default_factory=list)

    @property
    def out(self) -> str:
        return strip_ansi(self.raw)


@pytest.fixture
def run_app(monkeypatch, capsys, fake_ollama, sync_threads):
    """run_app(inputs, argv=[], approve=[...]) -> AppRun.

    `inputs` are fed to each `You[...]>` prompt (an exception item, e.g. KeyboardInterrupt, is raised
    at that prompt instead); once exhausted the next prompt sees EOF, which ends the
    app the same way Ctrl+D does. `approve` scripts the answers to y/N mutation confirmations (each
    True/False); confirmations beyond the script are declined. Exceptions from main() propagate.
    """
    def _run(inputs: list[str], argv: list[str] | None = None, approve: list[bool] | None = None) -> AppRun:
        feed = list(inputs)
        prompts: list[str] = []
        answers = list(approve or [])
        confirmations: list[str] = []

        def fake_input(prompt=""):
            prompts.append(prompt)
            if not feed:
                raise EOFError
            item = feed.pop(0)
            if isinstance(item, BaseException) or (isinstance(item, type) and issubclass(item, BaseException)):
                raise item
            return item

        def fake_confirm(prompt_text):
            confirmations.append(strip_ansi(prompt_text))
            return answers.pop(0) if answers else False

        monkeypatch.setattr(builtins, "input", fake_input)
        monkeypatch.setattr(chat_agent, "_ask_confirmation", fake_confirm)
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        monkeypatch.setattr(sys, "argv", ["chat_agent.py", *(argv or [])])
        exit_code = None
        try:
            chat_agent.main()
        except SystemExit as e:
            exit_code = e.code
        captured = capsys.readouterr()
        return AppRun(raw=captured.out, prompts=prompts, exit_code=exit_code, confirmations=confirmations)

    return _run


@pytest.fixture
def memory_file(project):
    def _path(session: str) -> Path:
        return project / "memory" / f"{session}.json"
    return _path
