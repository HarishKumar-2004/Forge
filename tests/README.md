# Forge test suite

```bash
pip install -r requirements-dev.txt
pytest                      # everything (~35s)
pytest -m "not e2e"         # skip the pseudo-terminal tests (~5s)
pytest -m known_bug -rx     # list every documented, still-open bug
pytest --cov=chat_agent --cov=tools --cov-report=term-missing
```

No Ollama server or model is needed — every layer runs against a fake.

## Layers

| Layer | Where | What it drives | Ollama is… |
|---|---|---|---|
| Unit | `tests/unit/` | individual functions: sandbox, each tool, formatting/wrapping, diffs, persistence, logging, completer & key bindings, confirmation gating, streaming reveal, recap, replay | `FakeOllama`, patched over `ollama.chat`/`ollama.list` |
| Integration | `tests/integration/` | the real `main()` loop in-process via its non-TTY `input()` path: CLI flags, every slash command, multi-round tool loops, approvals, errors, interrupt rollback, session switching | `FakeOllama` |
| End-to-end | `tests/e2e/` | the real `chat_agent.py` as a subprocess in a pseudo-terminal (rendered through the `pyte` VT100 emulator): typing, Alt+Enter, Esc/Ctrl+C mid-stream, y/N on a real TTY, slash menu, `--resume` | `FakeOllamaServer` — real HTTP, real `ollama` client |

Conversations are scripted the same way in every layer, with `Turn`s from `tests/fakes.py`:

```python
fake_ollama.script(
    tool_round(call("read_file", path="notes.txt")),   # round 1: model calls a tool
    reply("The word is pineapple.", chunk_size=4),     # round 2: streamed final answer
)
```

## Isolation guarantees

- Every test gets its own temp project root: `tools.PROJECT_ROOT`, `chat_agent.MEMORY_DIR`, and `chat_agent.LOGS_DIR` are all redirected there (autouse fixture in `conftest.py`). E2E tests run a *copy* of the app inside a temp dir.
- Any accidental real Ollama call fails the test.
- A session-level guard snapshots the real `memory/` and `logs/` and fails the run if either changes.

## Known bugs (`@known_bug`)

Tests marked `@known_bug(...)` assert the **correct** behavior for a defect that exists today. They run as strict xfails pinned to the bug's specific symptom:

- while the bug exists → `xfailed` (suite stays green)
- once it's fixed → `XPASS(strict)` **fails the run** — delete the marker and the test becomes a permanent regression guard
- if the test breaks for some *other* reason → it's reported as a normal failure, not hidden
