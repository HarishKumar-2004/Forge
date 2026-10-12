# Forge

A terminal-based AI chat agent backed by a **local Ollama model** — persistent memory, multi-session support, sandboxed file tool-calling with human-in-the-loop approval, and a polished `prompt_toolkit`-driven terminal UI.

Everything runs on your machine. No API keys, no cloud calls — just your local Ollama server and a plain JSON memory store on disk.

![demo](docs/demo.gif)
<!--
  TODO: record a short demo GIF and drop it at docs/demo.gif, then this line will render it.
  Quick way to make one on macOS:
    1. brew install asciinema agg
    2. asciinema rec docs/demo.cast          # record a short session, then Ctrl+D to stop
    3. agg docs/demo.cast docs/demo.gif       # convert the recording to a GIF
-->

## Features

- **Local-first** — talks to a locally running [Ollama](https://ollama.com) model; no data leaves your machine.
- **Persistent memory** — every session's conversation history is saved to `memory/<session>.json` and reloaded automatically.
- **Multiple sessions** — switch, list, and rename sessions without losing context on any of them.
- **Tool-calling with safety guardrails** — the model can list, search, read, write, and edit files, but every mutating action (`write_file`, `edit_file`) is sandboxed to the project directory, shown to you as a colored diff, and requires explicit `y/N` confirmation before it runs.
- **Full audit trail** — every tool call (approved, declined, or failed) is logged to `logs/tool_calls.jsonl` with arguments, timing, and outcome.
- **Polished terminal UX** — slash-command autocomplete, a live "Thinking… (Ns)" indicator, Esc/Ctrl+C to interrupt a generation (your message is restored to the prompt box, not lost), and Alt+Enter for multi-line input.

## Architecture

```mermaid
flowchart TB
    subgraph Terminal["Terminal UI (chat_agent.py)"]
        P["prompt_toolkit PromptSession<br/>slash-command autocomplete, live spinner"]
    end

    subgraph Core["Agent loop"]
        H["Conversation history<br/>(in memory, trimmed to last N messages)"]
        O["ollama.chat()<br/>(background thread + spinner)"]
    end

    subgraph Tools["tools.py — sandboxed file tools"]
        S["resolve_sandboxed_path()<br/>blocks absolute paths, '..' escapes,<br/>and memory/ logs/ .venv/ .git/"]
        RO["read_file / list_files / search_file<br/>(read-only, auto-approved)"]
        RW["write_file / edit_file<br/>(mutating — gated)"]
    end

    subgraph Disk["Local disk"]
        MEM[("memory/&lt;session&gt;.json")]
        LOG[("logs/tool_calls.jsonl")]
        FS[("project files")]
    end

    P -->|user message| H --> O
    O -->|no tool calls| Reply["print_agent_reply()"]
    O -->|tool_calls| S
    S --> RO --> FS
    S --> RW -->|diff preview + y/N confirm| FS
    RO --> LOG
    RW --> LOG
    RO --> O
    RW --> O
    H <--> MEM
    Reply --> P
```

Every turn: your message is appended to history and saved, sent to Ollama along with the available tool schemas, and the loop continues feeding tool results back to the model (up to `MAX_TOOL_ROUNDS`) until it returns a final text reply.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- A pulled model (default: `qwen3.5:9b`)

## Installation

```bash
git clone git@github.com:HarishKumar-2004/Forge.git
cd Forge
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pull the default model (or any model you plan to use):

```bash
ollama pull qwen3.5:9b
```

Make sure the Ollama server is running (`ollama serve`, or just open the Ollama app) before starting the agent.

## Usage

```bash
source .venv/bin/activate
python chat_agent.py
```

| Flag | Description |
|---|---|
| `--model NAME` | Use a specific Ollama model (default: `qwen3.5:9b`) |
| `--resume NAME` | Resume an existing named session (must already exist) |
| `--reset` | Clear the active session's memory before starting |

Type `exit`, `quit`, or `bye` to end the chat.

### Slash commands

| Command | Description |
|---|---|
| `/reset` | Clear the active session's memory |
| `/session [name]` | Switch to (or show) the active session |
| `/sessions` | List all sessions |
| `/rename <name>` | Rename the active session |
| `/model [name]` | Switch to (or show) the active model |
| `/models` | List all installed Ollama models |

Typing `/` opens an autocomplete menu for these; it narrows as you keep typing and supports Escape to dismiss.

### Keyboard shortcuts

| Key | Effect |
|---|---|
| `Enter` | Submit (ignored on a blank prompt) |
| `Shift+Enter` or `Alt+Enter` | Insert a newline for a multi-line message |
| `Esc` or `Ctrl+C` (while the agent is generating) | Cancel the current turn; your message is restored to the prompt box for editing |
| `Esc` (while the slash-command menu is open) | Dismiss the menu |

## Safety guardrails

The model never touches your filesystem directly — every tool call goes through `tools.py`'s sandbox:

- **Path confinement** — `resolve_sandboxed_path()` rejects absolute paths and any path that resolves outside the project root, and blocks `memory/`, `logs/`, `.venv/`, `__pycache__/`, `.git/`, and `node_modules/` outright.
- **Read vs. write separation** — `list_files`, `search_file`, and `read_file` run automatically. `write_file` and `edit_file` always stop first: you see a colored, line-numbered diff of exactly what would change, and must type `y`/`yes` to approve. Decline, and the model is told not to retry the same action unprompted.
- **Size caps** — reads, searches, and writes are all capped (file size, line count, match count) so a single tool call can't blow up your context window or write unbounded data.
- **Full audit log** — `logs/tool_calls.jsonl` records every call: tool name, arguments (truncated), whether it mutated anything, who approved it, outcome, and duration.

## Sessions & memory

Each session's history lives in its own `memory/<session>.json` file, created automatically on first use. History is trimmed to the most recent `MAX_HISTORY_MESSAGES` entries per session to keep requests fast. If a turn is interrupted (Esc/Ctrl+C), that turn is rolled back entirely — nothing partial gets saved.

## Project structure

```
.
├── chat_agent.py   # terminal UI, agent loop, session/memory management
├── tools.py        # sandboxed file tools exposed to the model
├── requirements.txt
├── memory/         # per-session conversation history (gitignored)
├── logs/           # tool-call audit log (gitignored)
└── .venv/          # virtual environment (gitignored)
```
