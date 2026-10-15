"""Run the real chat_agent.py in a pseudo-terminal and drive it like a user at a keyboard.

The app is launched from a private copy of chat_agent.py/tools.py inside a temp directory, so its
memory/, logs/ and sandboxed PROJECT_ROOT all live there and never touch the real project. Output is
fed through `pyte` (a VT100 emulator) so tests can assert on what is actually visible on screen, and is
also kept as an ANSI-stripped transcript for "did this ever appear" checks.
"""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import struct
import subprocess
import sys
import termios
import threading
import time
from pathlib import Path

import pyte

from tests.helpers import strip_ansi

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_FILES = ("chat_agent.py", "tools.py")

KEY_ENTER = "\r"
KEY_ESC = "\x1b"
KEY_CTRL_C = "\x03"
KEY_ALT_ENTER = "\x1b\r"
CPR_REQUEST = b"\x1b[6n"


def make_app_dir(base: Path) -> Path:
    app = base / "app"
    app.mkdir()
    for name in APP_FILES:
        shutil.copy2(REPO_ROOT / name, app / name)
    return app


class PtyApp:
    def __init__(self, app_dir: Path, args: list[str], ollama_host: str, cols: int = 100, rows: int = 40):
        self.app_dir, self.cols, self.rows = app_dir, cols, rows
        self.screen = pyte.HistoryScreen(cols, rows, history=5000)
        self.stream = pyte.ByteStream(self.screen)
        self._raw = bytearray()
        self._lock = threading.Lock()
        self._mark = 0

        master, slave = os.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST", "COV_"))}
        env.update(OLLAMA_HOST=ollama_host, TERM="xterm-256color", PYTHONUNBUFFERED="1",
                   COLUMNS=str(cols), LINES=str(rows), PYTHONDONTWRITEBYTECODE="1")

        def make_controlling_tty():  # so Ctrl+C on the pty delivers SIGINT, like a real terminal
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

        self.proc = subprocess.Popen(
            [sys.executable, "chat_agent.py", *args], cwd=app_dir, env=env,
            stdin=slave, stdout=slave, stderr=slave, start_new_session=True,
            preexec_fn=make_controlling_tty, close_fds=True)
        os.close(slave)
        self._master = master
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    # -- I/O -------------------------------------------------------------------------------------
    def _read_loop(self) -> None:
        while True:
            try:
                data = os.read(self._master, 65536)
            except OSError:
                break
            if not data:
                break
            if CPR_REQUEST in data:  # answer cursor-position queries like a real terminal would
                os.write(self._master, f"\x1b[{self.screen.cursor.y + 1};{self.screen.cursor.x + 1}R".encode())
            with self._lock:
                self._raw += data
                self.stream.feed(data)

    def send(self, keys: str, delay: float = 0.0) -> None:
        os.write(self._master, keys.encode())
        if delay:
            time.sleep(delay)

    def type(self, text: str, per_key: float = 0.005) -> None:
        for ch in text:
            self.send(ch, per_key)

    def submit(self, text: str) -> None:
        self.type(text)
        self.send(KEY_ENTER)

    # -- observing -------------------------------------------------------------------------------
    @property
    def transcript(self) -> str:
        with self._lock:
            return strip_ansi(self._raw.decode("utf-8", errors="replace"))

    def screen_text(self) -> str:
        with self._lock:
            return "\n".join(line.rstrip() for line in self.screen.display)

    def expect(self, pattern: str, timeout: float = 10.0, regex: bool = False) -> str:
        """Wait until `pattern` appears in the transcript *after* the previous expect() match."""
        deadline = time.monotonic() + timeout
        rx = re.compile(pattern if regex else re.escape(pattern))
        while time.monotonic() < deadline:
            text = self.transcript
            m = rx.search(text, self._mark)
            if m:
                self._mark = m.end()
                return m.group(0)
            if self.proc.poll() is not None:
                time.sleep(0.1)
                text = self.transcript
                if rx.search(text, self._mark):
                    continue
                raise AssertionError(f"app exited ({self.proc.returncode}) before {pattern!r} appeared.\n"
                                     f"--- transcript tail ---\n{text[-3000:]}")
            time.sleep(0.02)
        raise AssertionError(f"timed out waiting for {pattern!r}\n--- transcript tail ---\n{self.transcript[-3000:]}")

    def expect_prompt(self, session: str = "default", timeout: float = 10.0) -> None:
        self.expect(f"You[{session}]>", timeout=timeout)
        time.sleep(0.15)  # let prompt_toolkit finish entering raw mode before we type

    def wait_screen(self, predicate, timeout: float = 5.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = self.screen_text()
            if predicate(text):
                return text
            time.sleep(0.03)
        raise AssertionError(f"screen never satisfied predicate\n--- screen ---\n{self.screen_text()}")

    def wait_exit(self, timeout: float = 10.0) -> int:
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise AssertionError(f"app did not exit\n--- transcript tail ---\n{self.transcript[-2000:]}")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)
        try:
            os.close(self._master)
        except OSError:
            pass
