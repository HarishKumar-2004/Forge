"""Small helpers shared across test modules."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\r")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def write(root: Path, rel: str, content: str | bytes) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


ASSERTION_FAILURES = (AssertionError, pytest.fail.Exception)


def known_bug(reason: str, condition: bool = True, raises=ASSERTION_FAILURES):
    """Mark a test that asserts the *correct* behavior for a confirmed defect.

    It runs as a strict xfail: the suite stays green while the bug exists, and turns red (XPASS) the
    moment the bug is fixed — a reminder to delete this marker so the test becomes a regression guard.
    `raises` pins the failure to the bug's own symptom (an assertion by default, or the specific crash
    exception), so an unrelated error in the test is still reported as a real failure.
    """
    marks = [pytest.mark.known_bug,
             pytest.mark.xfail(condition=condition, reason=f"BUG: {reason}", strict=True, raises=raises)]

    def apply(fn):
        for m in marks:
            fn = m(fn)
        return fn
    return apply
