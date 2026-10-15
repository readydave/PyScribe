"""Skip test modules whose optional heavy dependencies (torch, Qt, ...) are not installed.

CI runs a light environment; modules that cannot import a third-party package are
reported as skipped instead of failing collection. Missing in-repo modules still fail.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_MISSING_RE = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")


def _is_repo_module(name: str) -> bool:
    top = name.split(".")[0]
    return (ROOT / top).is_dir() or (ROOT / f"{top}.py").is_file()


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector: pytest.Collector):
    outcome = yield
    report = outcome.get_result()
    if not report.failed or not isinstance(collector, pytest.Module):
        return
    match = _MISSING_RE.search(str(report.longrepr))
    if match and not _is_repo_module(match.group(1)):
        report.outcome = "skipped"
        report.longrepr = (str(collector.path), 0, f"Skipped: optional dependency '{match.group(1)}' not installed")
