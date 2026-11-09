import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).parent.parent / "plugin" / "scripts" / "log_tool_call.py"

EVENT = {
    "session_id": "s1",
    "hook_event_name": "PreToolUse",
    "tool_name": "Bash",
    "tool_input": {"command": "echo SECRET_TOKEN"},
    "tool_use_id": "toolu_1",
}


def run_hook(stdin: str, log: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=stdin,
        text=True,
        capture_output=True,
        env={**os.environ, "PORTCULLIS_LOG": str(log)},
    )


def read_log(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def expected_hash(tool_input: dict) -> str:
    canonical = json.dumps(tool_input, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def test_logs_the_call_and_prints_no_decision(tmp_path):
    log = tmp_path / "calls.jsonl"

    result = run_hook(json.dumps(EVENT), log)

    assert result.returncode == 0
    assert result.stdout == ""  # no decision, so Claude Code's normal permission flow applies
    [entry] = read_log(log)
    assert entry["tool_name"] == "Bash"
    assert entry["session_id"] == "s1"
    assert entry["tool_use_id"] == "toolu_1"
    assert entry["ts"]


def test_log_holds_a_hash_of_the_input_not_the_input_itself(tmp_path):
    log = tmp_path / "calls.jsonl"

    run_hook(json.dumps(EVENT), log)

    assert "SECRET_TOKEN" not in log.read_text()
    assert read_log(log)[0]["input_sha256"] == expected_hash(EVENT["tool_input"])


def test_non_ascii_input_is_read_as_utf8_whatever_the_windows_locale(tmp_path):
    # Claude Code sends UTF-8 bytes; a text-mode stdin on Windows would decode them as cp1252.
    log = tmp_path / "calls.jsonl"
    event = {**EVENT, "tool_input": {"text": "café €"}}
    payload = json.dumps(event, ensure_ascii=False).encode("utf-8")

    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input=payload,
        capture_output=True,
        env={**os.environ, "PORTCULLIS_LOG": str(log)},
    )

    assert result.returncode == 0
    assert read_log(log)[0]["input_sha256"] == expected_hash(event["tool_input"])


def test_appends_one_line_per_call(tmp_path):
    log = tmp_path / "calls.jsonl"

    run_hook(json.dumps(EVENT), log)
    run_hook(json.dumps({**EVENT, "tool_use_id": "toolu_2"}), log)

    assert [e["tool_use_id"] for e in read_log(log)] == ["toolu_1", "toolu_2"]


@pytest.mark.parametrize("stdin", ["not json", "[]", "null", "{}"], ids=["garbage", "list", "null", "no_tool"])
def test_bad_input_is_a_visible_non_blocking_error_and_writes_nothing(tmp_path, stdin):
    log = tmp_path / "calls.jsonl"

    result = run_hook(stdin, log)

    assert result.returncode == 1  # exit 2 would block the tool call
    assert "portcullis hook" in result.stderr
    assert not log.exists()
