"""PreToolUse hook, day-one stub: log every tool call, decide nothing.

Prints no decision, so Claude Code's normal permission flow still applies.
Printing "allow" here would silently auto-approve every tool call.
Stdlib only, so it runs on any Python 3.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def log_path() -> Path:
    override = os.environ.get("PORTCULLIS_LOG")
    if override:
        return Path(override)
    return Path(os.environ.get("CLAUDE_PLUGIN_DATA", ".portcullis")) / "tool_calls.jsonl"


def build_entry(event: object) -> dict:
    if not isinstance(event, dict):
        raise ValueError("expected a JSON object on stdin")
    # Tool input can hold secrets (commands, file contents), so only its hash is stored.
    canonical_input = json.dumps(event.get("tool_input"), sort_keys=True, separators=(",", ":"))
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": event.get("session_id"),
        "tool_use_id": event.get("tool_use_id"),
        "tool_name": event["tool_name"],
        "input_sha256": hashlib.sha256(canonical_input.encode()).hexdigest(),
    }


def main() -> int:
    try:
        # Read bytes: text-mode stdin on Windows decodes with the locale codec, but Claude Code sends UTF-8.
        entry = build_entry(json.load(sys.stdin.buffer))
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(entry) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Exit 1 is a visible but non-blocking error (exit 2 would block the tool call).
        # ponytail: the real policy hook must fail closed with exit 2; this stub decides nothing.
        print(f"portcullis hook: {exc!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
