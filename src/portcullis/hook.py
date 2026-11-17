"""Claude Code PreToolUse hook backed by the policy engine.

Claude Code runs this before every tool call and sends the call as JSON on stdin. The policy
decides, and the answer goes back as Claude Code's hook output:

- allow: print nothing. Claude Code's own permission flow still applies, so portcullis only ever
  narrows what is permitted. (Printing "allow" would skip the user's own prompts.)
- deny: exit 2 with the reason on stderr. Exit 2 blocks the call even if stdout were truncated or
  polluted; a deny sent as exit 0 plus JSON would turn into "proceed" if that JSON were invalid.
- ask: exit 0 with the decision as JSON (there is no other way to ask).
- any error (bad input, missing or invalid policy, failed audit write): exit 2, which blocks the call.

Configuration, all from the environment:
  PORTCULLIS_POLICY        path to a policy JSON file (default .portcullis/policy.json under
                           CLAUDE_PROJECT_DIR, or the current folder if that is not set)
  PORTCULLIS_TENANT        which tenant's rules apply (default claude-code)
  PORTCULLIS_DATABASE_URL  optional. If set, every decision is written to the hash-chained audit
                           log before the answer is returned, and a failed write blocks the call.

Limits: each call is a separate process, so budgets and loop detection do not apply here yet
(the state is always fresh). An "ask" is answered by the user in Claude Code, which does not
tell the hook what they chose, so approvals are not recorded.
"""

import json
import os
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from portcullis.policy import Decision, Policy, RunState, ToolCall, Verdict, decide, fingerprint

DEFAULT_POLICY = ".portcullis/policy.json"
DEFAULT_TENANT = "claude-code"


class HookError(Exception):
    """Something is wrong with the input, the policy or the audit log. The call is blocked."""


def _read_event(stdin: bytes) -> dict[str, Any]:
    try:
        event = json.loads(stdin)
    except ValueError as exc:  # includes bad UTF-8 and integers beyond Python's digit limit
        raise HookError("input is not valid JSON") from exc
    if not isinstance(event, dict):
        raise HookError("input must be a JSON object")
    if not isinstance(event.get("tool_name"), str) or not event["tool_name"]:
        raise HookError("tool_name is missing")
    if not isinstance(event.get("tool_input", {}), dict):
        raise HookError("tool_input must be an object")
    return event


def load_policy(path: Path) -> Policy:
    try:
        return Policy.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HookError(f"cannot read policy {path}: {exc.strerror}") from exc
    except (ValidationError, UnicodeDecodeError) as exc:
        raise HookError(f"policy {path} is invalid") from exc


def _load_policy(env: Mapping[str, str]) -> Policy:
    project = env.get("CLAUDE_PROJECT_DIR")
    default = Path(project) / DEFAULT_POLICY if project else Path(DEFAULT_POLICY)
    return load_policy(Path(env.get("PORTCULLIS_POLICY") or default))


def _audit(env: Mapping[str, str], event: dict[str, Any], tenant: str, decision: Decision) -> None:
    url = env.get("PORTCULLIS_DATABASE_URL")
    if not url:
        return
    # Imported here, not at the top: psycopg costs about 0.7 s to import, and this hook runs
    # before every tool call, so callers who do not audit should not pay for it.
    import psycopg

    from portcullis.audit import append, create_audit_log, require_project_database

    session = event.get("session_id") if isinstance(event.get("session_id"), str) else "no-session"
    tool_use = event.get("tool_use_id") if isinstance(event.get("tool_use_id"), str) else uuid.uuid4().hex
    payload = {
        "tool": event["tool_name"],
        "verdict": decision.verdict.value,
        "rule": decision.rule,
        "reason": decision.reason,
        "request_sha256": fingerprint(event["tool_name"], event.get("tool_input", {})),  # a hash, not the command
    }
    try:
        require_project_database(url)
        # The timeouts keep a stuck lock or slow query from outlasting the hook: Claude Code lets a
        # call through when a hook times out, so this must fail (and block) first.
        with psycopg.connect(
            url, autocommit=True, connect_timeout=5, options="-c statement_timeout=5000 -c lock_timeout=5000"
        ) as conn:
            create_audit_log(conn)
            append(conn, session, tenant, f"{tool_use}:decision", "decision", payload)
    except Exception as exc:  # fail closed on any audit problem; never print the URL, it holds a password
        detail = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        raise HookError(f"audit write failed: {detail}") from exc


def _render_ask(decision: Decision) -> str:
    answer = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": f"Portcullis [{decision.rule}]: {decision.reason}",
        }
    }
    return json.dumps(answer)


def run(stdin: bytes, env: Mapping[str, str]) -> tuple[int, str, str]:
    """Return (exit code, stdout, stderr). Exit 0 answers; exit 2 blocks the call."""
    try:
        event = _read_event(stdin)
        policy = _load_policy(env)
        tenant = env.get("PORTCULLIS_TENANT") or DEFAULT_TENANT
        call = ToolCall(tenant_id=tenant, tool=event["tool_name"], args=event.get("tool_input", {}))
        decision = decide(policy, call, RunState())
        _audit(env, event, tenant, decision)  # written before the answer goes back
        if decision.verdict is Verdict.DENY:
            return 2, "", f"Portcullis [{decision.rule}]: {decision.reason}\n"
        return 0, _render_ask(decision) if decision.verdict is Verdict.ASK else "", ""
    except HookError as exc:
        return 2, "", f"portcullis hook: {exc}\n"
    except Exception as exc:  # deliberate catch-all: exit 1 would let the call through
        return 2, "", f"portcullis hook: unexpected {type(exc).__name__}\n"


def main() -> int:
    try:
        code, out, err = run(sys.stdin.buffer.read(), os.environ)
        sys.stdout.write(out)
        sys.stdout.flush()
        sys.stderr.write(err)
        sys.stderr.flush()
        return code
    except BaseException:  # a broken pipe or encoding error must not exit 1, which lets the call through
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
