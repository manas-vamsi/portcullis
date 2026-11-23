"""Fast path for the Claude Code hook: decides without loading pydantic when it safely can.

The full hook (`portcullis.hook`) spends most of each call starting Python, loading pydantic and building
the policy's models; the decision itself takes microseconds. This module decides the common case with the
standard library only, using the same checks (`portcullis.rules`), and hands everything else to the full
hook unchanged:

- a database is set (PORTCULLIS_DATABASE_URL): the full hook, which records and keeps state;
- the event is not plainly well formed: the full hook, which reports the problem and blocks;
- the policy file is anything but plainly valid (unknown keys, odd types, duplicate keys, a number that
  is not finite, a regex that does not compile ...): the full hook, which validates it with pydantic.

So this path can only be faster, never looser: a policy it accepts is one the full validator accepts with
the same meaning, and the answer is the one the full hook would give (both are tested on generated input).
Any unexpected error here also falls back to the full hook, which fails closed.
"""

import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from portcullis.rules import Risk, Verdict, decide

DEFAULT_POLICY = ".portcullis/policy.json"  # the same defaults as portcullis.hook
DEFAULT_TENANT = "claude-code"
SHA256 = re.compile(r"[0-9a-f]{64}")
RISKS = {r.value: r for r in Risk}


class NotPlain(Exception):
    """The input is not in the plain form this path handles. Not an error: the full hook takes over."""


# --- strict reading: accept only what pydantic would accept, and give it the same meaning ---


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise NotPlain("duplicate key")
    return dict(pairs)


def _reject_constant(name: str) -> None:
    raise NotPlain(f"{name} is not JSON")


def _object(value: Any, required: set[str], optional: frozenset[str] = frozenset()) -> dict[str, Any]:
    if type(value) is not dict or not required <= value.keys() <= required | optional:
        raise NotPlain("unexpected shape")
    return value


def _text(value: Any) -> str:
    if type(value) is not str:
        raise NotPlain("not a string")
    value.encode("utf-8")  # a lone surrogate raises here
    return value


def _count(value: Any) -> int:
    """A pydantic `int` with gt=0."""
    if type(value) is not int or value <= 0:
        raise NotPlain("not a positive integer")
    return value


def _number(value: Any, positive: bool = False) -> float:
    """A pydantic `float` with ge=0 (or gt=0) and allow_inf_nan=False, converted to float as pydantic does."""
    if type(value) not in (int, float):
        raise NotPlain("not a number")
    number = float(value)  # OverflowError for a huge int: caught by the caller, full hook takes over
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise NotPlain("out of range")
    return number


def _optional_number(value: Any) -> float | None:
    return None if value is None else _number(value)


def _patterns(value: Any) -> tuple[SimpleNamespace, ...]:
    if type(value) is not list:
        raise NotPlain("patterns must be a list")
    found = []
    for item in value:
        raw = _object(item, {"arg", "regex"}, frozenset({"required"}))
        required = raw.get("required", False)
        if type(required) is not bool:
            raise NotPlain("required must be true or false")
        regex = _text(raw["regex"])
        re.compile(regex)  # re.error: the full hook reports it
        found.append(SimpleNamespace(arg=_text(raw["arg"]), regex=regex, required=required))
    return tuple(found)


def _rule(value: Any) -> SimpleNamespace:
    raw = _object(value, {"risk"}, frozenset({"pinned_sha256", "approve_above", "deny_if", "ask_if"}))
    risk = RISKS.get(raw["risk"]) if type(raw["risk"]) is str else None
    pin = raw.get("pinned_sha256")
    if risk is None or not (pin is None or (type(pin) is str and SHA256.fullmatch(pin))):
        raise NotPlain("bad risk or pin")
    threshold = raw.get("approve_above")
    if threshold is not None:
        threshold = _object(threshold, {"arg", "limit"})
        threshold = SimpleNamespace(arg=_text(threshold["arg"]), limit=_number(threshold["limit"]))
    return SimpleNamespace(
        risk=risk,
        pinned_sha256=pin,
        approve_above=threshold,
        deny_if=_patterns(raw.get("deny_if", [])),
        ask_if=_patterns(raw.get("ask_if", [])),
    )


def _budget(value: Any) -> SimpleNamespace:
    raw = _object(
        value,
        {"max_steps", "max_tokens", "max_seconds", "max_identical_calls"},
        frozenset({"max_spend", "max_spend_per_day"}),
    )
    return SimpleNamespace(
        max_steps=_count(raw["max_steps"]),
        max_tokens=_count(raw["max_tokens"]),
        max_seconds=_number(raw["max_seconds"], positive=True),
        max_identical_calls=_count(raw["max_identical_calls"]),
        max_spend=_optional_number(raw.get("max_spend")),
        max_spend_per_day=_optional_number(raw.get("max_spend_per_day")),
    )


def read_policy(text: str) -> SimpleNamespace:
    """The policy as plain objects, or NotPlain (or another exception) if it is not plainly valid."""
    raw = _object(json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant), {"tenants"})
    tenants = raw["tenants"]
    if type(tenants) is not dict:
        raise NotPlain("tenants must be an object")
    plain = {}
    for name, tenant in tenants.items():
        tenant = _object(tenant, {"budget", "tools"})
        if type(tenant["tools"]) is not dict:
            raise NotPlain("tools must be an object")
        tools = {_text(tool): _rule(rule) for tool, rule in tenant["tools"].items()}
        plain[_text(name)] = SimpleNamespace(budget=_budget(tenant["budget"]), tools=tools)
    return SimpleNamespace(tenants=plain)


# --- the hook ---


def _fast(stdin: bytes, env: Mapping[str, str]) -> tuple[int, str, str]:
    if env.get("PORTCULLIS_DATABASE_URL"):
        raise NotPlain("a database is set")
    event = json.loads(stdin, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    if type(event) is not dict or type(event.get("tool_name")) is not str or not event["tool_name"]:
        raise NotPlain("unexpected event")
    args = event.get("tool_input", {})
    if type(args) is not dict:
        raise NotPlain("unexpected tool_input")
    json.dumps(args, ensure_ascii=False).encode("utf-8")  # a lone surrogate raises here: left to the full hook
    project = env.get("CLAUDE_PROJECT_DIR")
    path = Path(env.get("PORTCULLIS_POLICY") or (Path(project) / DEFAULT_POLICY if project else DEFAULT_POLICY))
    policy = read_policy(path.read_bytes().decode("utf-8"))
    tenant = env.get("PORTCULLIS_TENANT") or DEFAULT_TENANT
    call = SimpleNamespace(tenant_id=tenant, tool=_text(event["tool_name"]), args=args, definition_sha256=None)
    state = SimpleNamespace(steps=0, tokens=0, seconds=0, history=(), spend=0, tenant_spend=0)  # RunState()
    outcome = decide(policy, call, state)
    if outcome.verdict is Verdict.DENY:
        return 2, "", f"Portcullis [{outcome.rule}]: {outcome.reason}\n"
    if outcome.verdict is Verdict.ASK:
        answer = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "ask",
                "permissionDecisionReason": f"Portcullis [{outcome.rule}]: {outcome.reason}",
            }
        }
        return 0, json.dumps(answer), ""
    return 0, "", ""


def run(stdin: bytes, env: Mapping[str, str]) -> tuple[int, str, str]:
    """Same contract as portcullis.hook.run: (exit code, stdout, stderr); exit 2 blocks the call."""
    try:
        return _fast(stdin, env)
    except Exception:  # NotPlain, or anything unexpected: the full hook decides, and fails closed
        from portcullis import hook

        return hook.run(stdin, env)


def main() -> int:  # pragma: no cover - the same I/O wrapper as portcullis.hook.main
    import sys

    try:
        code, out, err = run(sys.stdin.buffer.read(), os.environ)
        sys.stdout.write(out)
        sys.stdout.flush()
        sys.stderr.write(err)
        sys.stderr.flush()
        return code
    except BaseException:  # exit 1 would let the call through
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
