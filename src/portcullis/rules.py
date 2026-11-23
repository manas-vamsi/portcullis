"""The policy checks themselves, in the standard library only.

Everything here reads its inputs by attribute (`policy.tenants`, `rule.deny_if`, `call.args`, ...), so it
works on the validated pydantic models in `portcullis.policy` and on the plain objects the fast Claude Code
hook builds (`portcullis.quickhook`), which must start without loading pydantic. There is one copy of the
rules: both of those call this module.

Pure: no I/O, no clock, no LLM. See `portcullis.policy` for the trust boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:  # names for the annotations only: importing them would load pydantic
    from portcullis.policy import ArgPattern, Budget, Policy, RunState, ToolCall, ToolRule

MAX_PATTERN_INPUT = 10_000  # longer inspected arguments are denied: bounds regex cost and blocks padding


# Rules after which a run should stop: the caller is out of budget, looping, or has no policy at all.
HALTING_RULES = frozenset({"budget_exceeded", "loop_detected", "unknown_tenant"})


class Risk(StrEnum):
    READ = "read"
    WRITE = "write"
    IRREVERSIBLE = "irreversible"


class Verdict(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class Outcome(NamedTuple):
    verdict: Verdict
    rule: str
    reason: str


def fingerprint(tool: str, args: dict[str, Any]) -> str:
    # default=str is best effort for non-JSON values (args normally come from JSON).
    # Any remaining failure is caught by decide() and becomes a deny.
    canonical = json.dumps({"tool": tool, "args": args}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def definition_hash(definition: Mapping[str, Any]) -> str:
    """Hash of a tool definition (name, description, input schema: whatever the model is shown).

    Pin the hash of a definition you reviewed in ToolRule.pinned_sha256. Strict JSON, no fallback to
    str(): a definition that cannot be hashed exactly raises instead of hashing something vaguer.
    """
    canonical = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _decision(verdict: Verdict, rule: str, reason: str) -> Outcome:
    return Outcome(verdict, rule, reason)


def _check_pin(rule: ToolRule, call: ToolCall) -> Outcome | None:
    if rule.pinned_sha256 is None or call.definition_sha256 == rule.pinned_sha256:
        return None
    return _decision(
        Verdict.DENY, "definition_drift", f"definition of {call.tool!r} does not match its pinned hash"
    )


def _check_budget(budget: Budget, state: RunState) -> Outcome | None:
    used_vs_limit = (
        ("steps", state.steps, budget.max_steps),
        ("tokens", state.tokens, budget.max_tokens),
        ("seconds", state.seconds, budget.max_seconds),
    )
    for name, used, limit in used_vs_limit:
        if used >= limit:
            return _decision(Verdict.DENY, "budget_exceeded", f"{name} budget used up ({used} of {limit})")
    return None


def _check_loop(budget: Budget, call: ToolCall, state: RunState) -> Outcome | None:
    # ponytail: counts identical calls over the whole run, so legitimate polling can trip it;
    # use a sliding window if that bites.
    seen = state.history.count(fingerprint(call.tool, call.args))
    if seen < budget.max_identical_calls:
        return None
    return _decision(
        Verdict.DENY, "loop_detected", f"{call.tool!r} already called {seen} times with the same arguments"
    )


def is_valid_amount(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0  # ints are always finite; math.isfinite overflows on huge ones
    return isinstance(value, float) and math.isfinite(value) and value >= 0


def _check_amount(rule: ToolRule, call: ToolCall) -> Outcome | None:
    threshold = rule.approve_above
    if threshold is None or is_valid_amount(call.args.get(threshold.arg)):
        return None
    return _decision(
        Verdict.DENY, "invalid_amount", f"argument {threshold.arg!r} is missing or not a valid amount"
    )


def _check_spend(budget: Budget, rule: ToolRule, call: ToolCall, state: RunState) -> Outcome | None:
    threshold = rule.approve_above
    if threshold is None:
        return None
    amount = call.args[threshold.arg]  # validated by _check_amount
    if budget.max_spend is not None and state.spend + amount > budget.max_spend:
        return _decision(
            Verdict.DENY,
            "spend_cap_exceeded",
            f"spend so far {state.spend} plus {amount} would pass the run's cap of {budget.max_spend}",
        )
    if budget.max_spend_per_day is not None and state.tenant_spend + amount > budget.max_spend_per_day:
        return _decision(
            Verdict.DENY,
            "tenant_spend_cap_exceeded",
            f"the tenant's spend today {state.tenant_spend} plus {amount} would pass its daily cap of {budget.max_spend_per_day}",
        )
    return None


def is_sensitive(policy: Policy, call: ToolCall) -> bool:
    """Whether the call matches any of its tool's sensitive-argument patterns, even if another rule decided it first."""
    tenant = policy.tenants.get(call.tenant_id)
    rule = tenant.tools.get(call.tool) if tenant else None
    return rule is not None and _first_match(rule.ask_if, call) is not None


def spend_amount(policy: Policy, call: ToolCall) -> float:
    """What a call that ran adds to the run's spend. Zero when the tool declares no amount or it is invalid."""
    tenant = policy.tenants.get(call.tenant_id)
    rule = tenant.tools.get(call.tool) if tenant else None
    if rule is None or rule.approve_above is None:
        return 0
    amount = call.args.get(rule.approve_above.arg)
    return amount if is_valid_amount(amount) else 0


def _argument_text(call: ToolCall, name: str) -> str | None:
    value = call.args.get(name)
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _first_match(patterns: tuple[ArgPattern, ...], call: ToolCall) -> ArgPattern | None:
    for pattern in patterns:
        text = _argument_text(call, pattern.arg)
        if text is not None and re.search(pattern.regex, text):
            return pattern
    return None


def _check_arguments(rule: ToolRule, call: ToolCall) -> Outcome | None:
    """Deny on an oversized inspected argument or a forbidden pattern. Never echoes the pattern itself."""
    for pattern in (*rule.deny_if, *rule.ask_if):
        text = _argument_text(call, pattern.arg)
        if text is None and pattern.required:
            return _decision(Verdict.DENY, "missing_argument", f"required argument {pattern.arg!r} is missing")
        if text is not None and len(text) > MAX_PATTERN_INPUT:
            return _decision(Verdict.DENY, "argument_too_long", f"argument {pattern.arg!r} is too long to inspect")
    hit = _first_match(rule.deny_if, call)
    if hit:
        return _decision(Verdict.DENY, "forbidden_argument", f"argument {hit.arg!r} matches a forbidden pattern")
    return None


def _check_sensitive(rule: ToolRule, call: ToolCall) -> Outcome | None:
    hit = _first_match(rule.ask_if, call)
    if hit:
        return _decision(Verdict.ASK, "sensitive_argument", f"argument {hit.arg!r} matches a sensitive pattern")
    return None


def _check_approval(rule: ToolRule, call: ToolCall) -> Outcome | None:
    if rule.risk is Risk.IRREVERSIBLE:
        return _decision(Verdict.ASK, "irreversible_action", f"{call.tool!r} cannot be undone")
    threshold = rule.approve_above
    # The per-call limit on one named argument; the run's total is capped by _check_spend.
    if threshold is not None and call.args[threshold.arg] > threshold.limit:  # validated by _check_amount
        return _decision(
            Verdict.ASK,
            "approval_required",
            f"{threshold.arg!r} {call.args[threshold.arg]} is above {threshold.limit}",
        )
    return None


def _decide(policy: Policy, call: ToolCall, state: RunState) -> Outcome:
    tenant = policy.tenants.get(call.tenant_id)
    if tenant is None:
        return _decision(Verdict.DENY, "unknown_tenant", f"no policy for tenant {call.tenant_id!r}")
    # Budget and loop checks come first and cover every call, even one for a tool that is not
    # allowed. Otherwise a model that keeps calling forbidden tools would never be stopped.
    # `or` stops at the first hit. (An Outcome is a non-empty tuple, so always truthy.)
    over_limit = _check_budget(tenant.budget, state) or _check_loop(tenant.budget, call, state)
    if over_limit:
        return over_limit
    rule = tenant.tools.get(call.tool)
    if rule is None:
        return _decision(Verdict.DENY, "tool_not_allowed", f"{call.tool!r} is not on the allowlist")

    # Order matters: all denies come before the ask, so deny beats ask.
    return (
        _check_pin(rule, call)
        or _check_amount(rule, call)
        or _check_spend(tenant.budget, rule, call, state)
        or _check_arguments(rule, call)
        or _check_approval(rule, call)
        or _check_sensitive(rule, call)
        or _decision(Verdict.ALLOW, "allowed", f"{rule.risk} tool {call.tool!r} is allowed")
    )


def decide(policy: Policy, call: ToolCall, state: RunState) -> Outcome:
    """Never raises: any failure while deciding is a deny, so a crash can never read as allow."""
    try:
        return _decide(policy, call, state)
    except Exception as exc:  # deliberate catch-all, fail closed
        return _decision(Verdict.DENY, "evaluation_error", f"could not evaluate: {type(exc).__name__}")
