"""Deterministic policy engine.

Pure: no I/O, no clock, no LLM. The caller supplies the run state (including
elapsed time), so the same inputs always give the same decision.

Trust boundary: the model controls only `tool` and `args`. The trusted runtime
must fill `tenant_id` (from the authenticated identity) and `definition_sha256`
(hashed from the live tool definition). Never copy either from model output.
"""

import hashlib
import json
import math
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Threshold(_Frozen):
    """Ask a human when the numeric argument `arg` is above `limit`."""

    arg: str
    limit: float = Field(ge=0, allow_inf_nan=False)


class ArgPattern(_Frozen):
    """A regex searched in the string form of the named argument. Add (?i) for case-insensitive.

    Patterns are trusted policy config run against model-controlled text, so avoid nested
    quantifiers: Python's re has no timeout.
    """

    arg: str
    regex: str
    required: bool = False  # if true, a call without this argument is denied, so a typo in `arg` cannot switch the rule off

    @field_validator("regex")
    @classmethod
    def _must_compile(cls, regex: str) -> str:
        try:
            re.compile(regex)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
        return regex


class ToolRule(_Frozen):
    risk: Risk
    pinned_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    approve_above: Threshold | None = None
    deny_if: tuple[ArgPattern, ...] = ()  # a match denies the call
    ask_if: tuple[ArgPattern, ...] = ()  # a match asks a human


class Budget(_Frozen):
    """Required on every tenant, so a budget cannot be forgotten."""

    max_steps: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    max_seconds: float = Field(gt=0, allow_inf_nan=False)
    max_identical_calls: int = Field(gt=0)
    # Cap on the total amount spent in one run, counted from the amount argument of tools that declare an
    # approve_above threshold. None means no cap. A human approval cannot exceed it. Per run, not per
    # tenant or per day: a new run starts at zero.
    max_spend: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class TenantPolicy(_Frozen):
    budget: Budget
    tools: dict[str, ToolRule]


class Policy(_Frozen):
    tenants: dict[str, TenantPolicy]


class ToolCall(_Frozen):
    tenant_id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    definition_sha256: str | None = None  # hash of the tool definition as advertised right now


class RunState(_Frozen):
    # Constrained so NaN or negative counters can never slip under a limit.
    steps: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    history: tuple[str, ...] = ()  # fingerprints of calls already made in this run
    spend: float = Field(default=0, ge=0, allow_inf_nan=False)  # total amount of the calls that already ran


class Decision(_Frozen):
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


def _decision(verdict: Verdict, rule: str, reason: str) -> Decision:
    return Decision(verdict=verdict, rule=rule, reason=reason)


def _check_pin(rule: ToolRule, call: ToolCall) -> Decision | None:
    if rule.pinned_sha256 is None or call.definition_sha256 == rule.pinned_sha256:
        return None
    return _decision(
        Verdict.DENY, "definition_drift", f"definition of {call.tool!r} does not match its pinned hash"
    )


def _check_budget(budget: Budget, state: RunState) -> Decision | None:
    used_vs_limit = (
        ("steps", state.steps, budget.max_steps),
        ("tokens", state.tokens, budget.max_tokens),
        ("seconds", state.seconds, budget.max_seconds),
    )
    for name, used, limit in used_vs_limit:
        if used >= limit:
            return _decision(Verdict.DENY, "budget_exceeded", f"{name} budget used up ({used} of {limit})")
    return None


def _check_loop(budget: Budget, call: ToolCall, state: RunState) -> Decision | None:
    # ponytail: counts identical calls over the whole run, so legitimate polling can trip it;
    # use a sliding window if that bites.
    seen = state.history.count(fingerprint(call.tool, call.args))
    if seen < budget.max_identical_calls:
        return None
    return _decision(
        Verdict.DENY, "loop_detected", f"{call.tool!r} already called {seen} times with the same arguments"
    )


def _is_valid_amount(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0  # ints are always finite; math.isfinite overflows on huge ones
    return isinstance(value, float) and math.isfinite(value) and value >= 0


def _check_amount(rule: ToolRule, call: ToolCall) -> Decision | None:
    threshold = rule.approve_above
    if threshold is None or _is_valid_amount(call.args.get(threshold.arg)):
        return None
    return _decision(
        Verdict.DENY, "invalid_amount", f"argument {threshold.arg!r} is missing or not a valid amount"
    )


def _check_spend(budget: Budget, rule: ToolRule, call: ToolCall, state: RunState) -> Decision | None:
    threshold = rule.approve_above
    if budget.max_spend is None or threshold is None:
        return None
    amount = call.args[threshold.arg]  # validated by _check_amount
    if state.spend + amount <= budget.max_spend:
        return None
    return _decision(
        Verdict.DENY,
        "spend_cap_exceeded",
        f"spend so far {state.spend} plus {amount} would pass the run's cap of {budget.max_spend}",
    )


def spend_amount(policy: Policy, call: ToolCall) -> float:
    """What a call that ran adds to the run's spend. Zero when the tool declares no amount or it is invalid."""
    tenant = policy.tenants.get(call.tenant_id)
    rule = tenant.tools.get(call.tool) if tenant else None
    if rule is None or rule.approve_above is None:
        return 0
    amount = call.args.get(rule.approve_above.arg)
    return amount if _is_valid_amount(amount) else 0


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


def _check_arguments(rule: ToolRule, call: ToolCall) -> Decision | None:
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


def _check_sensitive(rule: ToolRule, call: ToolCall) -> Decision | None:
    hit = _first_match(rule.ask_if, call)
    if hit:
        return _decision(Verdict.ASK, "sensitive_argument", f"argument {hit.arg!r} matches a sensitive pattern")
    return None


def _check_approval(rule: ToolRule, call: ToolCall) -> Decision | None:
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


def _decide(policy: Policy, call: ToolCall, state: RunState) -> Decision:
    tenant = policy.tenants.get(call.tenant_id)
    if tenant is None:
        return _decision(Verdict.DENY, "unknown_tenant", f"no policy for tenant {call.tenant_id!r}")
    # Budget and loop checks come first and cover every call, even one for a tool that is not
    # allowed. Otherwise a model that keeps calling forbidden tools would never be stopped.
    # `or` stops at the first hit. (Decision is a pydantic model and always truthy.)
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


def decide(policy: Policy, call: ToolCall, state: RunState) -> Decision:
    """Never raises: any failure while deciding is a deny, so a crash can never read as allow."""
    try:
        return _decide(policy, call, state)
    except Exception as exc:  # deliberate catch-all, fail closed
        return _decision(Verdict.DENY, "evaluation_error", f"could not evaluate: {type(exc).__name__}")
