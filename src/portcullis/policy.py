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
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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


class ToolRule(_Frozen):
    risk: Risk
    pinned_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    approve_above: Threshold | None = None


class Budget(_Frozen):
    """Required on every tenant, so a budget cannot be forgotten."""

    max_steps: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    max_seconds: float = Field(gt=0, allow_inf_nan=False)
    max_identical_calls: int = Field(gt=0)


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


class Decision(_Frozen):
    verdict: Verdict
    rule: str
    reason: str


def fingerprint(tool: str, args: dict[str, Any]) -> str:
    # default=str is best effort for non-JSON values (args normally come from JSON).
    # Any remaining failure is caught by decide() and becomes a deny.
    canonical = json.dumps({"tool": tool, "args": args}, sort_keys=True, separators=(",", ":"), default=str)
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


def _check_approval(rule: ToolRule, call: ToolCall) -> Decision | None:
    if rule.risk is Risk.IRREVERSIBLE:
        return _decision(Verdict.ASK, "irreversible_action", f"{call.tool!r} cannot be undone")
    threshold = rule.approve_above
    # ponytail: per-call limit on one named argument; add a cumulative cap (RunState) if needed.
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
    rule = tenant.tools.get(call.tool)
    if rule is None:
        return _decision(Verdict.DENY, "tool_not_allowed", f"{call.tool!r} is not on the allowlist")

    # `or` stops at the first hit. Order matters: all denies come before the ask, so deny beats ask.
    # (Decision is a pydantic model and always truthy.)
    return (
        _check_pin(rule, call)
        or _check_budget(tenant.budget, state)
        or _check_loop(tenant.budget, call, state)
        or _check_amount(rule, call)
        or _check_approval(rule, call)
        or _decision(Verdict.ALLOW, "allowed", f"{rule.risk} tool {call.tool!r} is allowed")
    )


def decide(policy: Policy, call: ToolCall, state: RunState) -> Decision:
    """Never raises: any failure while deciding is a deny, so a crash can never read as allow."""
    try:
        return _decide(policy, call, state)
    except Exception as exc:  # deliberate catch-all, fail closed
        return _decision(Verdict.DENY, "evaluation_error", f"could not evaluate: {type(exc).__name__}")
