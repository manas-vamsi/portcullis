"""Deterministic policy engine: the validated policy models and `decide`.

Pure: no I/O, no clock, no LLM. The caller supplies the run state (including
elapsed time), so the same inputs always give the same decision.

Trust boundary: the model controls only `tool` and `args`. The trusted runtime
must fill `tenant_id` (from the authenticated identity) and `definition_sha256`
(hashed from the live tool definition). Never copy either from model output.

The checks themselves live in `portcullis.rules` (standard library only), shared with the fast hook.
"""

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from portcullis import rules
from portcullis.rules import (  # noqa: F401  (re-exported: callers import these from here)
    HALTING_RULES,
    MAX_PATTERN_INPUT,
    Risk,
    Verdict,
    definition_hash,
    fingerprint,
)


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
    # Cap on what one tenant spends across ALL its runs in a rolling 24 hours, so starting a new run does not
    # reset it. Only the server tracks this (it needs a place that sees every run); the library agent loop and
    # the hook do not, and treat the tenant's spend as zero. None means no cap.
    max_spend_per_day: float | None = Field(default=None, ge=0, allow_inf_nan=False)


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
    tenant_spend: float = Field(default=0, ge=0, allow_inf_nan=False)  # the tenant's spend over the last 24 hours


class Decision(_Frozen):
    verdict: Verdict
    rule: str
    reason: str


def decide(policy: Policy, call: ToolCall, state: RunState) -> Decision:
    """Never raises: any failure while deciding is a deny, so a crash can never read as allow."""
    outcome = rules.decide(policy, call, state)
    return Decision(verdict=outcome.verdict, rule=outcome.rule, reason=outcome.reason)


def is_sensitive(policy: Policy, call: ToolCall) -> bool:
    """Whether the call matches any of its tool's sensitive-argument patterns, even if another rule decided it first."""
    return rules.is_sensitive(policy, call)


def spend_amount(policy: Policy, call: ToolCall) -> float:
    """What a call that ran adds to the run's spend. Zero when the tool declares no amount or it is invalid."""
    return rules.spend_amount(policy, call)
