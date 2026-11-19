"""A run's state, rebuilt from its audit rows. The server and the Claude Code hook both use this.

Nothing about a run is stored apart from its hash-chained audit log: its steps, its loop history,
its spend, the tokens its runtime reported and the time it has been running are all folded from the
rows on every request. Callers verify the chain first, so what is folded can be trusted.

Events that matter: `decision` (one per call, written first), `approval` (a human's answer to an
`ask`), `executed` (the runtime saying a call it was allowed to make has been made). Other events
are ignored.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from portcullis.policy import HALTING_RULES, RunState, Verdict


class InconsistentRun(Exception):
    """An answer or execution refers to a call that was never decided: the rows do not tell a coherent story."""


@dataclass
class CallRecord:
    decision: dict[str, Any]
    decided_at: datetime
    approval: dict[str, Any] | None = None
    approved_at: datetime | None = None
    executed: dict[str, Any] | None = None


REQUIRED = {
    "decision": ("call_id", "tool", "verdict", "rule", "reason", "request_sha256", "amount"),
    "approval": ("call_id", "approved"),
    "executed": ("call_id",),
}


def fold(rows: Sequence[Mapping[str, Any]]) -> dict[str, CallRecord]:
    """The run's calls in the order they were decided, with their answers.

    Any row that is not shaped like the ones this code writes is InconsistentRun, never a KeyError or a
    TypeError, so a damaged or hostile row stops the run cleanly instead of crashing a request."""
    calls: dict[str, CallRecord] = {}
    try:
        for row in rows:
            event = row["event"]
            if event not in REQUIRED:
                continue
            payload = json.loads(row["payload"])
            missing = [key for key in REQUIRED[event] if key not in payload]
            if missing:
                raise InconsistentRun(f"{event} row without {missing}")
            when = datetime.fromisoformat(row["ts"])
            if event == "decision":
                calls[payload["call_id"]] = CallRecord(payload, when)
                continue
            record = calls.get(payload["call_id"])
            if record is None:
                raise InconsistentRun(payload["call_id"])
            if event == "approval":
                record.approval, record.approved_at = payload, when
            else:
                record.executed = payload
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise InconsistentRun(type(exc).__name__) from exc
    return calls


def counts_as_spent(record: CallRecord) -> bool:
    return record.decision["verdict"] == Verdict.ALLOW or bool(record.approval and record.approval["approved"])


def spent(calls: Mapping[str, CallRecord]) -> float:
    return float(sum(r.decision["amount"] for r in calls.values() if counts_as_spent(r)))


def tokens(calls: Mapping[str, CallRecord]) -> int:
    """Tokens the runtime reported. Rows written before tokens existed count as zero."""
    return sum(int(r.decision.get("tokens", 0)) for r in calls.values())


def elapsed_seconds(calls: Mapping[str, CallRecord], now: datetime) -> float:
    """Time since the first call, leaving out time a human took to answer an ask.

    Only answered asks are left out, and overlapping waits are counted once. An ask nobody has answered
    does NOT stop the clock: calls are not blocked while one is open, so an agent could otherwise open an
    ask it knows will go unanswered and keep working with its time budget frozen."""
    if not calls:
        return 0.0
    start = min(r.decided_at for r in calls.values())
    waits = sorted(
        (r.decided_at, r.approved_at)
        for r in calls.values()
        if r.decision["verdict"] == Verdict.ASK and r.approved_at is not None and r.approved_at > r.decided_at
    )
    merged: list[list[datetime]] = []
    for began, ended in waits:
        if merged and began <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], ended)
        else:
            merged.append([began, ended])
    waited = sum((ended - began).total_seconds() for began, ended in merged)
    return max(0.0, (now - start).total_seconds() - waited)


def run_state(calls: Mapping[str, CallRecord], now: datetime, tenant_spend: float = 0.0) -> RunState:
    return RunState(
        steps=len(calls),
        tokens=tokens(calls),
        seconds=elapsed_seconds(calls, now),
        history=tuple(r.decision["request_sha256"] for r in calls.values()),
        spend=spent(calls),
        tenant_spend=tenant_spend,
    )


def halted(calls: Mapping[str, CallRecord]) -> bool:
    return any(r.decision["rule"] in HALTING_RULES for r in calls.values())


def status(record: CallRecord) -> str:
    if record.executed is not None:
        return "executed"
    verdict = record.decision["verdict"]
    if verdict == Verdict.ALLOW:
        return "allowed"
    if verdict == Verdict.DENY:
        return "denied"
    if record.approval is None:
        return "pending_approval"
    return "approved" if record.approval["approved"] else "approval_denied"


def view(record: CallRecord) -> dict[str, Any]:
    """What a caller is told about a call."""
    decision = record.decision
    shown = {
        "call_id": decision["call_id"],
        "status": status(record),
        "rule": decision["rule"],
        "reason": decision["reason"],
        "halt": decision["rule"] in HALTING_RULES,
    }
    if "args_preview" in decision:  # only calls that wait for a human carry one
        shown["args_preview"] = decision["args_preview"]
        shown["args_truncated"] = decision["args_truncated"]
    if record.approval and record.approval.get("blocked_by"):
        shown["blocked_by"] = record.approval["blocked_by"]
    return shown
