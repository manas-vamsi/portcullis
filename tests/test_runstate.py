"""Run state rebuilt from audit rows: the same code the server and the Claude Code hook use."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from portcullis.runstate import InconsistentRun, fold, run_state, status, view

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def decision(call_id, verdict="allow", rule="allowed", amount=0, tokens=0, request="r", seconds=0.0, **extra) -> dict:
    payload = {"call_id": call_id, "tool": "t", "verdict": verdict, "rule": rule, "reason": "why",
               "request_sha256": request + call_id, "amount": amount, "tokens": tokens} | extra
    return {"event": "decision", "payload": json.dumps(payload), "ts": at(seconds).isoformat()}


def approval(call_id, approved, seconds, **extra) -> dict:
    payload = {"call_id": call_id, "tool": "t", "approved": approved, "approved_by": "alice"} | extra
    return {"event": "approval", "payload": json.dumps(payload), "ts": at(seconds).isoformat()}


def executed(call_id, seconds=0.0) -> dict:
    return {"event": "executed", "payload": json.dumps({"call_id": call_id, "tool": "t"}), "ts": at(seconds).isoformat()}


def test_the_calls_come_back_in_order_with_their_answers():
    calls = fold([decision("a"), decision("b", "ask", "approval_required"), approval("b", True, 5), executed("a", 2)])
    assert list(calls) == ["a", "b"]
    assert calls["a"].executed is not None and calls["b"].approval["approved"] is True


def test_events_that_are_not_part_of_a_calls_story_are_ignored():
    rows = [decision("a"), {"event": "emergency_escalation", "payload": "{}", "ts": at(0).isoformat()}]
    assert list(fold(rows)) == ["a"]


@pytest.mark.parametrize("row", [approval("ghost", True, 1), executed("ghost", 1)], ids=["approval", "executed"])
def test_an_answer_or_execution_for_a_call_that_was_never_decided_is_inconsistent(row):
    with pytest.raises(InconsistentRun):
        fold([decision("a"), row])


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([decision("a")], "allowed"),
        ([decision("a", "deny", "tool_not_allowed")], "denied"),
        ([decision("a", "ask", "approval_required")], "pending_approval"),
        ([decision("a", "ask", "approval_required"), approval("a", True, 1)], "approved"),
        ([decision("a", "ask", "approval_required"), approval("a", False, 1)], "approval_denied"),
        ([decision("a"), executed("a")], "executed"),
        ([decision("a", "ask", "approval_required"), approval("a", True, 1), executed("a", 2)], "executed"),
    ],
)
def test_the_status_of_a_call(rows, expected):
    assert status(fold(rows)["a"]) == expected


def test_the_view_carries_the_rule_the_halt_flag_and_the_preview_for_a_waiting_call():
    ask = decision("a", "ask", "approval_required", args_preview='{"cost":900}', args_truncated=False)
    shown = view(fold([ask])["a"])
    assert shown == {"call_id": "a", "status": "pending_approval", "rule": "approval_required", "reason": "why",
                     "halt": False, "args_preview": '{"cost":900}', "args_truncated": False}
    assert "args_preview" not in view(fold([decision("b")])["b"])
    assert view(fold([decision("h", "deny", "loop_detected")])["h"])["halt"] is True
    blocked = fold([ask, approval("a", False, 1, blocked_by="run_halted")])["a"]
    assert view(blocked)["blocked_by"] == "run_halted"


def test_steps_history_and_spend_are_rebuilt_and_only_allowed_or_approved_calls_spend():
    rows = [
        decision("a", amount=100),
        decision("b", "deny", "forbidden_argument", amount=500),
        decision("c", "ask", "approval_required", amount=900),  # waiting: not spent yet
        decision("d", "ask", "approval_required", amount=300),
        approval("d", True, 3),
        decision("e", "ask", "approval_required", amount=700),
        approval("e", False, 4),
    ]
    state = run_state(fold(rows), at(10))
    assert (state.steps, state.spend) == (5, 400.0)  # a (100) and the approved d (300)
    assert state.history == ("ra", "rb", "rc", "rd", "re")


def test_tokens_are_summed_and_older_rows_without_a_count_count_as_zero():
    old = decision("b")
    payload = json.loads(old["payload"])
    del payload["tokens"]
    old["payload"] = json.dumps(payload)
    assert run_state(fold([decision("a", tokens=30), old, decision("c", tokens=12)]), at(1)).tokens == 42


def test_elapsed_time_runs_from_the_first_call_and_leaves_out_time_spent_waiting_for_a_human():
    rows = [decision("a", seconds=0), decision("b", "ask", "approval_required", seconds=10), approval("b", True, 70)]
    assert run_state(fold(rows), at(100)).seconds == 100 - 60  # a person took 60 seconds to answer


def test_an_ask_nobody_answers_never_stops_the_clock():
    # Calls are not blocked while an ask is open, so an agent could otherwise open one it knows will go
    # unanswered and keep working with its time budget frozen.
    rows = [decision("a", seconds=0), decision("b", "ask", "approval_required", seconds=10)]
    assert run_state(fold(rows), at(500)).seconds == 500
    assert run_state(fold(rows), at(100000)).seconds == 100000


def test_overlapping_waits_are_counted_once():
    rows = [
        decision("a", "ask", "approval_required", seconds=10),
        decision("b", "ask", "approval_required", seconds=11),
        approval("a", True, 60),
        approval("b", True, 60),
    ]
    assert elapsed(rows, 100) == 40  # 90 seconds since the first call, of which one 50-second wait (not two)


def test_an_answer_stamped_before_its_ask_is_clock_skew_and_changes_nothing():
    rows = [decision("a", "ask", "approval_required", seconds=50), approval("a", True, 40)]
    assert elapsed(rows, 100) == 50


def test_separate_waits_are_each_left_out():
    rows = [
        decision("a", "ask", "approval_required", seconds=0),
        approval("a", True, 10),
        decision("b", "ask", "approval_required", seconds=20),
        approval("b", True, 50),
    ]
    assert elapsed(rows, 100) == 100 - 10 - 30


def test_an_empty_run_has_the_empty_state_and_the_tenant_spend_is_passed_through():
    state = run_state(fold([]), at(5), tenant_spend=42.5)
    assert (state.steps, state.tokens, state.seconds, state.spend, state.tenant_spend, state.history) == (0, 0, 0, 0, 42.5, ())


def elapsed(rows, now_seconds):
    return run_state(fold(rows), at(now_seconds)).seconds


@pytest.mark.parametrize(
    "row",
    [
        {"event": "decision", "payload": "[]", "ts": at(0).isoformat()},
        {"event": "decision", "payload": "not json", "ts": at(0).isoformat()},
        {"event": "decision", "payload": json.dumps({"call_id": "a"}), "ts": at(0).isoformat()},
        {"event": "decision", "payload": decision("a")["payload"], "ts": "yesterday"},
        {"event": "approval", "payload": json.dumps({"approved": True}), "ts": at(0).isoformat()},
        {"event": "executed", "payload": "7", "ts": at(0).isoformat()},
        {"event": "decision", "ts": at(0).isoformat()},
    ],
    ids=["list", "not_json", "missing_fields", "bad_timestamp", "approval_without_call", "number", "no_payload"],
)
def test_a_row_that_is_not_shaped_like_ours_is_an_inconsistent_run_never_a_crash(row):
    with pytest.raises(InconsistentRun):
        fold([row])


def test_a_clock_that_runs_backwards_never_gives_negative_time():
    assert run_state(fold([decision("a", seconds=100)]), at(0)).seconds == 0
