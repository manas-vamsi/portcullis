"""The property-maintenance triage demo: synthetic data, scripted model, real policy, ledger and audit log."""

from dataclasses import replace

import pytest

from portcullis.audit import Verification, verify
from portcullis.demo import SCENARIOS, PropertyOps, _print_result, demo_policy, find_emergency, main, triage
from portcullis.llm import FakeLLM
from portcullis.policy import RunState, ToolCall, Verdict, decide


def approve(_payload: dict) -> dict:
    return {"approved": True, "by": "alice (property manager)"}


def deny(_payload: dict) -> dict:
    return {"approved": False, "by": "alice (property manager)"}


def run(conn, run_id, name, approver=approve, ops=None):
    scenario = SCENARIOS[name]
    ops = ops or PropertyOps()
    llm = FakeLLM(scenario.script)
    return triage(conn, llm, scenario.request, run_id, ops, approver=approver), ops, llm


def event_names(conn, run_id) -> list[str]:
    rows = conn.execute("SELECT event FROM audit_log WHERE run_id = %s ORDER BY idx", (run_id,)).fetchall()
    return [event for (event,) in rows]


@pytest.mark.parametrize(
    "text",
    ["There is a gas leak in unit 3", "I SMELL GAS in the hallway", "the kitchen is on fire", "carbon monoxide alarm is going off", "a  burst   pipe is flooding unit 9"],
)
def test_emergency_wording_is_recognised(text):
    assert find_emergency(text) is not None


@pytest.mark.parametrize(
    "text",
    ["The fireplace needs cleaning", "kitchen faucet drips", "the gas stove knob is loose", "please fire up the boiler schedule"],
)
def test_ordinary_requests_are_not_emergencies(text):
    assert find_emergency(text) is None


def test_an_emergency_skips_the_model_and_every_tool_and_goes_to_a_human(conn, run_id):
    result, ops, llm = run(conn, run_id, "emergency")

    assert result.status == "escalated"
    assert llm.calls == []  # the model was never asked
    assert ops.orders == {} and ops.messages == []
    assert event_names(conn, run_id) == ["emergency_escalation"]
    assert verify(conn, run_id).ok


def test_a_routine_request_runs_end_to_end_without_asking_anyone(conn, run_id):
    result, ops, _ = run(conn, run_id, "routine", approver=None)

    assert result.status == "done"
    assert len(ops.orders) == 1 and len(ops.messages) == 1
    assert "approval" not in event_names(conn, run_id)
    assert verify(conn, run_id).ok


def test_an_expensive_job_waits_for_a_human_and_nothing_is_ordered_meanwhile(conn, run_id):
    result, ops, _ = run(conn, run_id, "expensive", approver=None)

    assert result.status == "waiting_for_approval"
    assert result.pending["tool"] == "create_work_order"
    assert ops.orders == {}


def test_an_approved_expensive_job_is_ordered_once_and_the_approver_is_in_the_audit_log(conn, run_id):
    result, ops, _ = run(conn, run_id, "expensive")

    assert result.status == "done"
    [order] = ops.orders.values()
    assert order["cost"] == 1450
    assert [a["approved_by"] for a in result.approvals] == ["alice (property manager)"]
    assert event_names(conn, run_id).count("approval") == 1
    assert verify(conn, run_id).ok


def test_a_denied_expensive_job_creates_nothing(conn, run_id):
    result, ops, _ = run(conn, run_id, "expensive_denied", approver=deny)

    assert result.status == "done"
    assert ops.orders == {}
    assert "approval_denied" in " ".join(result.tool_log)


def test_a_manipulated_model_is_refused_twice_and_changes_nothing(conn, run_id):
    result, ops, _ = run(conn, run_id, "attack", approver=None)

    assert ops.orders == {} and ops.messages == []
    assert "tool_not_allowed" in " ".join(result.tool_log)
    assert "invalid_amount" in " ".join(result.tool_log)  # a string amount cannot dodge the spend threshold


def test_the_spend_threshold_is_exactly_five_hundred():
    policy = demo_policy()

    def verdict(cost):
        call = ToolCall(tenant_id="demo-property", tool="create_work_order", args={"cost": cost})
        return decide(policy, call, RunState()).verdict

    assert verdict(500) is Verdict.ALLOW
    assert verdict(500.01) is Verdict.ASK


def test_work_orders_are_idempotent_by_key_and_cancelling_twice_is_harmless():
    ops = PropertyOps()
    create = ops.tools()["create_work_order"]
    args = {"vendor_id": "V-101", "unit": "12", "description": "faucet", "cost": 180}

    first, again, other = create(args, "key-1"), create(args, "key-1"), create(args, "key-2")

    assert first == again and first != other
    assert len(ops.orders) == 2
    cancel = ops.tools()["cancel_work_order"]
    cancel({"order_id": first["order_id"]}, "k-a")
    cancel({"order_id": first["order_id"]}, "k-b")
    assert ops.orders[first["order_id"]]["status"] == "cancelled"


def test_the_command_line_runs_a_scenario_and_verifies_its_audit_chain(postgres_url, conn, capsys, monkeypatch):
    monkeypatch.setenv("PORTCULLIS_DATABASE_URL", postgres_url)

    code = main(["--scenario", "routine"])

    out = capsys.readouterr().out
    assert code == 0
    assert "routine" in out and "audit chain verified" in out
    for table in ("audit_log", "tool_effects"):  # the demo's own rows, not part of this test's fixtures
        conn.execute(f"DELETE FROM {table} WHERE run_id LIKE 'demo-%'")


def test_the_command_line_shows_who_approved_an_expensive_job(postgres_url, conn, capsys, monkeypatch):
    monkeypatch.setenv("PORTCULLIS_DATABASE_URL", postgres_url)

    code = main(["--scenario", "expensive"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Dana (property manager) approved create_work_order" in out
    for table in ("audit_log", "tool_effects"):
        conn.execute(f"DELETE FROM {table} WHERE run_id LIKE 'demo-%'")


def test_a_pending_approval_and_a_broken_chain_are_shown_plainly(conn, run_id, capsys):
    waiting, _, _ = run(conn, run_id, "expensive", approver=None)

    _print_result("expensive", SCENARIOS["expensive"], waiting)
    _print_result("expensive", SCENARIOS["expensive"], replace(waiting, audit=Verification(False, 3, "x", "row 1 was changed")))

    out = capsys.readouterr().out
    assert "waiting for a human: create_work_order" in out
    assert "CHAIN BROKEN: row 1 was changed" in out


def test_cancelling_an_order_that_does_not_exist_is_reported_not_raised():
    result = PropertyOps().tools()["cancel_work_order"]({"order_id": "WO-9999"}, "k")

    assert result == {"cancelled": False, "reason": "unknown order"}


def test_the_command_line_refuses_a_database_that_is_not_named_for_this_project(capsys, monkeypatch):
    monkeypatch.setenv("PORTCULLIS_DATABASE_URL", "postgresql://u:p@127.0.0.1:5432/some_other_db")

    assert main(["--scenario", "routine"]) == 2
    assert "must start with 'portcullis'" in capsys.readouterr().err


def test_the_command_line_needs_a_database_url(capsys, monkeypatch):
    monkeypatch.delenv("PORTCULLIS_DATABASE_URL", raising=False)

    assert main(["--scenario", "routine"]) == 2
    assert "PORTCULLIS_DATABASE_URL" in capsys.readouterr().err
