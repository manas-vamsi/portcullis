"""The HTTP server: authenticated decisions, approvals and the audit trail, all backed by Postgres.

The Gateway holds the logic and is tested directly (including concurrency); the HTTP layer is
tested for authentication, validation and error mapping.
"""

import json
import threading
from urllib.parse import quote
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient

from portcullis.audit import events, namespaced, verify
from portcullis.keys import KeyEntry, hash_key, new_key
from portcullis.policy import ArgPattern, Budget, Policy, Risk, TenantPolicy, Threshold, ToolRule, definition_hash
from portcullis.server import (
    Busy,
    ChainBroken,
    Conflict,
    Gateway,
    NotFound,
    app_from_env,
    create_app,
    lock_key,
    open_pool,
)

PIN = definition_hash({"name": "pinned_lookup", "description": "reviewed"})


def budget(**overrides) -> Budget:
    limits = {"max_steps": 8, "max_tokens": 1000, "max_seconds": 60, "max_identical_calls": 3, "max_spend": 1000}
    return Budget(**(limits | overrides))


def make_policy(**overrides) -> Policy:
    tools = {
        "check_vendor_availability": ToolRule(risk=Risk.READ),
        "create_work_order": ToolRule(
            risk=Risk.WRITE,
            approve_above=Threshold(arg="cost", limit=500),
            deny_if=(ArgPattern(arg="vendor_id", regex=r"^(?!acme-)", required=True),),
        ),
        "delete_property": ToolRule(risk=Risk.IRREVERSIBLE),
        "pinned_lookup": ToolRule(risk=Risk.READ, pinned_sha256=PIN),
    }
    globex = TenantPolicy(budget=budget(), tools={"check_vendor_availability": ToolRule(risk=Risk.READ)})
    return Policy(tenants={"acme": TenantPolicy(budget=budget(**overrides), tools=tools), "globex": globex})


@pytest.fixture
def pool(postgres_url):
    opened = open_pool(postgres_url)
    yield opened
    opened.close()


@pytest.fixture
def rid(conn):
    """A run name unique to this test; every audit row it created is removed afterwards."""
    name = f"srv-{uuid4()}"
    yield name
    conn.execute("DELETE FROM audit_log WHERE run_id LIKE %s", (f"%{name}%",))


@pytest.fixture
def gateway(pool):
    return Gateway(make_policy(), pool)


def order(cost=100, unit="1", **extra) -> dict:
    return {"tool": "create_work_order", "args": {"cost": cost, "vendor_id": "acme-V1", "unit": unit, **extra}}


def decide(gateway, run, call_id, tool="check_vendor_availability", args=None, tenant="acme", **kw):
    return gateway.decide_call(tenant, run, call_id, tool, args or {}, kw.get("definition_sha256"))


def place(gateway, run, call_id, cost=100, tenant="acme", unit=None):
    spec = order(cost, unit or call_id)
    return gateway.decide_call(tenant, run, call_id, spec["tool"], spec["args"], None)


# --- decisions ---


def test_an_allowed_call_is_answered_and_recorded_for_its_tenant(gateway, conn, rid):
    view = decide(gateway, rid, "c1", args={"trade": "plumber"})

    assert (view["status"], view["rule"], view["halt"]) == ("allowed", "allowed", False)
    [row] = events(conn, namespaced("acme", rid))
    assert (row["event"], row["tenant_id"], row["event_id"]) == ("decision", "acme", "c1:decision")


def test_a_call_to_an_unlisted_tool_is_denied(gateway, rid):
    view = decide(gateway, rid, "c1", tool="drop_database")
    assert (view["status"], view["rule"]) == ("denied", "tool_not_allowed")


def test_an_unknown_tenant_is_denied_and_halted(gateway, rid):
    view = decide(gateway, rid, "c1", tenant="ghost")
    assert (view["status"], view["rule"], view["halt"]) == ("denied", "unknown_tenant", True)


def test_a_call_above_the_threshold_waits_and_runs_only_once_approved(gateway, rid):
    assert place(gateway, rid, "c1", cost=900)["status"] == "pending_approval"
    assert gateway.call_status("acme", rid, "c1")["status"] == "pending_approval"

    answered = gateway.answer("acme", rid, "c1", True, "alice")

    assert answered["status"] == "approved"
    assert gateway.call_status("acme", rid, "c1")["status"] == "approved"


def test_a_denied_approval_is_recorded_with_who_denied_it(gateway, conn, rid):
    place(gateway, rid, "c1", cost=900)

    assert gateway.answer("acme", rid, "c1", False, "alice")["status"] == "approval_denied"

    approval = json.loads(events(conn, namespaced("acme", rid))[-1]["payload"])
    assert (approval["approved"], approval["approved_by"]) == (False, "alice")


def test_repeating_a_call_returns_the_same_answer_and_writes_nothing_new(gateway, conn, rid):
    first = decide(gateway, rid, "c1", args={"trade": "plumber"})
    again = decide(gateway, rid, "c1", args={"trade": "plumber"})

    assert again == first
    assert len(events(conn, namespaced("acme", rid))) == 1


def test_reusing_a_call_id_for_a_different_call_is_a_conflict(gateway, rid):
    decide(gateway, rid, "c1", args={"trade": "plumber"})
    with pytest.raises(Conflict):
        decide(gateway, rid, "c1", args={"trade": "electrician"})


def test_a_pinned_tool_with_the_wrong_definition_is_refused(gateway, rid):
    assert decide(gateway, rid, "c1", tool="pinned_lookup", definition_sha256="b" * 64)["rule"] == "definition_drift"
    assert decide(gateway, rid, "c2", tool="pinned_lookup", definition_sha256=PIN)["status"] == "allowed"


# --- run state: steps, loops, spend ---


def test_the_step_budget_halts_the_run_and_later_calls_are_not_even_recorded(gateway, conn, rid):
    for i in range(8):
        assert decide(gateway, rid, f"c{i}", args={"n": i})["status"] == "allowed"

    stop = decide(gateway, rid, "c8", args={"n": 8})
    assert (stop["rule"], stop["halt"]) == ("budget_exceeded", True)
    rows_after_halt = len(events(conn, namespaced("acme", rid)))

    for i in range(9, 20):
        later = decide(gateway, rid, f"c{i}", args={"n": i})
        assert (later["status"], later["rule"], later["halt"]) == ("denied", "run_halted", True)
    assert len(events(conn, namespaced("acme", rid))) == rows_after_halt  # a flood of calls cannot grow the log


def test_repeating_the_same_call_under_new_ids_is_caught_as_a_loop(gateway, rid):
    views = [decide(gateway, rid, f"c{i}", args={"trade": "plumber"}) for i in range(4)]
    assert [v["rule"] for v in views] == ["allowed", "allowed", "allowed", "loop_detected"]
    assert views[-1]["halt"] is True


def test_spend_is_added_up_across_calls_and_the_run_carries_on_after_a_refusal(gateway, rid):
    views = [place(gateway, rid, f"c{i}", cost=400) for i in range(3)]
    assert [v["rule"] for v in views] == ["allowed", "allowed", "spend_cap_exceeded"]
    assert views[-1]["halt"] is False
    assert decide(gateway, rid, "after", args={"trade": "plumber"})["status"] == "allowed"


def test_a_call_waiting_for_approval_adds_no_spend_until_it_is_approved(gateway, rid):
    place(gateway, rid, "big", cost=900)  # waiting
    assert place(gateway, rid, "small", cost=400)["status"] == "allowed"  # 900 is not counted yet

    answered = gateway.answer("acme", rid, "big", True, "alice")  # 400 + 900 would pass the 1000 cap

    assert (answered["status"], answered["blocked_by"]) == ("approval_denied", "spend_cap_exceeded")


def test_an_approval_for_a_tenant_that_the_policy_no_longer_has_is_refused(pool, rid):
    place(Gateway(make_policy(), pool), rid, "c1", cost=900)  # asked under the old policy
    without_acme = Policy(tenants={"globex": make_policy().tenants["globex"]})  # the server restarted with a new one

    answered = Gateway(without_acme, pool).answer("acme", rid, "c1", True, "alice")

    assert (answered["status"], answered["blocked_by"]) == ("approval_denied", "unknown_tenant")


def test_an_approval_within_the_cap_counts_toward_it(gateway, rid):
    place(gateway, rid, "big", cost=900)
    gateway.answer("acme", rid, "big", True, "alice")
    assert place(gateway, rid, "next", cost=200)["rule"] == "spend_cap_exceeded"


def test_concurrent_calls_cannot_overshoot_the_spend_cap(pool, conn, rid):
    gateway = Gateway(make_policy(max_steps=50), pool)
    views, barrier = [], threading.Barrier(8)

    def worker(i):
        barrier.wait()
        views.append(place(gateway, rid, f"c{i}", cost=400))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert sorted(v["status"] for v in views) == ["allowed"] * 2 + ["denied"] * 6  # 2 x 400 fits in 1000, no more
    assert verify(conn, namespaced("acme", rid)).rows == 8


def test_a_retry_must_repeat_the_definition_hash_too(gateway, rid):
    decide(gateway, rid, "c1", tool="pinned_lookup", definition_sha256=PIN)
    assert decide(gateway, rid, "c1", tool="pinned_lookup", definition_sha256=PIN)["status"] == "allowed"
    with pytest.raises(Conflict):  # a retry that shows a drifted definition is not the same request
        decide(gateway, rid, "c1", tool="pinned_lookup", definition_sha256="b" * 64)
    with pytest.raises(Conflict):
        decide(gateway, rid, "c1", tool="pinned_lookup")


def test_known_gap__replaying_an_allowed_call_id_is_answered_again_and_costs_nothing(gateway, conn, rid):
    # Retries must be free so a runtime can re-ask after a network failure. The price: whoever picks the
    # call_id can replay an `allowed` answer without it counting as a step or as spend. So call_id must
    # come from the runtime (a counter or a uuid), never from model output. See the README.
    assert [place(gateway, rid, "c1", cost=400)["status"] for _ in range(5)] == ["allowed"] * 5
    assert len(events(conn, namespaced("acme", rid))) == 1
    assert place(gateway, rid, "c2", cost=400)["status"] == "allowed"  # still 400 under the 1000 cap after five asks


def test_approving_a_call_in_a_halted_run_is_recorded_as_blocked(gateway, rid):
    place(gateway, rid, "big", cost=900)  # waiting
    for i in range(4):
        decide(gateway, rid, f"loop{i}", args={"trade": "plumber"})  # the fourth identical call halts the run

    answered = gateway.answer("acme", rid, "big", True, "alice")

    assert (answered["status"], answered["blocked_by"]) == ("approval_denied", "run_halted")


def test_a_tenant_cannot_hold_more_than_its_share_of_the_connections(pool, rid):
    small = Gateway(make_policy(), pool, per_tenant=1, wait=0.2)
    with small._slot("acme"):
        with pytest.raises(Busy):
            decide(small, rid, "c1")
        assert decide(small, rid, "c1", tenant="globex")["status"] == "allowed"  # other tenants are unaffected
    assert decide(small, rid, "c2")["status"] == "allowed"  # and the slot is released afterwards


def test_the_run_lock_key_is_a_stable_64_bit_number_per_run():
    assert lock_key("a") == lock_key("a") and lock_key("a") != lock_key("b")
    assert -(2**63) <= lock_key("anything") < 2**63


def test_pending_can_be_paged_and_carries_the_request_hash(gateway, rid):
    for i in range(3):
        place(gateway, rid, f"c{i}", cost=900 + i)

    ours = lambda **kw: [p for p in gateway.pending("acme", **kw) if p["run_id"] == rid]  # noqa: E731
    assert [p["call_id"] for p in ours(limit=2, offset=0)] == ["c0", "c1"]
    assert [p["call_id"] for p in ours(limit=2, offset=2)] == ["c2"]
    assert all(len(p["request_sha256"]) == 64 for p in ours())


# --- tenant isolation, approvals bookkeeping ---


def test_one_tenant_cannot_see_or_answer_anothers_calls(gateway, rid):
    place(gateway, rid, "c1", cost=900)
    with pytest.raises(NotFound):
        gateway.call_status("globex", rid, "c1")
    with pytest.raises(NotFound):
        gateway.answer("globex", rid, "c1", True, "mallory")
    assert gateway.call_status("acme", rid, "c1")["status"] == "pending_approval"


def test_two_tenants_using_the_same_run_name_keep_separate_chains(gateway, conn, rid):
    decide(gateway, rid, "c1", args={"a": 1})
    decide(gateway, rid, "c1", args={"a": 2}, tenant="globex")  # the same ids, no conflict: another run

    assert len(events(conn, namespaced("acme", rid))) == len(events(conn, namespaced("globex", rid))) == 1


def test_namespaced_run_names_cannot_collide_through_odd_names():
    assert namespaced("a", "b/c") != namespaced("a/b", "c")
    assert namespaced("a", 'b", "c') != namespaced("a", "b")


def test_only_a_call_that_is_waiting_can_be_answered_and_only_once(gateway, rid):
    decide(gateway, rid, "c1")
    with pytest.raises(Conflict):
        gateway.answer("acme", rid, "c1", True, "alice")  # it was allowed, nothing to approve
    place(gateway, rid, "c2", cost=900)
    gateway.answer("acme", rid, "c2", True, "alice")
    with pytest.raises(Conflict):
        gateway.answer("acme", rid, "c2", False, "bob")  # already answered; the first answer stands
    with pytest.raises(NotFound):
        gateway.answer("acme", rid, "missing", True, "alice")
    assert gateway.call_status("acme", rid, "c2")["status"] == "approved"


def test_pending_lists_only_this_tenants_unanswered_asks(gateway, rid):
    place(gateway, rid, "c1", cost=900)
    place(gateway, rid, "c2", cost=950)
    decide(gateway, rid, "c3")
    gateway.answer("acme", rid, "c1", False, "alice")

    pending = [p for p in gateway.pending("acme") if p["run_id"] == rid]
    assert [(p["call_id"], p["tool"], p["rule"]) for p in pending] == [("c2", "create_work_order", "approval_required")]
    assert [p for p in gateway.pending("globex") if p["run_id"] == rid] == []


def test_the_audit_view_lists_events_and_verifies_the_chain(gateway, rid):
    decide(gateway, rid, "c1")
    place(gateway, rid, "c2", cost=900)
    gateway.answer("acme", rid, "c2", True, "alice")

    view = gateway.audit("acme", rid)

    assert [e["event"] for e in view["events"]] == ["decision", "decision", "approval"]
    assert view["events"][2]["payload"]["approved_by"] == "alice"
    assert view["verified"]["ok"] and view["verified"]["rows"] == 3 and len(view["verified"]["head"]) == 64


def test_a_run_whose_chain_was_tampered_with_gets_no_more_decisions(gateway, conn, rid):
    decide(gateway, rid, "c1")
    conn.execute("UPDATE audit_log SET payload = '{}' WHERE run_id = %s", (namespaced("acme", rid),))

    with pytest.raises(ChainBroken):
        decide(gateway, rid, "c2")
    assert gateway.audit("acme", rid)["verified"]["ok"] is False  # but it can still be read


# --- HTTP ---

AGENT, APPROVER, OTHER_AGENT = new_key(), new_key(), new_key()
KEYS = (
    KeyEntry(sha256=hash_key(AGENT), tenant="acme", role="agent", name="runtime-1"),
    KeyEntry(sha256=hash_key(APPROVER), tenant="acme", role="approver", name="alice"),
    KeyEntry(sha256=hash_key(OTHER_AGENT), tenant="globex", role="agent", name="runtime-2"),
)


def bearer(key):
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def client(postgres_url):
    with TestClient(create_app(make_policy(), KEYS, postgres_url)) as test_client:
        yield test_client


def post_call(client, run, call_id, spec=None, key=AGENT):
    spec = spec or {"tool": "check_vendor_availability", "args": {"trade": "plumber"}}
    return client.post(f"/v1/runs/{run}/calls", json={"call_id": call_id, **spec}, headers=bearer(key))


def test_the_health_check_needs_no_key_and_reports_the_database(client):
    assert client.get("/healthz").json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer pc_wrong"}, {"Authorization": "Basic abc"}, {"Authorization": AGENT}])
def test_a_missing_or_wrong_key_is_refused(client, rid, headers):
    response = client.post(f"/v1/runs/{rid}/calls", json={"call_id": "c1", "tool": "x", "args": {}}, headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_an_agent_key_can_ask_for_decisions_but_not_approve(client, rid):
    assert post_call(client, rid, "c1", order(900)).json()["status"] == "pending_approval"
    forged = client.post(f"/v1/runs/{rid}/calls/c1/approval", json={"approved": True}, headers=bearer(AGENT))
    assert forged.status_code == 403
    assert client.get(f"/v1/runs/{rid}/calls/c1", headers=bearer(AGENT)).json()["status"] == "pending_approval"


def test_an_approver_key_can_approve_but_not_ask_for_decisions(client, rid):
    assert post_call(client, rid, "c1", key=APPROVER).status_code == 403
    assert client.get("/v1/approvals", headers=bearer(AGENT)).status_code == 403


def test_the_whole_approval_flow_over_http(client, rid):
    assert post_call(client, rid, "c1", order(900)).json()["status"] == "pending_approval"

    waiting = client.get("/v1/approvals", headers=bearer(APPROVER)).json()["pending"]
    assert [(p["run_id"], p["call_id"]) for p in waiting if p["run_id"] == rid] == [(rid, "c1")]

    answered = client.post(f"/v1/runs/{rid}/calls/c1/approval", json={"approved": True}, headers=bearer(APPROVER))
    assert (answered.status_code, answered.json()["status"]) == (200, "approved")
    assert client.get(f"/v1/runs/{rid}/calls/c1", headers=bearer(AGENT)).json()["status"] == "approved"

    audit = client.get(f"/v1/runs/{rid}/audit", headers=bearer(APPROVER)).json()
    assert audit["verified"]["ok"] and audit["events"][-1]["payload"]["approved_by"] == "alice"  # the name comes from the key


def test_a_denied_call_is_still_http_200_with_the_verdict_in_the_body(client, rid):
    response = post_call(client, rid, "c1", {"tool": "drop_database", "args": {}})
    assert (response.status_code, response.json()["status"], response.json()["rule"]) == (200, "denied", "tool_not_allowed")


def test_the_tenant_comes_from_the_key_never_from_the_request(client, rid):
    smuggled = client.post(
        f"/v1/runs/{rid}/calls",
        json={"call_id": "c1", "tool": "check_vendor_availability", "args": {}, "tenant_id": "acme"},
        headers=bearer(OTHER_AGENT),
    )
    assert smuggled.status_code == 422
    # globex's key reaches only globex's policy: it cannot place orders, and cannot see acme's run
    assert post_call(client, rid, "c2", order(100), key=OTHER_AGENT).json()["rule"] == "tool_not_allowed"
    post_call(client, rid, "c3", order(900))
    assert client.get(f"/v1/runs/{rid}/calls/c3", headers=bearer(OTHER_AGENT)).status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"tool": "t", "args": {}},
        {"call_id": "", "tool": "t", "args": {}},
        {"call_id": "c" * 129, "tool": "t", "args": {}},
        {"call_id": "c", "tool": "t" * 129, "args": {}},
        {"call_id": "c", "tool": "t", "args": []},
        {"call_id": "c", "tool": "t", "args": {"n": 2**70}},
        {"call_id": "c", "tool": "t", "args": {"a": {"b": {}}} | {"deep": json.loads("[" * 40 + "]" * 40)}},
        {"call_id": "c", "tool": "t", "args": {}, "definition_sha256": "short"},
    ],
    ids=["no_id", "empty_id", "long_id", "long_tool", "args_list", "huge_int", "deep", "bad_hash"],
)
def test_a_malformed_call_is_rejected_before_any_decision(client, rid, body):
    assert client.post(f"/v1/runs/{rid}/calls", json=body, headers=bearer(AGENT)).status_code == 422


@pytest.mark.parametrize("approved", ["yes", 1, None, "true"])
def test_only_a_real_boolean_approves(client, rid, approved):
    post_call(client, rid, "c1", order(900))
    response = client.post(f"/v1/runs/{rid}/calls/c1/approval", json={"approved": approved}, headers=bearer(APPROVER))
    assert response.status_code == 422


def test_an_oversized_or_unsized_body_is_refused(client, rid):
    big = {"call_id": "c1", "tool": "t", "args": {"x": "a" * 300_000}}
    assert client.post(f"/v1/runs/{rid}/calls", json=big, headers=bearer(AGENT)).status_code == 413
    odd = client.post(f"/v1/runs/{rid}/calls", content=b"{}", headers={**bearer(AGENT), "Content-Length": "abc"})
    assert odd.status_code in (400, 411)  # the HTTP client or our middleware refuses it; either way, not accepted


def test_conflicts_and_missing_calls_map_to_409_and_404(client, rid):
    post_call(client, rid, "c1")
    clash = post_call(client, rid, "c1", {"tool": "check_vendor_availability", "args": {"trade": "other"}})
    assert clash.status_code == 409
    assert client.get(f"/v1/runs/{rid}/calls/nope", headers=bearer(AGENT)).status_code == 404
    again = client.post(f"/v1/runs/{rid}/calls/c1/approval", json={"approved": True}, headers=bearer(APPROVER))
    assert again.status_code == 409  # c1 was allowed, so there is nothing to approve


def test_a_broken_audit_chain_is_a_503_not_an_answer(client, conn, rid):
    post_call(client, rid, "c1")
    conn.execute("UPDATE audit_log SET payload = '{}' WHERE run_id = %s", (namespaced("acme", rid),))
    response = post_call(client, rid, "c2")
    assert response.status_code == 503 and "audit chain" in response.json()["detail"]


def test_a_database_failure_is_a_503_that_says_nothing_about_the_database(client, rid, monkeypatch):
    def boom(*args, **kwargs):
        raise psycopg.OperationalError("connection failed: password=hunter2")

    monkeypatch.setattr(Gateway, "decide_call", boom)
    response = post_call(client, rid, "c1")
    assert response.status_code == 503 and "hunter2" not in response.text


def test_a_busy_tenant_gets_429_with_a_retry_hint(client, rid, monkeypatch):
    def busy(*args, **kwargs):
        raise Busy("acme")

    monkeypatch.setattr(Gateway, "decide_call", busy)
    response = post_call(client, rid, "c1")
    assert (response.status_code, response.headers["retry-after"]) == (429, "1")


@pytest.mark.parametrize("path", ["", "/calls/c1", "/audit"])
def test_a_request_without_a_key_is_refused_before_its_body_is_read(client, rid, path):
    garbage = {"content": b"{bad", "headers": {"content-type": "application/json"}}
    assert client.post(f"/v1/runs/{rid}{path or '/calls'}", **garbage).status_code == 401
    assert client.get("/v1/approvals").status_code == 401


@pytest.mark.parametrize("bad", ["a b", "a/b", "a" + chr(0) + "b", "a" + chr(10), "\u00e9", "x;y"])
def test_a_call_id_with_odd_characters_is_rejected(client, rid, bad):
    spec = {"call_id": bad, "tool": "check_vendor_availability", "args": {}}
    assert client.post(f"/v1/runs/{rid}/calls", json=spec, headers=bearer(AGENT)).status_code == 422
    encoded = quote(bad, safe="")  # a raw control character cannot even be put in a URL
    assert client.get(f"/v1/runs/{rid}/calls/{encoded}", headers=bearer(AGENT)).status_code in (404, 422)


def test_the_pending_list_validates_its_paging(client):
    for query in ("limit=0", "limit=101", "offset=-1", "limit=x"):
        assert client.get(f"/v1/approvals?{query}", headers=bearer(APPROVER)).status_code == 422
    assert client.get("/v1/approvals?limit=1&offset=0", headers=bearer(APPROVER)).status_code == 200


def test_the_health_check_reports_an_unreachable_database(client, monkeypatch):
    def boom(self):
        raise psycopg.OperationalError("down")

    monkeypatch.setattr(Gateway, "ping", boom)
    assert client.get("/healthz").status_code == 503


def test_the_api_docs_pages_are_not_exposed(client):
    assert [client.get(path).status_code for path in ("/docs", "/redoc", "/openapi.json")] == [404, 404, 404]


def test_a_database_not_named_portcullis_is_refused_at_startup():
    with pytest.raises(ValueError, match="portcullis"):
        create_app(make_policy(), KEYS, "postgresql://u:p@127.0.0.1:5432/other")


# --- configuration from the environment ---


def test_the_app_is_built_from_environment_settings(tmp_path, postgres_url):
    policy, keys = tmp_path / "policy.json", tmp_path / "keys.json"
    policy.write_text(make_policy().model_dump_json(), encoding="utf-8")
    keys.write_text(json.dumps({"keys": [k.model_dump() for k in KEYS]}), encoding="utf-8")
    env = {"PORTCULLIS_POLICY": str(policy), "PORTCULLIS_KEYS": str(keys), "PORTCULLIS_DATABASE_URL": postgres_url}

    with TestClient(app_from_env(env)) as client:
        assert client.get("/healthz").status_code == 200


@pytest.mark.parametrize("missing", ["PORTCULLIS_POLICY", "PORTCULLIS_KEYS", "PORTCULLIS_DATABASE_URL"])
def test_a_missing_setting_is_an_error_not_a_default(missing, tmp_path):
    env = {"PORTCULLIS_POLICY": "p", "PORTCULLIS_KEYS": "k", "PORTCULLIS_DATABASE_URL": "u"} | {missing: ""}
    with pytest.raises(ValueError, match=missing):
        app_from_env(env)


def test_an_invalid_policy_file_is_an_error(tmp_path):
    bad = tmp_path / "policy.json"
    bad.write_text("{}", encoding="utf-8")
    env = {"PORTCULLIS_POLICY": str(bad), "PORTCULLIS_KEYS": "k", "PORTCULLIS_DATABASE_URL": "u"}
    with pytest.raises(ValueError, match="policy"):
        app_from_env(env)
