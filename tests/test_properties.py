"""Property tests: rules that must hold for every input, checked against thousands of generated ones.

Example tests show the engine does the right thing for the cases someone thought of. These search for the
cases nobody thought of: odd tool names, deeply nested or huge arguments, extreme counters and amounts.
"""

import json
import math
from datetime import datetime, timedelta, timezone

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from portcullis.audit import GENESIS, _row_hash, check_rows
from portcullis.jsondata import MAX_DEPTH, MAX_INT, MIN_INT, check_json
from portcullis.keys import KeyEntry, authenticate, hash_key, new_key
from portcullis.policy import (
    HALTING_RULES,
    ArgPattern,
    Budget,
    Policy,
    Risk,
    RunState,
    TenantPolicy,
    Threshold,
    ToolCall,
    ToolRule,
    Verdict,
    decide,
    fingerprint,
)
from portcullis.runstate import fold, run_state

SETTINGS = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])

BUDGET = Budget(max_steps=50, max_tokens=10_000, max_seconds=600, max_identical_calls=5, max_spend=1000, max_spend_per_day=5000)
POLICY = Policy(
    tenants={
        "acme": TenantPolicy(
            budget=BUDGET,
            tools={
                "read": ToolRule(risk=Risk.READ),
                "pay": ToolRule(
                    risk=Risk.WRITE,
                    approve_above=Threshold(arg="amount", limit=500),
                    deny_if=(ArgPattern(arg="to", regex=r"^(?!acme-)", required=True),),
                    ask_if=(ArgPattern(arg="memo", regex=r"(?i)password"),),
                ),
                "wipe": ToolRule(risk=Risk.IRREVERSIBLE, deny_if=(ArgPattern(arg="path", regex=r"^/$"),)),
            },
        )
    }
)

# Plain JSON values of the kind a model can send, including awkward ones.
scalars = st.one_of(
    st.none(), st.booleans(), st.integers(min_value=MIN_INT, max_value=MAX_INT),
    st.floats(allow_nan=True, allow_infinity=True), st.text(max_size=40),
)
json_values = st.recursive(scalars, lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=8), inner, max_size=4), max_leaves=20)
arguments = st.dictionaries(st.sampled_from(["amount", "to", "memo", "path", "x"]) | st.text(max_size=8), json_values, max_size=6)
tool_names = st.sampled_from(["read", "pay", "wipe", "drop_database", ""]) | st.text(max_size=30)
tenants = st.sampled_from(["acme", "globex", ""]) | st.text(max_size=12)
states = st.builds(
    RunState,
    steps=st.integers(min_value=0, max_value=10**6),
    tokens=st.integers(min_value=0, max_value=10**9),
    seconds=st.floats(min_value=0, max_value=10**7, allow_nan=False, allow_infinity=False),
    history=st.lists(st.text(alphabet="0123456789abcdef", min_size=64, max_size=64), max_size=8).map(tuple),
    spend=st.floats(min_value=0, max_value=10**7, allow_nan=False, allow_infinity=False),
    tenant_spend=st.floats(min_value=0, max_value=10**7, allow_nan=False, allow_infinity=False),
)


def call(tool, args, tenant="acme"):
    return ToolCall(tenant_id=tenant, tool=tool, args=args)


# --- the engine ---


@SETTINGS
@given(tool=tool_names, args=arguments, tenant=tenants, state=states)
def test_decide_never_raises_and_always_names_a_rule(tool, args, tenant, state):
    decision = decide(POLICY, call(tool, args, tenant), state)
    assert decision.verdict in Verdict and decision.rule and decision.reason


@SETTINGS
@given(tool=tool_names, args=arguments, state=states)
def test_the_same_question_always_gets_the_same_answer(tool, args, state):
    assert decide(POLICY, call(tool, args), state) == decide(POLICY, call(tool, args), state)


@SETTINGS
@given(tool=tool_names.filter(lambda t: t not in ("read", "pay", "wipe")), args=arguments, state=states)
def test_a_tool_off_the_allowlist_is_never_allowed_or_asked_about(tool, args, state):
    assert decide(POLICY, call(tool, args), state).verdict is Verdict.DENY


@SETTINGS
@given(tool=tool_names, args=arguments, tenant=tenants.filter(lambda t: t != "acme"), state=states)
def test_an_unknown_tenant_gets_nothing(tool, args, tenant, state):
    decision = decide(POLICY, call(tool, args, tenant), state)
    assert (decision.verdict, decision.rule) == (Verdict.DENY, "unknown_tenant")


@SETTINGS
@given(tool=tool_names, args=arguments, state=states, extra=st.integers(min_value=0, max_value=1000))
def test_once_the_step_budget_is_spent_nothing_more_is_allowed(tool, args, state, extra):
    spent = state.model_copy(update={"steps": BUDGET.max_steps + extra})
    decision = decide(POLICY, call(tool, args), spent)
    assert decision.verdict is Verdict.DENY and decision.rule in HALTING_RULES


@SETTINGS
@given(amount=st.integers(min_value=0, max_value=10**6) | st.floats(min_value=0, max_value=10**6, allow_nan=False), state=states)
def test_a_payment_that_would_pass_either_spend_cap_is_never_allowed_or_asked_about(amount, state):
    calm = state.model_copy(update={"steps": 0, "seconds": 0.0, "tokens": 0, "history": ()})
    decision = decide(POLICY, call("pay", {"amount": amount, "to": "acme-1"}), calm)
    if calm.spend + amount > BUDGET.max_spend or calm.tenant_spend + amount > BUDGET.max_spend_per_day:
        assert decision.verdict is Verdict.DENY
    else:
        assert decision.verdict in (Verdict.ALLOW, Verdict.ASK)


@SETTINGS
@given(amount=st.one_of(st.text(), st.booleans(), st.none(), st.just(float("nan")), st.just(float("inf")),
                        st.integers(max_value=-1), st.floats(max_value=-1e-9, allow_nan=False)))
def test_an_amount_that_is_not_a_real_non_negative_number_is_always_refused(amount):
    decision = decide(POLICY, call("pay", {"amount": amount, "to": "acme-1"}), RunState())
    assert (decision.verdict, decision.rule) == (Verdict.DENY, "invalid_amount")


@SETTINGS
@given(to=st.text(max_size=40), memo=st.text(max_size=40))
def test_deny_always_beats_ask(to, memo):
    decision = decide(POLICY, call("pay", {"amount": 900, "to": to, "memo": memo}), RunState())
    if not to.startswith("acme-"):
        assert (decision.verdict, decision.rule) == (Verdict.DENY, "forbidden_argument")  # even though 900 would ask


@SETTINGS
@given(args=arguments)
def test_an_irreversible_tool_is_never_simply_allowed(args):
    assert decide(POLICY, call("wipe", args), RunState()).verdict is not Verdict.ALLOW


@SETTINGS
@given(args=arguments, state=states)
def test_a_call_repeated_past_the_limit_is_stopped(args, state):
    seen = fingerprint("read", args)
    looping = state.model_copy(update={"steps": 0, "seconds": 0.0, "tokens": 0, "history": (seen,) * BUDGET.max_identical_calls})
    assert decide(POLICY, call("read", args), looping).verdict is Verdict.DENY


@SETTINGS
@given(args=st.dictionaries(st.text(max_size=8), json_values, max_size=6))
def test_the_fingerprint_ignores_key_order(args):
    reordered = dict(reversed(list(args.items())))
    assert fingerprint("t", args) == fingerprint("t", reordered)


# --- the argument check ---


@SETTINGS
@given(value=json_values)
def test_values_that_pass_the_json_check_can_always_be_written_as_json(value):
    try:
        check_json(value)
    except ValueError:
        return
    json.dumps(value)  # anything accepted is storable


@SETTINGS
@given(depth=st.integers(min_value=MAX_DEPTH + 1, max_value=MAX_DEPTH + 200))
def test_nesting_past_the_limit_is_always_refused(depth):
    value: list = []
    for _ in range(depth):
        value = [value]
    try:
        check_json(value)
    except ValueError:
        return
    raise AssertionError("deep nesting was accepted")


@SETTINGS
@given(number=st.integers(min_value=MAX_INT + 1) | st.integers(max_value=MIN_INT - 1))
def test_integers_outside_64_bits_are_always_refused(number):
    try:
        check_json({"n": number})
    except ValueError:
        return
    raise AssertionError("an out-of-range integer was accepted")


# --- the audit chain ---


def build_chain(run, payloads):
    rows, prev = [], GENESIS
    for idx, payload in enumerate(payloads):
        ts = f"2026-01-01T00:00:{idx:02d}+00:00"
        h = _row_hash(prev, idx, run, "t", f"e{idx}", "decision", ts, payload)
        rows.append((idx, f"e{idx}", "t", "decision", ts, payload, prev, h))
        prev = h
    return rows


@SETTINGS
@given(payloads=st.lists(st.text(max_size=30), min_size=1, max_size=12), data=st.data())
def test_changing_any_field_of_any_row_is_always_detected(payloads, data):
    rows = build_chain("run", payloads)
    assert check_rows("run", rows).ok
    row = data.draw(st.integers(min_value=0, max_value=len(rows) - 1))
    field = data.draw(st.integers(min_value=1, max_value=5))  # event_id, tenant_id, event, ts, payload
    new_value = data.draw(st.text(max_size=30).filter(lambda v: v != rows[row][field]))
    tampered = list(rows)
    tampered[row] = rows[row][:field] + (new_value,) + rows[row][field + 1 :]
    assert not check_rows("run", tampered).ok


@SETTINGS
@given(payloads=st.lists(st.text(max_size=20), min_size=2, max_size=12), data=st.data())
def test_removing_or_swapping_rows_in_the_middle_is_always_detected(payloads, data):
    rows = build_chain("run", payloads)
    i = data.draw(st.integers(min_value=0, max_value=len(rows) - 2))
    assert not check_rows("run", rows[:i] + rows[i + 1 :]).ok  # a row removed (not the last)
    swapped = list(rows)
    swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
    assert not check_rows("run", swapped).ok


@SETTINGS
@given(payloads=st.lists(st.text(max_size=20), min_size=1, max_size=8))
def test_a_chain_cannot_be_moved_to_another_run_name(payloads):
    assert not check_rows("other-run", build_chain("run", payloads)).ok


# --- run state ---

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def rows_from(events):
    """events: (kind, call index, seconds) -> audit-shaped rows."""
    out = []
    for kind, i, at in events:
        when = (T0 + timedelta(seconds=at)).isoformat()
        if kind == "decision":
            payload = {"call_id": f"c{i}", "tool": "t", "verdict": "ask" if i % 2 else "allow", "rule": "r", "reason": "x",
                       "request_sha256": f"h{i}", "amount": i, "tokens": i}
        else:
            payload = {"call_id": f"c{i}", "approved": bool(i % 3)}
        out.append({"event": kind, "payload": json.dumps(payload), "ts": when})
    return out


@SETTINGS
@given(
    starts=st.lists(st.floats(min_value=0, max_value=1000, allow_nan=False), min_size=1, max_size=10),
    answers=st.lists(st.floats(min_value=0, max_value=2000, allow_nan=False), max_size=10),
    now=st.floats(min_value=0, max_value=5000, allow_nan=False),
)
def test_elapsed_time_is_never_negative_and_never_more_than_the_wall_clock(starts, answers, now):
    events = [("decision", i, at) for i, at in enumerate(sorted(starts))]
    events += [("approval", i, at) for i, at in enumerate(answers) if i < len(starts) and i % 2]
    calls = fold(rows_from(events))
    seconds = run_state(calls, T0 + timedelta(seconds=now)).seconds
    wall = max(0.0, now - min(starts))
    assert 0 <= seconds <= wall + 1e-6
    assert not math.isnan(seconds)


@SETTINGS
@given(count=st.integers(min_value=0, max_value=20))
def test_steps_spend_and_tokens_are_never_negative(count):
    state = run_state(fold(rows_from([("decision", i, i) for i in range(count)])), T0 + timedelta(seconds=100))
    assert state.steps == count and state.spend >= 0 and state.tokens >= 0


# --- API keys ---


@settings(max_examples=60, deadline=None)
@given(other=st.text(min_size=0, max_size=80))
def test_only_the_exact_key_authenticates(other):
    key = new_key()
    assume(other != key)
    entries = [KeyEntry(sha256=hash_key(key), tenant="acme", role="agent", name="a")]
    assert authenticate(entries, other) is None and authenticate(entries, key) is not None
