import math
import re

import pytest
from pydantic import ValidationError

from portcullis.policy import (
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
    definition_hash,
    fingerprint,
    spend_amount,
)

PIN = "a" * 64
BUDGET = Budget(max_steps=10, max_tokens=1000, max_seconds=60, max_identical_calls=3)

POLICY = Policy(
    tenants={
        "acme": TenantPolicy(
            budget=BUDGET,
            tools={
                "check_vendor": ToolRule(risk=Risk.READ),
                "create_work_order": ToolRule(
                    risk=Risk.WRITE, approve_above=Threshold(arg="cost", limit=500)
                ),
                "delete_property": ToolRule(risk=Risk.IRREVERSIBLE),
                "pinned_tool": ToolRule(risk=Risk.READ, pinned_sha256=PIN),
                "pinned_delete": ToolRule(risk=Risk.IRREVERSIBLE, pinned_sha256=PIN),
                "wire_funds": ToolRule(
                    risk=Risk.IRREVERSIBLE, approve_above=Threshold(arg="cost", limit=500)
                ),
            },
        ),
        "globex": TenantPolicy(budget=BUDGET, tools={"check_vendor": ToolRule(risk=Risk.READ)}),
    }
)


def call(tool="check_vendor", tenant="acme", **kwargs) -> ToolCall:
    return ToolCall(tenant_id=tenant, tool=tool, **kwargs)


def check(tool_call: ToolCall, state: RunState | None = None):
    return decide(POLICY, tool_call, state or RunState())


def test_allowed_read_tool_is_allowed():
    decision = check(call())

    assert decision.verdict is Verdict.ALLOW
    assert decision.rule == "allowed"
    assert decision.reason


def test_unknown_tenant_is_denied():
    decision = check(call(tenant="nobody"))

    assert (decision.verdict, decision.rule) == (Verdict.DENY, "unknown_tenant")


def test_tool_missing_from_the_allowlist_is_denied():
    decision = check(call(tool="drop_database"))

    assert (decision.verdict, decision.rule) == (Verdict.DENY, "tool_not_allowed")


def test_allowlists_are_per_tenant():
    allowed = check(call("create_work_order", tenant="acme", args={"cost": 10}))
    refused = check(call("create_work_order", tenant="globex", args={"cost": 10}))

    assert allowed.verdict is Verdict.ALLOW
    assert (refused.verdict, refused.rule) == (Verdict.DENY, "tool_not_allowed")


def test_irreversible_tool_always_asks_a_human():
    decision = check(call("delete_property"))

    assert (decision.verdict, decision.rule) == (Verdict.ASK, "irreversible_action")


@pytest.mark.parametrize("cost", [0, 120, 500])
def test_write_at_or_below_the_threshold_is_allowed(cost):
    assert check(call("create_work_order", args={"cost": cost})).verdict is Verdict.ALLOW


@pytest.mark.parametrize("cost", [500.01, 900])
def test_write_above_the_threshold_asks_a_human(cost):
    decision = check(call("create_work_order", args={"cost": cost}))

    assert (decision.verdict, decision.rule) == (Verdict.ASK, "approval_required")
    assert f"'cost' {cost} is above" in decision.reason  # the audit reader sees the argument and values


@pytest.mark.parametrize(
    "args",
    [{}, {"cost": "600"}, {"cost": True}, {"cost": None}, {"cost": math.nan}, {"cost": math.inf}, {"cost": -5}],
    ids=["missing", "string", "bool", "none", "nan", "inf", "negative"],
)
def test_unusable_spend_amount_is_denied_not_allowed(args):
    decision = check(call("create_work_order", args=args))

    assert (decision.verdict, decision.rule) == (Verdict.DENY, "invalid_amount")


def test_invalid_amount_is_denied_even_on_an_irreversible_tool():
    bad = check(call("wire_funds", args={"cost": math.nan}))
    fine = check(call("wire_funds", args={"cost": 10}))

    assert (bad.verdict, bad.rule) == (Verdict.DENY, "invalid_amount")
    assert (fine.verdict, fine.rule) == (Verdict.ASK, "irreversible_action")


def test_huge_integer_amount_asks_a_human_instead_of_crashing():
    decision = check(call("create_work_order", args={"cost": 10**400}))

    assert (decision.verdict, decision.rule) == (Verdict.ASK, "approval_required")


def test_pinned_tool_with_matching_definition_is_allowed():
    assert check(call("pinned_tool", definition_sha256=PIN)).verdict is Verdict.ALLOW


@pytest.mark.parametrize("seen", ["b" * 64, None], ids=["changed", "not_provided"])
def test_pinned_tool_whose_definition_changed_or_is_unknown_is_denied(seen):
    decision = check(call("pinned_tool", definition_sha256=seen))

    assert (decision.verdict, decision.rule) == (Verdict.DENY, "definition_drift")


def test_hash_is_ignored_for_tools_without_a_pin():
    assert check(call(definition_sha256="c" * 64)).verdict is Verdict.ALLOW


@pytest.mark.parametrize(
    "state",
    [RunState(steps=10), RunState(tokens=1000), RunState(seconds=60)],
    ids=["steps", "tokens", "seconds"],
)
def test_used_up_budget_is_denied(state):
    decision = check(call(), state)

    assert (decision.verdict, decision.rule) == (Verdict.DENY, "budget_exceeded")
    assert "budget used up" in decision.reason


def test_budget_not_yet_used_up_is_allowed():
    state = RunState(steps=9, tokens=999, seconds=59.9)

    assert check(call(), state).verdict is Verdict.ALLOW


def test_repeating_the_same_call_too_often_is_a_loop():
    same = fingerprint("check_vendor", {"unit": 4})
    other = fingerprint("check_vendor", {"unit": 5})

    two = check(call(args={"unit": 4}), RunState(history=(same, same, other)))
    three = check(call(args={"unit": 4}), RunState(history=(same, same, same)))

    assert two.verdict is Verdict.ALLOW
    assert (three.verdict, three.rule) == (Verdict.DENY, "loop_detected")


@pytest.mark.parametrize(
    ("tool", "definition", "state", "rule"),
    [
        ("pinned_delete", "b" * 64, RunState(), "definition_drift"),
        ("delete_property", None, RunState(steps=10), "budget_exceeded"),
        ("delete_property", None, RunState(history=(fingerprint("delete_property", {}),) * 3), "loop_detected"),
    ],
    ids=["pin_drift", "budget", "loop"],
)
def test_deny_beats_ask(tool, definition, state, rule):
    # Every call here would otherwise be ASK because the tool is irreversible.
    decision = check(call(tool, definition_sha256=definition), state)

    assert (decision.verdict, decision.rule) == (Verdict.DENY, rule)


@pytest.mark.parametrize(
    "tool_call",
    [
        call(args={"x": {1: "a", "b": 2}}),  # sorting mixed-type keys raises TypeError
        call("create_work_order", args={"cost": 10**5000}),  # str() of a huge int raises ValueError
    ],
    ids=["mixed_key_dict", "huge_int"],
)
def test_a_crash_while_deciding_is_a_deny_not_an_exception(tool_call):
    decision = check(tool_call)

    assert (decision.verdict, decision.rule) == (Verdict.DENY, "evaluation_error")


def test_checks_stop_at_the_first_deny_so_later_checks_cannot_crash_it():
    crashing_args = {"x": {1: "a", "b": 2}}  # would crash the loop check's fingerprint

    decision = check(call(args=crashing_args), RunState(steps=10))

    assert (decision.verdict, decision.rule) == (Verdict.DENY, "budget_exceeded")


def test_an_exhausted_budget_denies_even_a_tool_that_is_not_allowlisted():
    # Otherwise a model that keeps calling forbidden tools would never be stopped.
    decision = check(call(tool="drop_database"), RunState(steps=10))

    assert (decision.verdict, decision.rule) == (Verdict.DENY, "budget_exceeded")


def test_repeating_a_call_to_a_tool_that_is_not_allowlisted_is_a_loop():
    same = fingerprint("drop_database", {})

    decision = check(call(tool="drop_database"), RunState(history=(same,) * 3))

    assert (decision.verdict, decision.rule) == (Verdict.DENY, "loop_detected")


def test_names_are_quoted_in_reasons_so_they_cannot_forge_log_lines():
    decision = check(call(tool="x\nrule=allow"))

    assert decision.rule == "tool_not_allowed"
    assert "\n" not in decision.reason


@pytest.mark.parametrize("bad", [{"steps": -1}, {"tokens": -1}, {"seconds": -1}, {"seconds": math.nan}])
def test_run_state_rejects_negative_or_nan_counters(bad):
    with pytest.raises(ValidationError):
        RunState(**bad)


def test_fingerprint_ignores_key_order_but_not_tool_or_values():
    assert fingerprint("t", {"a": 1, "b": 2}) == fingerprint("t", {"b": 2, "a": 1})
    assert fingerprint("t", {"a": 1}) != fingerprint("u", {"a": 1})
    assert fingerprint("t", {"a": 1}) != fingerprint("t", {"a": 2})


def test_policy_loads_from_json_and_rejects_unknown_fields():
    raw = {"tenants": {"acme": {"budget": BUDGET.model_dump(), "tools": {"x": {"risk": "read"}}}}}

    policy = Policy.model_validate(raw)
    assert policy.tenants["acme"].tools["x"].risk is Risk.READ

    raw["tenants"]["acme"]["tools"]["x"]["typo_field"] = 1
    with pytest.raises(ValidationError):
        Policy.model_validate(raw)


@pytest.mark.parametrize("field", ["max_steps", "max_tokens", "max_seconds", "max_identical_calls"])
def test_budget_limits_must_be_positive(field):
    with pytest.raises(ValidationError):
        Budget(**{**BUDGET.model_dump(), field: 0})


@pytest.mark.parametrize(
    "build",
    [
        lambda: TenantPolicy(tools={}),  # budget is required, so it cannot be forgotten
        lambda: ToolRule(risk=Risk.READ, pinned_sha256="not-a-sha256"),
        lambda: Threshold(arg="cost", limit=-1),
        lambda: Threshold(arg="cost", limit=math.nan),
        lambda: Budget(**{**BUDGET.model_dump(), "max_seconds": math.nan}),
    ],
    ids=["no_budget", "bad_pin", "negative_limit", "nan_limit", "nan_seconds"],
)
def test_bad_policy_config_is_rejected(build):
    with pytest.raises(ValidationError):
        build()



SHELL_POLICY = Policy(
    tenants={
        "dev": TenantPolicy(
            budget=BUDGET,
            tools={
                "Bash": ToolRule(
                    risk=Risk.WRITE,
                    deny_if=(ArgPattern(arg="command", regex=r"(?i)\brm\s+-rf\s+/"),),
                    ask_if=(ArgPattern(arg="command", regex=r"(?i)git\s+push"),),
                ),
            },
        )
    }
)


def shell(command, state=None):
    call = ToolCall(tenant_id="dev", tool="Bash", args={"command": command})
    return decide(SHELL_POLICY, call, state or RunState())


def test_an_argument_that_matches_a_forbidden_pattern_is_denied_without_echoing_the_pattern():
    decision = shell("sudo RM -RF /home")

    assert (decision.verdict, decision.rule) == (Verdict.DENY, "forbidden_argument")
    assert "rm" not in decision.reason.lower() and "command" in decision.reason


def test_an_argument_that_matches_a_sensitive_pattern_asks_a_human():
    decision = shell("git push origin main")

    assert (decision.verdict, decision.rule) == (Verdict.ASK, "sensitive_argument")


def test_a_forbidden_match_beats_a_sensitive_one():
    assert shell("git push && rm -rf /").verdict is Verdict.DENY


def test_a_command_matching_nothing_is_allowed():
    assert shell("ls -la").verdict is Verdict.ALLOW


def test_a_missing_inspected_argument_matches_nothing():
    call = ToolCall(tenant_id="dev", tool="Bash", args={})

    assert decide(SHELL_POLICY, call, RunState()).verdict is Verdict.ALLOW


def test_an_inspected_argument_that_is_not_a_string_is_still_checked():
    call = ToolCall(tenant_id="dev", tool="Bash", args={"command": ["rm", "-rf", "/"]})

    assert decide(SHELL_POLICY, call, RunState()).verdict is Verdict.ALLOW  # str() of a list does not match
    call = ToolCall(tenant_id="dev", tool="Bash", args={"command": {"cmd": "rm -rf /x"}})
    assert decide(SHELL_POLICY, call, RunState()).verdict is Verdict.DENY


def test_a_very_long_inspected_argument_is_denied_so_padding_cannot_hide_a_command():
    padded = "a" * 20_000 + "; rm -rf /"

    decision = shell(padded)

    assert (decision.verdict, decision.rule) == (Verdict.DENY, "argument_too_long")


def test_the_length_limit_applies_to_arguments_inspected_by_ask_rules_too():
    policy = Policy(
        tenants={
            "dev": TenantPolicy(
                budget=BUDGET,
                tools={"Write": ToolRule(risk=Risk.WRITE, ask_if=(ArgPattern(arg="file_path", regex=r"\.env"),))},
            )
        }
    )
    call = ToolCall(tenant_id="dev", tool="Write", args={"file_path": "x" * 20_000})

    assert decide(policy, call, RunState()).rule == "argument_too_long"


def test_argument_rules_come_after_the_budget_check():
    assert shell("ls", RunState(steps=10)).rule == "budget_exceeded"


def test_an_invalid_regex_is_rejected_when_the_policy_loads():
    with pytest.raises(ValidationError):
        ArgPattern(arg="command", regex="(unclosed")


def test_argument_rules_load_from_json():
    raw = {
        "tenants": {
            "dev": {
                "budget": BUDGET.model_dump(),
                "tools": {"Bash": {"risk": "write", "deny_if": [{"arg": "command", "regex": "rm -rf"}]}},
            }
        }
    }

    policy = Policy.model_validate(raw)

    assert policy.tenants["dev"].tools["Bash"].deny_if[0].regex == "rm -rf"


def test_a_required_argument_that_is_missing_is_denied_not_ignored():
    # A typo in the argument name (file_path vs notebook_path) must not silently switch a rule off.
    policy = Policy(
        tenants={
            "dev": TenantPolicy(
                budget=BUDGET,
                tools={
                    "Edit": ToolRule(
                        risk=Risk.WRITE, deny_if=(ArgPattern(arg="file_path", regex=r"\.env", required=True),)
                    )
                },
            )
        }
    )

    missing = decide(policy, ToolCall(tenant_id="dev", tool="Edit", args={"path": "x"}), RunState())
    present = decide(policy, ToolCall(tenant_id="dev", tool="Edit", args={"file_path": "a.py"}), RunState())

    assert (missing.verdict, missing.rule) == (Verdict.DENY, "missing_argument")
    assert present.verdict is Verdict.ALLOW


def test_a_definition_hash_ignores_key_order_and_changes_with_any_field():
    base = {"name": "t", "description": "d", "schema": {"a": "string", "b": "int"}}
    same = {"schema": {"b": "int", "a": "string"}, "description": "d", "name": "t"}
    assert definition_hash(base) == definition_hash(same)
    assert re.fullmatch(r"[0-9a-f]{64}", definition_hash(base))
    for field, value in (("description", "d "), ("name", "u"), ("schema", {"a": "int", "b": "int"})):
        assert definition_hash(base | {field: value}) != definition_hash(base)


# --- cumulative spend cap ---


def capped(max_spend) -> Policy:
    budget = BUDGET.model_copy(update={"max_spend": max_spend})
    tools = {
        "create_work_order": ToolRule(risk=Risk.WRITE, approve_above=Threshold(arg="cost", limit=500)),
        "check_vendor": ToolRule(risk=Risk.READ),
    }
    return Policy(tenants={"acme": TenantPolicy(budget=budget, tools=tools)})


def order_decision(cost, spent, cap=1000, tool="create_work_order"):
    return decide(capped(cap), call(tool, args={"cost": cost}), RunState(spend=spent))


def test_without_a_cap_spend_is_not_limited():
    assert BUDGET.max_spend is None
    assert check(call("create_work_order", args={"cost": 499}), RunState(spend=10**9)).verdict is Verdict.ALLOW


def test_a_call_that_stays_within_the_cap_is_allowed_and_one_that_reaches_it_exactly_too():
    assert order_decision(300, spent=600).verdict is Verdict.ALLOW
    assert order_decision(400, spent=600).verdict is Verdict.ALLOW  # exactly at the cap


def test_a_call_that_would_pass_the_cap_is_denied_with_its_own_rule():
    decision = order_decision(401, spent=600)
    assert (decision.verdict, decision.rule) == (Verdict.DENY, "spend_cap_exceeded")


def test_the_cap_beats_the_approval_threshold_so_a_human_cannot_exceed_it():
    # 900 is above the 500 approval line, which would normally ask; over the cap it is denied instead.
    assert order_decision(900, spent=500).rule == "spend_cap_exceeded"
    assert order_decision(900, spent=0).rule == "approval_required"  # within the cap it still asks


def test_a_cap_of_zero_allows_no_spend_but_still_allows_free_calls():
    assert order_decision(1, spent=0, cap=0).rule == "spend_cap_exceeded"
    assert order_decision(0, spent=0, cap=0).verdict is Verdict.ALLOW


def test_the_cap_only_counts_tools_that_declare_an_amount():
    assert order_decision(0, spent=999_999, tool="check_vendor").verdict is Verdict.ALLOW


def test_an_invalid_amount_is_still_denied_as_invalid_not_as_over_the_cap():
    assert decide(capped(1000), call("create_work_order", args={"cost": "900"}), RunState()).rule == "invalid_amount"


@pytest.mark.parametrize("cap", [-1, math.nan, math.inf])
def test_a_cap_must_be_a_finite_non_negative_number(cap):
    with pytest.raises(ValidationError):
        Budget(**{**BUDGET.model_dump(), "max_spend": cap})


@pytest.mark.parametrize("spend", [-1, math.nan, math.inf])
def test_spend_in_the_run_state_must_be_finite_and_non_negative(spend):
    with pytest.raises(ValidationError):
        RunState(spend=spend)


def test_spend_amount_is_what_a_call_would_spend_and_zero_when_there_is_nothing_to_count():
    pol = capped(1000)
    assert spend_amount(pol, call("create_work_order", args={"cost": 250})) == 250
    assert spend_amount(pol, call("create_work_order", args={"cost": "250"})) == 0  # invalid: never executed
    assert spend_amount(pol, call("check_vendor")) == 0
    assert spend_amount(pol, call("nope")) == 0
    assert spend_amount(pol, call("create_work_order", tenant="nobody", args={"cost": 5})) == 0
