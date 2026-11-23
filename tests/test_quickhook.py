"""The fast hook must answer exactly as the full hook does, and only ever be faster.

The main check is differential: Hypothesis generates policies (valid ones, and valid ones broken in one
place) and tool calls, and the fast hook's output (exit code, stdout, stderr) must equal the full hook's,
byte for byte. A second check pins that a policy the fast reader accepts is one pydantic accepts.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from portcullis import hook, quickhook
from portcullis.policy import Policy

SETTINGS = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
TOOLS = ["Bash", "Read", "create_order", "notify"]
REGEXES = [r"(?i)password\s*=", r"^(?!V-[0-9]+$)", r"rm\s+-rf", r"[", r"x{2,}", ""]  # "[" does not compile

numbers = st.one_of(
    st.integers(min_value=-5, max_value=2000),
    st.floats(allow_nan=True, allow_infinity=True),
    st.just(10**400),
    st.booleans(),
)
pattern = st.fixed_dictionaries(
    {"arg": st.sampled_from(["command", "message", "vendor", "cost"]), "regex": st.sampled_from(REGEXES)},
    optional={"required": st.booleans()},
)
rule = st.fixed_dictionaries(
    {"risk": st.sampled_from(["read", "write", "irreversible"])},
    optional={
        "pinned_sha256": st.one_of(st.none(), st.just("a" * 64)),
        "approve_above": st.one_of(st.none(), st.fixed_dictionaries({"arg": st.just("cost"), "limit": numbers})),
        "deny_if": st.lists(pattern, max_size=2),
        "ask_if": st.lists(pattern, max_size=2),
    },
)
budget = st.fixed_dictionaries(
    {
        "max_steps": st.integers(min_value=1, max_value=50),
        "max_tokens": st.integers(min_value=1, max_value=50),
        "max_seconds": st.one_of(st.integers(min_value=1, max_value=600), st.floats(min_value=0.5, max_value=600)),
        "max_identical_calls": st.integers(min_value=1, max_value=5),
    },
    optional={"max_spend": st.one_of(st.none(), numbers), "max_spend_per_day": st.one_of(st.none(), numbers)},
)
policy = st.fixed_dictionaries(
    {
        "tenants": st.dictionaries(
            st.sampled_from(["claude-code", "other"]),
            st.fixed_dictionaries({"budget": budget, "tools": st.dictionaries(st.sampled_from(TOOLS), rule, max_size=4)}),
            max_size=2,
        )
    }
)
junk = st.one_of(st.none(), st.booleans(), st.integers(), st.text(max_size=3), st.lists(st.integers(), max_size=2), st.just({}))


@st.composite
def maybe_broken(draw) -> str:
    """A generated policy as JSON text, sometimes with one value replaced, one key added or one removed."""
    doc = draw(policy)
    objects = [doc]
    stack = [doc]
    while stack:  # every dict inside, to pick one to break
        node = stack.pop()
        for value in node.values() if isinstance(node, dict) else node:
            if isinstance(value, (dict, list)):
                stack.append(value)
                if isinstance(value, dict):
                    objects.append(value)
    how = draw(st.sampled_from(["none", "none", "replace", "add", "remove"]))
    target = draw(st.sampled_from(objects))
    if how == "replace" and target:
        target[draw(st.sampled_from(sorted(target)))] = draw(junk)
    elif how == "add":
        target[draw(st.sampled_from(["extra", "risk", "budget", "arg"]))] = draw(junk)
    elif how == "remove" and target:
        del target[draw(st.sampled_from(sorted(target)))]
    return json.dumps(doc)  # NaN and Infinity come out as bare words, which pydantic refuses


tool_input = st.dictionaries(
    st.sampled_from(["command", "message", "vendor", "cost"]),
    st.one_of(
        st.sampled_from(["ls", "rm -rf /", "password=x", "V-12", "acme-1", "x" * 10_001]),
        numbers,
        st.none(),
        st.lists(st.integers(), max_size=2),
    ),
    max_size=3,
)
event = st.fixed_dictionaries(
    {"hook_event_name": st.just("PreToolUse"), "tool_name": st.sampled_from([*TOOLS, "Write"]), "tool_input": tool_input},
    optional={"session_id": st.just("s")},
)


def both(policy_text: str, stdin: bytes, **env) -> tuple[tuple, tuple]:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "policy.json"
        path.write_text(policy_text, encoding="utf-8")
        environment = {"PORTCULLIS_POLICY": str(path), **env}
        return hook.run(stdin, environment), quickhook.run(stdin, environment)


@SETTINGS
@given(policy_text=maybe_broken(), body=event, tenant=st.sampled_from(["claude-code", "other", "missing"]))
def test_the_fast_hook_answers_exactly_as_the_full_hook(policy_text, body, tenant):
    full, fast = both(policy_text, json.dumps(body).encode(), PORTCULLIS_TENANT=tenant)
    assert fast == full


def same(a, b) -> bool:
    """Equal values of the same types, all the way down (so 60 and 60.0 differ, as they do in a message)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def plain_dump(value):
    """The fast reader's objects in the shape of pydantic's model_dump()."""
    if isinstance(value, dict):
        return {k: plain_dump(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(plain_dump(v) for v in value)
    if isinstance(value, SimpleNamespace):
        return {k: plain_dump(v) for k, v in vars(value).items()}
    return value


def check_reader(policy_text: str) -> None:
    """Either the fast reader refuses it, or pydantic accepts it and reads exactly the same content."""
    try:
        plain = quickhook.read_policy(policy_text)
    except Exception:
        return  # refused: the full hook takes over, nothing to compare
    validated = Policy.model_validate_json(policy_text)  # raises if the fast reader was too lenient
    assert same(plain_dump(plain), validated.model_dump())


@SETTINGS
@given(policy_text=maybe_broken())
def test_a_policy_the_fast_reader_accepts_is_one_pydantic_accepts_with_the_same_content(policy_text):
    check_reader(policy_text)


def base() -> dict:
    return {
        "tenants": {
            "claude-code": {
                "budget": {"max_steps": 10, "max_tokens": 10, "max_seconds": 60, "max_identical_calls": 3,
                           "max_spend": 100, "max_spend_per_day": 500},
                "tools": {
                    "create_order": {
                        "risk": "write",
                        "pinned_sha256": None,
                        "approve_above": {"arg": "cost", "limit": 500},
                        "deny_if": [{"arg": "vendor", "regex": "x", "required": True}],
                        "ask_if": [{"arg": "note", "regex": "y"}],
                    }
                },
            }
        }
    }


# Every place a value can go, as a path from the document root.
BUDGET = ("tenants", "claude-code", "budget")
RULE = ("tenants", "claude-code", "tools", "create_order")
PLACES = [
    *[(*BUDGET, k) for k in ("max_steps", "max_tokens", "max_seconds", "max_identical_calls", "max_spend", "max_spend_per_day")],
    *[(*RULE, k) for k in ("risk", "pinned_sha256", "approve_above", "deny_if", "ask_if")],
    (*RULE, "approve_above", "arg"), (*RULE, "approve_above", "limit"),
    (*RULE, "deny_if", 0), (*RULE, "deny_if", 0, "arg"), (*RULE, "deny_if", 0, "regex"), (*RULE, "deny_if", 0, "required"),
    ("tenants",), ("tenants", "claude-code"), (*RULE[:-1],), RULE, BUDGET,
]
VALUES = [None, True, False, 0, -1, 1, 1.5, 0.0, 10**400, "x", "", "read", "[", "a" * 64, "A" * 64, [], {}, [1], {"x": 1}]


def put(doc, path, value):
    for key in path[:-1]:
        doc = doc[key]
    doc[path[-1]] = value


@pytest.mark.parametrize("path", PLACES, ids=["/".join(map(str, p[1:])) for p in PLACES])
def test_every_field_with_every_odd_value_is_refused_or_read_as_pydantic_reads_it(path):
    for value in VALUES:
        doc = base()
        put(doc, path, value)
        check_reader(json.dumps(doc))
    for container in {path[:i] for i in range(1, len(path) + 1)}:  # an extra key at each level on the way
        doc = base()
        node = doc
        for key in container:
            node = node[key]
        if isinstance(node, dict):
            node["extra"] = 1
            check_reader(json.dumps(doc))
    for word in ("NaN", "Infinity", "1e999", "-1e999"):  # 1e999 is valid JSON that reads as infinity
        check_reader(json.dumps(base()).replace("500}", f"{word}}}"))


def test_the_fully_filled_base_policy_is_read_by_the_fast_reader():
    assert same(plain_dump(quickhook.read_policy(json.dumps(base()))), Policy.model_validate(base()).model_dump())


# --- the fast path is taken when it should be, and only then ---

PLAIN = {
    "tenants": {
        "claude-code": {
            "budget": {"max_steps": 10, "max_tokens": 10, "max_seconds": 60, "max_identical_calls": 3},
            "tools": {"Bash": {"risk": "write", "deny_if": [{"arg": "command", "regex": r"rm\s+-rf"}]}},
        }
    }
}


@pytest.fixture
def full_hook_calls(monkeypatch):
    seen = []
    real = hook.run
    monkeypatch.setattr(hook, "run", lambda stdin, env: seen.append(1) or real(stdin, env))
    return seen


def call_fast(tmp_path, policy_text, stdin, **env):
    path = tmp_path / "policy.json"
    path.write_text(policy_text, encoding="utf-8")
    return quickhook.run(stdin, {"PORTCULLIS_POLICY": str(path), **env})


def bash(command: str) -> bytes:
    return json.dumps({"tool_name": "Bash", "tool_input": {"command": command}}).encode()


def test_a_plain_policy_is_decided_without_the_full_hook(tmp_path, full_hook_calls):
    assert call_fast(tmp_path, json.dumps(PLAIN), bash("ls")) == (0, "", "")
    code, _, err = call_fast(tmp_path, json.dumps(PLAIN), bash("rm -rf /"))
    assert code == 2 and "forbidden_argument" in err
    assert full_hook_calls == []


@pytest.mark.parametrize(
    "policy_text, stdin, env",
    [
        (json.dumps(PLAIN), bash("ls"), {"PORTCULLIS_DATABASE_URL": "postgresql://u:p@127.0.0.1:1/nope"}),
        ('{"tenants": {}, "tenants": {}}', bash("ls"), {}),  # a duplicate key
        (json.dumps(PLAIN).replace('"write"', '"WRITE"'), bash("ls"), {}),  # a value pydantic refuses
        (json.dumps(PLAIN), b"not json", {}),
        (json.dumps(PLAIN), b'{"tool_input": {}}', {}),  # no tool_name
        (json.dumps(PLAIN), b'{"tool_name": "Bash", "tool_input": [1]}', {}),  # tool_input not an object
        (json.dumps(PLAIN), b'{"tool_name": "Bash", "tool_input": {"command": "\\ud800"}}', {}),  # a lone surrogate
    ],
    ids=["database", "duplicate_key", "bad_value", "bad_event", "no_tool_name", "input_not_object", "surrogate"],
)
def test_anything_unusual_goes_to_the_full_hook_and_gets_its_answer(tmp_path, full_hook_calls, policy_text, stdin, env):
    fast = call_fast(tmp_path, policy_text, stdin, **env)
    assert full_hook_calls == [1]
    assert fast == hook.run(stdin, {"PORTCULLIS_POLICY": str(tmp_path / "policy.json"), **env})  # the full hook's answer


def test_a_missing_policy_file_is_refused_by_the_full_hook(tmp_path, full_hook_calls):
    code, _, err = quickhook.run(bash("ls"), {"PORTCULLIS_POLICY": str(tmp_path / "nope.json")})
    assert code == 2 and "cannot read policy" in err and full_hook_calls == [1]


def test_the_default_policy_path_is_the_full_hooks(tmp_path, monkeypatch, full_hook_calls):
    (tmp_path / ".portcullis").mkdir()
    (tmp_path / ".portcullis" / "policy.json").write_text(json.dumps(PLAIN), encoding="utf-8")
    assert quickhook.run(bash("ls"), {"CLAUDE_PROJECT_DIR": str(tmp_path)}) == (0, "", "")
    monkeypatch.chdir(tmp_path)
    assert quickhook.run(bash("ls"), {}) == (0, "", "")
    assert full_hook_calls == []


def test_the_fast_hook_never_loads_pydantic_on_the_fast_path(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(PLAIN), encoding="utf-8")
    probe = "import sys, portcullis.quickhook as q; r = q.run(sys.stdin.buffer.read(), {'PORTCULLIS_POLICY': sys.argv[1]}); print(r[0], 'pydantic' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", probe, str(path)], input=bash("ls"), capture_output=True, text=False, timeout=60)
    assert done.stdout.decode().split() == ["0", "False"]


@pytest.mark.parametrize(
    "args",
    [
        {"vendor": "x", "cost": 50},  # allowed
        {"vendor": "x", "cost": 200},  # over the run's spend cap
        {"vendor": "x", "cost": 600},  # over the cap and the approval limit
        {"vendor": "x", "cost": 50, "note": "y"},  # sensitive: ask
        {"cost": 50},  # missing a required argument
        {"vendor": "x", "cost": "50"},  # not an amount
    ],
)
@pytest.mark.parametrize("spend_cap", [100, 100.5, None])
def test_each_kind_of_answer_and_its_message_is_the_full_hooks(args, spend_cap):
    doc = base()
    doc["tenants"]["claude-code"]["budget"]["max_spend"] = spend_cap
    stdin = json.dumps({"tool_name": "create_order", "tool_input": args}).encode()
    full, fast = both(json.dumps(doc), stdin)
    assert fast == full


def test_the_example_policy_the_plugin_ships_takes_the_fast_path():
    text = (Path(__file__).parent.parent / "plugin" / "policy.example.json").read_text(encoding="utf-8")
    assert "claude-code" in quickhook.read_policy(text).tenants  # refused would mean users never get the speed-up
    check_reader(text)
