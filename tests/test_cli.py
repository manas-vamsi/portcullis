"""agentctl: verify an audit chain, show it, and dry-run a policy."""

import json
import tomllib
from pathlib import Path

import psycopg
import pytest

from portcullis.audit import append, hook_run, namespaced
from portcullis.keys import authenticate, hash_key
from portcullis.cli import connect, main

EXAMPLE_POLICY = Path(__file__).parent.parent / "plugin" / "policy.example.json"


@pytest.fixture
def env(postgres_url):
    return {"PORTCULLIS_DATABASE_URL": postgres_url}


def fill(conn, run_id, count=3):
    return [append(conn, run_id, "t1", f"e{i}", "decision", {"n": i}) for i in range(count)]


# --- verify ---


def test_verify_accepts_an_intact_chain_and_matching_expectations(conn, run_id, env, capsys):
    hashes = fill(conn, run_id)
    assert main(["verify", run_id, "--rows", "3", "--head", hashes[-1]], env) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "3 rows" in out and hashes[-1] in out


def test_verify_reports_a_changed_row(conn, run_id, env, capsys):
    fill(conn, run_id)
    conn.execute("UPDATE audit_log SET payload = '{\"n\":99}' WHERE run_id = %s AND idx = 1", (run_id,))
    assert main(["verify", run_id], env) == 1
    assert "row 1 was changed" in capsys.readouterr().out


def test_verify_with_expected_head_catches_a_deleted_tail(conn, run_id, env, capsys):
    hashes = fill(conn, run_id)  # the library alone cannot see this; a remembered head can
    conn.execute("DELETE FROM audit_log WHERE run_id = %s AND idx = 2", (run_id,))
    assert main(["verify", run_id], env) == 0  # the shorter chain is still self-consistent
    assert main(["verify", run_id, "--head", hashes[-1]], env) == 1
    assert "head" in capsys.readouterr().out


def test_verify_with_expected_rows_catches_a_deleted_tail(conn, run_id, env, capsys):
    fill(conn, run_id)
    conn.execute("DELETE FROM audit_log WHERE run_id = %s AND idx = 2", (run_id,))
    assert main(["verify", run_id, "--rows", "3"], env) == 1
    assert "2 rows" in capsys.readouterr().out


def test_verify_treats_an_empty_run_as_a_failure_unless_zero_rows_is_expected(run_id, env, capsys):
    assert main(["verify", run_id], env) == 1
    assert "no rows" in capsys.readouterr().out
    assert main(["verify", run_id, "--rows", "0"], env) == 0


def test_verify_and_show_read_a_server_run_with_tenant(conn, run_id, env, capsys):
    server_run = namespaced("acme", run_id)
    try:
        append(conn, server_run, "acme", "c1:decision", "decision", {"call_id": "c1"})
        assert main(["verify", run_id, "--tenant", "acme", "--rows", "1"], env) == 0
        assert main(["verify", run_id], env) == 1  # without the tenant that run name does not exist
        assert main(["show", run_id, "--tenant", "acme"], env) == 0
        assert '"event_id": "c1:decision"' in capsys.readouterr().out
    finally:
        conn.execute("DELETE FROM audit_log WHERE run_id = %s", (server_run,))


def test_verify_and_show_read_a_claude_code_session_with_hook(conn, run_id, env, capsys):
    session_run = hook_run("claude-code", run_id)
    try:
        append(conn, session_run, "claude-code", "t1:decision", "decision", {"call_id": "t1"})
        assert main(["verify", run_id, "--hook", "--rows", "1"], env) == 0
        assert main(["verify", run_id, "--hook", "--tenant", "someone-else"], env) == 1  # another tenant's session
        assert main(["verify", run_id], env) == 1  # a plain run with that name does not exist
        assert main(["show", run_id, "--hook"], env) == 0
        assert '"event_id": "t1:decision"' in capsys.readouterr().out
    finally:
        conn.execute("DELETE FROM audit_log WHERE run_id = %s", (session_run,))


# --- show ---


def test_show_prints_one_json_object_per_row_then_the_chain_status(conn, run_id, env, capsys):
    fill(conn, run_id, 2)
    assert main(["show", run_id], env) == 0
    *rows, status = capsys.readouterr().out.splitlines()
    parsed = [json.loads(line) for line in rows]
    assert [r["idx"] for r in parsed] == [0, 1]
    assert parsed[1]["payload"] == {"n": 1} and parsed[1]["tenant_id"] == "t1" and parsed[1]["event"] == "decision"
    assert status.startswith("chain: OK")


def test_show_exits_1_when_the_chain_is_broken(conn, run_id, env, capsys):
    fill(conn, run_id, 2)
    conn.execute("UPDATE audit_log SET payload = 'not json' WHERE run_id = %s AND idx = 0", (run_id,))
    assert main(["show", run_id], env) == 1
    out = capsys.readouterr().out
    assert '"payload": "not json"' in out and "BROKEN" in out


def test_show_cannot_inject_terminal_escapes(conn, run_id, env, capsys):
    append(conn, run_id, "t1", "e0", "decision", {"tool": "\x1b[31mred\x07"})
    main(["show", run_id], env)
    assert "\x1b" not in capsys.readouterr().out  # json escapes control characters


# --- database access ---


def test_the_cli_connection_is_read_only(env):
    with connect(env) as conn, pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        conn.execute("DELETE FROM audit_log")


@pytest.mark.parametrize("command", [["verify", "r"], ["show", "r"]])
def test_commands_fail_without_a_database_url(command, capsys):
    assert main(command, {}) == 2
    assert "PORTCULLIS_DATABASE_URL" in capsys.readouterr().err


def test_a_database_not_named_portcullis_is_refused_without_printing_the_url(capsys):
    url = "postgresql://user:hunter2@127.0.0.1:5432/other"
    assert main(["verify", "r"], {"PORTCULLIS_DATABASE_URL": url}) == 2
    err = capsys.readouterr().err
    assert "portcullis" in err and "hunter2" not in err


def test_a_connection_failure_exits_2_without_printing_the_url(capsys):
    url = "postgresql://user:hunter2@127.0.0.1:1/portcullis_x?connect_timeout=1"
    assert main(["verify", "r"], {"PORTCULLIS_DATABASE_URL": url}) == 2
    assert "hunter2" not in capsys.readouterr().err


def test_a_database_error_during_a_command_exits_2_with_only_its_type(env, monkeypatch, capsys):
    def boom(conn, run_id):
        raise psycopg.errors.UndefinedTable("relation audit_log does not exist, password=hunter2")

    monkeypatch.setattr("portcullis.cli.verify", boom)
    assert main(["verify", "r"], env) == 2
    err = capsys.readouterr().err
    assert "UndefinedTable" in err and "hunter2" not in err


# --- check (policy dry run, no database) ---


def check(tool, args=None, tenant="claude-code", policy=EXAMPLE_POLICY, env=None):
    argv = ["check", str(policy), "--tenant", tenant, "--tool", tool]
    if args is not None:
        argv += ["--args", args if isinstance(args, str) else json.dumps(args)]
    return main(argv, env or {})


def test_check_allows_a_safe_call(capsys):
    assert check("Read", {"file_path": "README.md"}) == 0
    assert capsys.readouterr().out.startswith("allow")


def test_check_denies_a_secret_file_with_the_rule_name(capsys):
    assert check("Read", {"file_path": ".env"}) == 1
    out = capsys.readouterr().out
    assert out.startswith("deny") and "forbidden_argument" in out


def test_check_reports_ask_as_exit_3(capsys):
    assert check("Bash", {"command": "git push origin main"}) == 3
    assert capsys.readouterr().out.startswith("ask")


def test_check_denies_an_unknown_tenant_and_an_unlisted_tool(capsys):
    assert check("Read", {"file_path": "x"}, tenant="nobody") == 1
    assert check("Teleport") == 1


def test_check_reads_arguments_from_a_file(tmp_path):
    args = tmp_path / "args.json"
    args.write_text('{"file_path": ".env"}', encoding="utf-8")
    assert check("Read", f"@{args}") == 1


@pytest.mark.parametrize("args", ["not json", "[1]", '"s"', "@does-not-exist.json"])
def test_check_rejects_bad_arguments(args, capsys):
    assert check("Read", args) == 2
    assert "agentctl:" in capsys.readouterr().err


def test_check_rejects_a_missing_or_invalid_policy(tmp_path, capsys):
    assert check("Read", policy=tmp_path / "missing.json") == 2
    bad = tmp_path / "bad.json"
    bad.write_text('{"tenants": 5}', encoding="utf-8")
    assert check("Read", policy=bad) == 2
    assert capsys.readouterr().err.count("agentctl:") == 2


@pytest.mark.parametrize(
    "tool, args", [("Read", {"file_path": "a.py"}), ("Read", {"file_path": ".env"}), ("Bash", {"command": "git push"})]
)
def test_check_gives_the_same_verdict_as_the_hook(tool, args, capsys):
    from portcullis.hook import run

    code, out, _ = run(json.dumps({"tool_name": tool, "tool_input": args}).encode(), {"PORTCULLIS_POLICY": str(EXAMPLE_POLICY)})
    hook_verdict = "deny" if code == 2 else "ask" if out else "allow"
    check(tool, args)
    assert capsys.readouterr().out.startswith(hook_verdict)


# --- keygen ---


def test_keygen_prints_the_key_once_and_the_entry_that_stores_only_its_hash(capsys):
    assert main(["keygen", "--tenant", "acme", "--role", "approver", "--name", "alice"], {}) == 0
    out = capsys.readouterr().out
    key = next(w for w in out.split() if w.startswith("pc_"))
    entry = json.loads(next(line for line in out.splitlines() if line.startswith("{")))
    assert entry == {"sha256": hash_key(key), "tenant": "acme", "role": "approver", "name": "alice"}
    assert key not in json.dumps(entry)
    from portcullis.keys import KeyEntry

    assert authenticate([KeyEntry(**entry)], key).role == "approver"


def test_keygen_makes_a_different_key_every_time(capsys):
    outs = []
    for _ in range(2):
        main(["keygen", "--tenant", "acme", "--role", "agent", "--name", "r"], {})
        outs.append(capsys.readouterr().out)
    assert outs[0] != outs[1]


@pytest.mark.parametrize("extra", [["--role", "admin"], ["--name", "bad name"], ["--tenant", ""]])
def test_keygen_rejects_a_bad_role_name_or_tenant(extra, capsys):
    argv = {"--tenant": "acme", "--role": "agent", "--name": "ok"} | dict(zip(extra[::2], extra[1::2]))
    assert main(["keygen", *[x for kv in argv.items() for x in kv]], {}) == 2
    capsys.readouterr()


# --- usage and wiring ---


def test_no_command_or_an_unknown_command_exits_2(capsys):
    assert main([], {}) == 2
    assert main(["frobnicate"], {}) == 2
    capsys.readouterr()


def test_the_agentctl_script_is_registered():
    project = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["agentctl"] == "portcullis.cli:main"
