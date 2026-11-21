"""agentctl: check an audit chain, read it, and dry-run a policy.

  agentctl verify RUN_ID [--rows N] [--head HASH]   exit 0 intact, 1 broken or mismatch
  agentctl show RUN_ID                              one JSON object per row, then the chain status
  agentctl check POLICY --tenant T --tool NAME      exit 0 allow, 1 deny, 3 ask (no database needed)
                [--args JSON | --args @file]
  agentctl keygen --tenant T --role agent|approver --name N   make a server API key (no database needed)

verify and show take --tenant to read a run that was made through the server, and --hook for a
Claude Code session recorded by the hook (its run id is the session id).

Exit 2 means the command itself failed (bad input, no database), not that anything is wrong with the log.

verify checks that the chain is self-consistent. A chain that lost its last rows is still
self-consistent, so pass --rows or --head from a copy you kept elsewhere to catch that.

The database is reached through PORTCULLIS_DATABASE_URL only (a URL on the command line would sit in
shell history) over a read-only connection, so this tool cannot change the log it inspects.
"""

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psycopg
from pydantic import ValidationError

from portcullis.audit import GENESIS, hook_run, namespaced, payload_data, read_verified, require_project_database, verify
from portcullis.hook import HookError, load_policy
from portcullis.jsondata import check_json
from portcullis.keys import KeyEntry, hash_key, new_key
from portcullis.policy import RunState, ToolCall, Verdict, decide

EXIT_OK, EXIT_FAIL, EXIT_ERROR, EXIT_ASK = 0, 1, 2, 3
READ_ONLY = "-c default_transaction_read_only=on -c statement_timeout=10000 -c lock_timeout=5000"


class CliError(Exception):
    """The command could not run. Reported without a traceback."""


def connect(env: Mapping[str, str]) -> psycopg.Connection:
    url = env.get("PORTCULLIS_DATABASE_URL")
    if not url:
        raise CliError("set PORTCULLIS_DATABASE_URL to the audit database")
    try:
        require_project_database(url)
        return psycopg.connect(url, autocommit=True, connect_timeout=5, options=READ_ONLY)
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    except psycopg.Error as exc:  # never echo the error: it can contain the URL
        raise CliError(f"cannot connect to the database ({type(exc).__name__})") from exc


def _run(args: argparse.Namespace) -> str:
    """The run name in the log: a server run is stored under its tenant, so --tenant names it."""
    if args.hook:  # a Claude Code session recorded by the hook (its tenant defaults to claude-code)
        return hook_run(args.tenant or "claude-code", args.run_id)
    return namespaced(args.tenant, args.run_id) if args.tenant else args.run_id


def _verify(args: argparse.Namespace, env: Mapping[str, str]) -> int:
    with connect(env) as conn:
        result = verify(conn, _run(args))
    if not result.ok:
        print(f"BROKEN: {result.problem}")
        return EXIT_FAIL
    if result.rows == 0 and args.rows != 0:
        print("FAILED: no rows for this run (an empty log looks the same as a missing one; pass --rows 0 to expect that)")
        return EXIT_FAIL
    if args.rows is not None and result.rows != args.rows:
        print(f"MISMATCH: expected {args.rows} rows, found {result.rows} rows")
        return EXIT_FAIL
    if args.head is not None and result.head != args.head:
        print(f"MISMATCH: expected head {args.head}, found {result.head}")
        return EXIT_FAIL
    print(f"OK: {result.rows} rows, head {result.head}")
    return EXIT_OK


def _show(args: argparse.Namespace, env: Mapping[str, str]) -> int:
    with connect(env) as conn:
        result, rows = read_verified(conn, _run(args))  # the rows shown are the rows that were checked
    for row in rows:
        # json.dumps escapes control characters, so a logged value cannot inject terminal escapes
        print(json.dumps({**row, "payload": payload_data(row["payload"])}))
    if result.ok:
        print(f"chain: OK, {result.rows} rows, head {result.head if result.rows else GENESIS}")
        return EXIT_OK
    print(f"chain: BROKEN: {result.problem}")
    return EXIT_FAIL


def _read_args(text: str) -> dict[str, Any]:
    try:
        raw = Path(text[1:]).read_text(encoding="utf-8") if text.startswith("@") else text
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError) as exc:
        raise CliError(f"cannot read the arguments file ({type(exc).__name__})") from exc
    except (ValueError, RecursionError) as exc:  # RecursionError: absurdly nested JSON; exit 1 would mean "deny"
        raise CliError("--args is not valid JSON") from exc
    if not isinstance(value, dict):
        raise CliError("--args must be a JSON object")
    try:
        check_json(value)  # the same rule as the server: plain JSON data, 64-bit integers, limited nesting
    except ValueError as exc:
        raise CliError(f"--args is not plain JSON data ({exc})") from exc
    return value


def _check(args: argparse.Namespace, env: Mapping[str, str]) -> int:
    try:
        policy = load_policy(Path(args.policy))
    except HookError as exc:
        raise CliError(str(exc)) from exc
    call = ToolCall(tenant_id=args.tenant, tool=args.tool, args=_read_args(args.args))
    decision = decide(policy, call, RunState())  # a fresh state: budgets and loop limits are not exercised
    print(f"{decision.verdict.value} [{decision.rule}]: {decision.reason}")
    return {Verdict.ALLOW: EXIT_OK, Verdict.DENY: EXIT_FAIL, Verdict.ASK: EXIT_ASK}[decision.verdict]


def _keygen(args: argparse.Namespace, env: Mapping[str, str]) -> int:
    key = new_key()
    try:
        entry = KeyEntry(sha256=hash_key(key), tenant=args.tenant, role=args.role, name=args.name)
    except ValidationError as exc:
        raise CliError("invalid tenant, role or name (a name is letters, digits, . _ - and at most 64 characters)") from exc
    print(f"API key (shown once; keep it in a secret manager): {key}")
    print('Add this entry to the "keys" list of your key file (it holds only the hash):')
    print(entry.model_dump_json())
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentctl", description="Inspect portcullis audit logs and policies.")
    commands = parser.add_subparsers(dest="command", required=True)

    verify_cmd = commands.add_parser("verify", help="check a run's hash chain")
    verify_cmd.add_argument("run_id")
    verify_cmd.add_argument("--tenant", help="the tenant, for a run made through the server")
    verify_cmd.add_argument("--hook", action="store_true", help="the run is a Claude Code session recorded by the hook")
    verify_cmd.add_argument("--rows", type=int, help="expected number of rows, from a copy kept elsewhere")
    verify_cmd.add_argument("--head", help="expected hash of the last row, from a copy kept elsewhere")
    verify_cmd.set_defaults(run=_verify)

    show_cmd = commands.add_parser("show", help="print a run's rows, then its chain status")
    show_cmd.add_argument("run_id")
    show_cmd.add_argument("--tenant", help="the tenant, for a run made through the server")
    show_cmd.add_argument("--hook", action="store_true", help="the run is a Claude Code session recorded by the hook")
    show_cmd.set_defaults(run=_show)

    keygen_cmd = commands.add_parser("keygen", help="make an API key for the server")
    keygen_cmd.add_argument("--tenant", required=True)
    keygen_cmd.add_argument("--role", required=True, choices=["agent", "approver"])
    keygen_cmd.add_argument("--name", required=True, help="who the key belongs to; recorded as the approver")
    keygen_cmd.set_defaults(run=_keygen)

    check_cmd = commands.add_parser("check", help="ask a policy what it would do with one call")
    check_cmd.add_argument("policy", help="path to a policy JSON file")
    check_cmd.add_argument("--tenant", required=True)
    check_cmd.add_argument("--tool", required=True)
    check_cmd.add_argument("--args", default="{}", help="a JSON object, or @path to read one from a file")
    check_cmd.set_defaults(run=_check)
    return parser


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    environment = os.environ if env is None else env
    try:
        args = _parser().parse_args(argv)
        return args.run(args, environment)
    except SystemExit as exc:  # argparse exits on bad usage (2) and on --help (0)
        return int(exc.code or 0)
    except CliError as exc:
        print(f"agentctl: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except psycopg.Error as exc:
        print(f"agentctl: database error ({type(exc).__name__})", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
