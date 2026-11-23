"""Measure what portcullis costs, on this machine. No model is involved.

    uv run --env-file .env python benchmarks/run.py [--quick] [--only NAME ...] [--out FILE]

Benchmarks (each is a function below; all numbers are wall-clock times on whatever machine runs this):
  policy   one policy decision, in process, no I/O
  audit    appending to and verifying a hash chain in Postgres
  server   a decision over real HTTP on loopback, and how it changes as the run's log grows
  hook     the Claude Code hook as a fresh process per call: the full hook with and without a database,
           and the fast hook (portcullis.quickhook), which needs no database
  mcp      what the MCP proxy adds to one tool call (in memory, no process boundary)

What the numbers are not: a comparison, a load test, or a claim about any other machine, database or
network. Postgres is the local Docker container. The server and its client share the machine and its
CPU, so they compete. Read them as "this is the order of magnitude here" and rerun on your own hardware.

Uses the project database from PORTCULLIS_DATABASE_URL (see .env.example) and removes every row it
writes (tenant "bench"). It refuses a database whose name does not start with "portcullis".
"""

import argparse
import asyncio
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import psycopg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from portcullis.audit import append, create_audit_log, require_project_database, verify  # noqa: E402
from portcullis.keys import KeyEntry, hash_key, new_key  # noqa: E402
from portcullis.policy import (  # noqa: E402
    ArgPattern,
    Budget,
    Policy,
    Risk,
    RunState,
    TenantPolicy,
    Threshold,
    ToolCall,
    ToolRule,
    decide,
)

TENANT = "bench"


def bench_policy_object() -> Policy:
    tools = {
        "read": ToolRule(risk=Risk.READ),
        "write": ToolRule(
            risk=Risk.WRITE,
            approve_above=Threshold(arg="cost", limit=500),
            deny_if=(ArgPattern(arg="vendor", regex=r"^(?!acme-)", required=True),),
            ask_if=(ArgPattern(arg="note", regex=r"(?i)password\s*="),),
        ),
    }
    budget = Budget(max_steps=100_000, max_tokens=10**9, max_seconds=10**6, max_identical_calls=10**6, max_spend=10**9)
    return Policy(tenants={TENANT: TenantPolicy(budget=budget, tools=tools)})


def summary(samples_seconds: Sequence[float]) -> dict[str, float]:
    """Median, 95th and 99th percentile (nearest rank) and mean, in milliseconds."""
    ordered = sorted(samples_seconds)

    def pct(fraction: float) -> float:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)] * 1000

    return {
        "n": len(ordered),
        "median_ms": statistics.median(ordered) * 1000,
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "mean_ms": statistics.fmean(ordered) * 1000,
    }


def timed(function: Callable[[], Any], repeat: int) -> list[float]:
    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return samples


# ---- policy ----


def bench_policy(quick: bool, **_: Any) -> dict[str, Any]:
    policy, state = bench_policy_object(), RunState()
    calls = {
        "allow": ToolCall(tenant_id=TENANT, tool="read", args={"path": "a.txt"}),
        "deny_pattern": ToolCall(tenant_id=TENANT, tool="write", args={"vendor": "evil-1", "cost": 10}),
        "ask_threshold": ToolCall(tenant_id=TENANT, tool="write", args={"vendor": "acme-1", "cost": 900}),
        "not_on_allowlist": ToolCall(tenant_id=TENANT, tool="drop_database", args={}),
    }
    n = 5_000 if quick else 50_000
    return {name: summary(timed(lambda c=call: decide(policy, c, state), n)) for name, call in calls.items()}


# ---- audit ----


def bench_audit(quick: bool, url: str, **_: Any) -> dict[str, Any]:
    rows = 300 if quick else 3_000
    run = f"bench-{uuid.uuid4().hex}"
    with psycopg.connect(url, autocommit=True) as conn:
        create_audit_log(conn)
        try:
            counter = iter(range(rows))
            append_samples = timed(
                lambda: append(conn, run, TENANT, f"e{next(counter)}", "decision", {"verdict": "allow", "n": 1}), rows
            )
            started = time.perf_counter()
            result = verify(conn, run)
            verify_seconds = time.perf_counter() - started
        finally:
            conn.execute("DELETE FROM audit_log WHERE run_id = %s", (run,))
    return {
        "append": summary(append_samples) | {"rows_per_second": rows / sum(append_samples)},
        "verify": {"rows": rows, "ok": result.ok, "seconds": verify_seconds, "microseconds_per_row": verify_seconds / rows * 1e6},
    }


# ---- server ----


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def bench_server(quick: bool, url: str, **_: Any) -> dict[str, Any]:
    import uvicorn

    from portcullis.server import create_app

    key = new_key()
    keys = [KeyEntry(sha256=hash_key(key), tenant=TENANT, role="agent", name="bench")]
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(bench_policy_object(), keys, url), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base, headers = f"http://127.0.0.1:{port}", {"Authorization": f"Bearer {key}"}
    with httpx.Client(base_url=base, headers=headers, timeout=60) as client:
        for _ in range(100):
            try:
                if client.get("/healthz").status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.1)
        results: dict[str, Any] = {}
        try:
            lengths = (10, 100) if quick else (10, 100, 500, 1000)
            sample = 10 if quick else 30
            for length in lengths:
                run = f"bench-{uuid.uuid4().hex[:12]}"

                def one(i: int, run=run) -> float:
                    started = time.perf_counter()
                    reply = client.post(f"/v1/runs/{run}/calls", json={"call_id": f"c{i}", "tool": "read", "args": {"n": i}})
                    reply.raise_for_status()
                    return time.perf_counter() - started

                for i in range(length):  # grow the run to this length (not timed)
                    one(i)
                samples = [one(length + j) for j in range(sample)]
                results[f"after_{length}_calls"] = summary(samples)
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            with psycopg.connect(url, autocommit=True) as conn:
                conn.execute("DELETE FROM audit_log WHERE tenant_id = %s", (TENANT,))
                conn.execute("DELETE FROM spend_ledger WHERE tenant = %s", (TENANT,))
    return results


# ---- hook (a fresh process per call, like Claude Code runs it) ----


def bench_hook(quick: bool, url: str, **_: Any) -> dict[str, Any]:
    launcher = ROOT / "plugin" / "scripts" / "policy_hook.py"
    n = 8 if quick else 25
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as folder:
        policy_file = Path(folder) / "policy.json"
        policy_file.write_text(bench_policy_object().model_dump_json(), encoding="utf-8")
        base_env = {k: v for k, v in os.environ.items() if not k.startswith("PORTCULLIS_")}
        base_env.update(PORTCULLIS_POLICY=str(policy_file), PORTCULLIS_TENANT=TENANT, PYTHONPATH=str(ROOT / "src"))
        session = f"bench-{uuid.uuid4().hex[:8]}"
        full = {**base_env, "PORTCULLIS_HOOK_MODULE": "portcullis.hook"}
        fast = {**base_env, "PORTCULLIS_HOOK_MODULE": "portcullis.quickhook"}  # decides without pydantic when it can
        runs = (
            ("no_database", full),
            ("no_database_fast", fast),
            ("with_database", {**full, "PORTCULLIS_DATABASE_URL": url}),
        )
        for label, env in runs:
            samples = []
            for i in range(n):
                event = json.dumps({"tool_name": "read", "tool_input": {"n": i}, "session_id": f"{session}-{label}", "tool_use_id": f"t{i}"})
                started = time.perf_counter()
                done = subprocess.run([sys.executable, str(launcher)], input=event.encode(), env=env, capture_output=True, timeout=60)
                samples.append(time.perf_counter() - started)
                if done.returncode != 0:
                    raise RuntimeError(f"the hook blocked a call it should allow: {done.stderr[-300:]!r}")
            results[label] = summary(samples)
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("DELETE FROM audit_log WHERE tenant_id = %s", (TENANT,))
    return results


# ---- MCP proxy overhead ----


def bench_mcp(quick: bool, **_: Any) -> dict[str, Any]:
    from mcp import Client
    from mcp.server.mcpserver import MCPServer

    from portcullis.mcp_proxy import Governor, build_server

    downstream = MCPServer("bench")

    @downstream.tool()
    def read(n: int) -> str:
        """Return n."""
        return str(n)

    n = 200 if quick else 1500

    async def measure() -> dict[str, Any]:
        async with Client(downstream) as direct:
            direct_samples = []
            for i in range(n):
                started = time.perf_counter()
                await direct.call_tool("read", {"n": i})
                direct_samples.append(time.perf_counter() - started)
            governor = Governor(bench_policy_object(), TENANT, direct)
            async with Client(build_server(governor)) as through:
                proxied_samples = []
                for i in range(n):
                    started = time.perf_counter()
                    await through.call_tool("read", {"n": i})
                    proxied_samples.append(time.perf_counter() - started)
        return {"direct": summary(direct_samples), "through_proxy": summary(proxied_samples)}

    result = asyncio.run(measure())
    result["added_median_ms"] = result["through_proxy"]["median_ms"] - result["direct"]["median_ms"]
    return result


BENCHMARKS: dict[str, Callable[..., dict[str, Any]]] = {
    "policy": bench_policy,
    "audit": bench_audit,
    "server": bench_server,
    "hook": bench_hook,
    "mcp": bench_mcp,
}


def environment(url: str) -> dict[str, Any]:
    """What this ran on. No host name, user name or paths: this is published."""
    with psycopg.connect(url) as conn:
        postgres = conn.execute("SHOW server_version").fetchone()[0]
    return {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "postgres": postgres,
        "postgres_where": "local Docker container, loopback",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="far fewer samples: a smoke test, not a measurement")
    parser.add_argument("--only", action="append", choices=list(BENCHMARKS), help="run only this benchmark (repeatable)")
    parser.add_argument("--out", help="write the results to this JSON file")
    args = parser.parse_args(argv)

    url = os.environ.get("PORTCULLIS_DATABASE_URL")
    if not url:
        print("Set PORTCULLIS_DATABASE_URL (see .env.example).", file=sys.stderr)
        return 2
    try:
        require_project_database(url)
    except ValueError as exc:
        print(f"Refusing to run: {exc}.", file=sys.stderr)
        return 2

    report: dict[str, Any] = {"environment": environment(url), "quick": args.quick, "results": {}}
    for name in args.only or list(BENCHMARKS):
        print(f"running {name} ...", flush=True)
        report["results"][name] = BENCHMARKS[name](quick=args.quick, url=url)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
