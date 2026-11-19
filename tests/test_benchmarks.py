"""The benchmark script runs, measures sensible things and leaves nothing behind. (These are smoke tests, not measurements.)"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "benchmarks" / "run.py"
spec = importlib.util.spec_from_file_location("bench_run", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def leftovers(conn) -> int:
    return conn.execute("SELECT count(*) FROM audit_log WHERE tenant_id = %s", (bench.TENANT,)).fetchone()[0]


def test_the_summary_reports_nearest_rank_percentiles_in_milliseconds():
    stats = bench.summary([i / 1000 for i in range(1, 101)])  # 1 ms .. 100 ms
    assert stats["n"] == 100
    assert stats["median_ms"] == pytest.approx(50.5)
    assert stats["p95_ms"] == pytest.approx(95.0) and stats["p99_ms"] == pytest.approx(99.0)
    assert stats["mean_ms"] == pytest.approx(50.5)


def test_one_sample_is_its_own_median_and_percentiles():
    stats = bench.summary([0.002])
    assert (stats["median_ms"], stats["p95_ms"], stats["p99_ms"]) == (2.0, 2.0, 2.0)


def test_the_policy_benchmark_covers_the_four_kinds_of_answer():
    result = bench.bench_policy(quick=True)
    assert set(result) == {"allow", "deny_pattern", "ask_threshold", "not_on_allowlist"}
    assert all(r["n"] == 5000 and 0 < r["median_ms"] < 50 for r in result.values())


def test_the_audit_benchmark_verifies_its_chain_and_removes_its_rows(postgres_url, conn):
    result = bench.bench_audit(quick=True, url=postgres_url)
    assert result["verify"]["ok"] is True and result["verify"]["rows"] == 300
    assert result["append"]["rows_per_second"] > 0 and leftovers(conn) == 0


def test_the_server_benchmark_measures_growing_runs_and_removes_its_rows(postgres_url, conn):
    result = bench.bench_server(quick=True, url=postgres_url)
    assert set(result) == {"after_10_calls", "after_100_calls"} and all(r["n"] == 10 for r in result.values())
    assert leftovers(conn) == 0
    assert conn.execute("SELECT count(*) FROM spend_ledger WHERE tenant = %s", (bench.TENANT,)).fetchone()[0] == 0


def test_the_hook_benchmark_times_a_real_process_with_and_without_a_database(postgres_url, conn):
    result = bench.bench_hook(quick=True, url=postgres_url)
    assert set(result) == {"no_database", "with_database"} and all(r["n"] == 8 for r in result.values())
    assert leftovers(conn) == 0


def test_the_mcp_benchmark_compares_a_direct_call_with_one_through_the_proxy():
    result = bench.bench_mcp(quick=True)
    assert result["direct"]["n"] == result["through_proxy"]["n"] == 200
    assert result["added_median_ms"] == pytest.approx(result["through_proxy"]["median_ms"] - result["direct"]["median_ms"])


def test_the_environment_description_has_no_host_or_user_names(postgres_url):
    described = json.dumps(bench.environment(postgres_url))
    import getpass
    import socket

    assert getpass.getuser().lower() not in described.lower() and socket.gethostname().lower() not in described.lower()
    assert set(bench.environment(postgres_url)) >= {"date", "os", "cpu", "python", "postgres"}


def test_a_database_not_named_for_the_project_is_refused(monkeypatch, capsys):
    monkeypatch.setenv("PORTCULLIS_DATABASE_URL", "postgresql://u:hunter2@127.0.0.1:5432/other")
    assert bench.main(["--only", "policy"]) == 2
    assert "hunter2" not in capsys.readouterr().err


def test_a_missing_database_url_is_a_usage_error(monkeypatch, capsys):
    monkeypatch.delenv("PORTCULLIS_DATABASE_URL", raising=False)
    assert bench.main(["--only", "policy"]) == 2
    assert "PORTCULLIS_DATABASE_URL" in capsys.readouterr().err


def test_main_runs_the_chosen_benchmarks_and_writes_the_report(postgres_url, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PORTCULLIS_DATABASE_URL", str(postgres_url))
    out = tmp_path / "report.json"
    assert bench.main(["--quick", "--only", "policy", "--only", "audit", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["quick"] is True and set(report["results"]) == {"policy", "audit"}
    assert "environment" in report and "running policy" in capsys.readouterr().out
