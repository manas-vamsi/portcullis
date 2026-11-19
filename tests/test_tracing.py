"""Tracing: spans carry a fixed set of non-sensitive attributes, never get in the way, and cost nothing when off."""

import asyncio
import hashlib
import os
import sys

import pytest
from harness import Harness, reply, tool_call

pytest.importorskip("opentelemetry.sdk")
from mcp import types  # noqa: E402
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from test_mcp_proxy import FakeDownstream, governor  # noqa: E402
from test_server import make_policy as server_policy  # noqa: E402
from test_server import place  # noqa: E402

from portcullis import tracing  # noqa: E402
from portcullis.server import Gateway, open_pool  # noqa: E402


@pytest.fixture(scope="session")
def exporter():
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    trace.set_tracer_provider(provider)  # can be set once per process
    return memory


@pytest.fixture
def spans(exporter):
    exporter.clear()
    yield lambda: [(s.name, dict(s.attributes)) for s in exporter.get_finished_spans()]
    exporter.clear()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# --- what a span may carry ---


def test_only_the_named_attributes_get_through_and_the_run_name_is_hashed(spans):
    with tracing.span("portcullis.test", tenant="acme", tool="Read", run="run-secret-name", verdict="allow", rule="allowed",
                      args={"password": "hunter2"}, api_key="sk-123", status="allowed", surface="server", approved=True):
        pass
    [(name, attributes)] = spans()
    assert name == "portcullis.test"
    assert attributes == {
        "portcullis.tenant": "acme", "portcullis.tool": "Read", "portcullis.run": digest("run-secret-name"),
        "portcullis.verdict": "allow", "portcullis.rule": "allowed", "portcullis.status": "allowed",
        "portcullis.surface": "server", "portcullis.approved": True,
    }
    assert "run-secret-name" not in str(attributes) and "hunter2" not in str(attributes) and "sk-123" not in str(attributes)


def test_values_that_are_not_plain_scalars_are_dropped_and_long_text_is_cut(spans):
    with tracing.span("portcullis.test", tool=["a", "b"], rule={"x": 1}, tenant="t" * 500, verdict=None, status=3.5):
        pass
    [(_, attributes)] = spans()
    assert attributes == {"portcullis.tenant": "t" * 128, "portcullis.status": 3.5}


def test_annotate_adds_attributes_to_the_span_that_is_open_and_does_nothing_otherwise(spans):
    tracing.annotate(verdict="allow")  # no span open: nothing happens
    with tracing.span("portcullis.test", tool="Read"):
        tracing.annotate(verdict="deny", rule="tool_not_allowed", args={"secret": "x"})
    [(_, attributes)] = spans()
    assert attributes["portcullis.verdict"] == "deny" and attributes["portcullis.rule"] == "tool_not_allowed"
    assert not any("secret" in key or "args" in key for key in attributes)


def test_an_exception_in_the_body_still_propagates_and_is_recorded_on_the_span(exporter, spans):
    with pytest.raises(ValueError, match="boom"):
        with tracing.span("portcullis.test", tool="Read"):
            raise ValueError("boom")
    [finished] = exporter.get_finished_spans()
    assert finished.status.status_code.name == "ERROR"
    assert [event.name for event in finished.events] == ["exception"]


# --- it never gets in the way ---


def test_a_tracer_that_breaks_does_not_break_the_work(monkeypatch):
    class Broken:
        def start_as_current_span(self, *args, **kwargs):
            raise RuntimeError("exporter on fire")

    monkeypatch.setattr(tracing, "_tracer", lambda: Broken())
    ran = []
    with tracing.span("portcullis.test", tool="Read"):
        ran.append(1)
        tracing.annotate(verdict="allow")  # also harmless
    assert ran == [1]


def test_annotating_a_span_that_cannot_take_attributes_is_harmless(monkeypatch):
    class Odd:
        def is_recording(self):
            raise RuntimeError("no")

    monkeypatch.setattr(trace, "get_current_span", lambda: Odd())
    tracing.annotate(verdict="allow")


def test_without_the_tracing_library_everything_still_works_and_emits_nothing(monkeypatch, spans):
    monkeypatch.setattr(tracing, "trace", None)  # as if opentelemetry were not installed
    ran = []
    with tracing.span("portcullis.test", tool="Read"):
        ran.append(1)
        tracing.annotate(verdict="allow")
    assert ran == [1] and spans() == []


# --- setting it up ---


def test_tracing_is_off_unless_an_endpoint_is_configured():
    assert tracing.setup_from_env({}) == "disabled"
    assert tracing.setup_from_env({"OTEL_EXPORTER_OTLP_ENDPOINT": ""}) == "disabled"


def test_an_endpoint_installs_an_exporting_provider_without_contacting_it(monkeypatch):
    installed = []
    monkeypatch.setattr(trace, "set_tracer_provider", installed.append)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")  # nothing listens there
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    result = tracing.setup_from_env(os.environ)
    try:
        assert result == "configured" and len(installed) == 1
        assert installed[0].resource.attributes["service.name"] == "portcullis"
    finally:
        installed[0].shutdown()


def test_a_missing_exporter_is_reported_not_raised(monkeypatch):
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", None)
    assert tracing.setup_from_env({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:9"}) == "unavailable"


# --- spans on each surface ---


def test_the_server_traces_a_decision_without_the_arguments(postgres_url, conn, spans):
    pool = open_pool(postgres_url)
    run = f"trace-{digest(postgres_url)}-1"
    try:
        gateway = Gateway(server_policy(), pool)
        gateway.decide_call("acme", run, "c1", "create_order_that_does_not_exist", {"password": "hunter2-very-secret"}, None)
        view = place(gateway, run, "c2", cost=900)
    finally:
        conn.execute("DELETE FROM audit_log WHERE run_id LIKE %s", (f"%{run}%",))
        conn.execute("DELETE FROM spend_ledger WHERE run_id = %s", (run,))
        pool.close()

    recorded = spans()
    decide = [a for n, a in recorded if n == "portcullis.decide"]
    assert [(a["portcullis.surface"], a["portcullis.tenant"], a["portcullis.status"], a["portcullis.rule"]) for a in decide] == [
        ("server", "acme", "denied", "tool_not_allowed"),
        ("server", "acme", "pending_approval", "approval_required"),
    ]
    assert decide[0]["portcullis.run"] == digest(run) and view["status"] == "pending_approval"
    assert "hunter2" not in str(recorded)


def test_the_server_traces_an_approval(postgres_url, conn, spans):
    pool = open_pool(postgres_url)
    run = f"trace-{digest(postgres_url)}-2"
    try:
        gateway = Gateway(server_policy(), pool)
        place(gateway, run, "big", cost=900)
        gateway.answer("acme", run, "big", True, "alice")
    finally:
        conn.execute("DELETE FROM audit_log WHERE run_id LIKE %s", (f"%{run}%",))
        conn.execute("DELETE FROM spend_ledger WHERE run_id = %s", (run,))
        pool.close()
    [approval] = [a for n, a in spans() if n == "portcullis.approval"]
    assert (approval["portcullis.surface"], approval["portcullis.status"], approval["portcullis.approved"]) == ("server", "approved", True)
    assert "alice" not in str(approval)  # who approved is in the audit log, not in traces


def test_the_agent_loop_traces_every_call(conn, run_id, spans):
    script = [reply(tool_call("a", "check_vendor_availability", trade="plumber"), tool_call("b", "drop_database")), reply(content="done")]
    Harness(conn, run_id, script).start()
    decided = [a for n, a in spans() if n == "portcullis.decide"]
    assert [(a["portcullis.surface"], a["portcullis.tool"], a["portcullis.verdict"], a["portcullis.rule"]) for a in decided] == [
        ("agent", "check_vendor_availability", "allow", "allowed"),
        ("agent", "drop_database", "deny", "tool_not_allowed"),
    ]
    assert all(a["portcullis.run"] == digest(run_id) for a in decided)


def test_the_mcp_proxy_traces_every_call(spans):
    g, _ = governor(FakeDownstream())
    asyncio.run(g.call_tool(None, types.CallToolRequestParams(name="check_vendor", arguments={"trade": "secret-trade"})))
    asyncio.run(g.call_tool(None, types.CallToolRequestParams(name="export_all", arguments={})))
    recorded = spans()
    decided = [a for n, a in recorded if n == "portcullis.decide"]
    assert [(a["portcullis.surface"], a["portcullis.tool"], a["portcullis.verdict"], a["portcullis.rule"]) for a in decided] == [
        ("mcp", "check_vendor", "allow", "allowed"),
        ("mcp", "export_all", "deny", "tool_not_allowed"),
    ]
    assert "secret-trade" not in str(recorded)


def test_decisions_are_identical_with_tracing_off(postgres_url, conn, monkeypatch):
    pool = open_pool(postgres_url)
    run = f"trace-{digest(postgres_url)}-3"
    try:
        gateway = Gateway(server_policy(), pool)
        with_tracing = gateway.decide_call("acme", run + "a", "c1", "check_vendor_availability", {"n": 1}, None)
        monkeypatch.setattr(tracing, "trace", None)
        without = gateway.decide_call("acme", run + "b", "c1", "check_vendor_availability", {"n": 1}, None)
    finally:
        conn.execute("DELETE FROM audit_log WHERE run_id LIKE %s", (f"%{run}%",))
        pool.close()
    assert with_tracing == without


def test_a_service_name_from_the_environment_is_used(monkeypatch):
    installed = []
    monkeypatch.setattr(trace, "set_tracer_provider", installed.append)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "my-gateway")
    try:
        assert tracing.setup_from_env(os.environ) == "configured"
        assert installed[0].resource.attributes["service.name"] == "my-gateway"
    finally:
        installed[0].shutdown()
