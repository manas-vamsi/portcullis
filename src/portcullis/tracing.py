"""Optional OpenTelemetry tracing for decisions.

    uv sync --extra tracing
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 python -m portcullis.server

Each decision is a span named `portcullis.decide` (an approval is `portcullis.approval`) with a fixed, short
list of attributes: the surface it came through (server, agent or mcp), the tenant, the tool, the verdict and
rule or status, and a hash of the run name. Nothing else can be put on a span: arguments, results, approver
names, call ids and secrets are not in the list, and anything not in it is dropped (see ALLOWED). The run name
is hashed so traces can be grouped by run without revealing the name.

Rules this module keeps:
- It never gets in the way. If the tracing library is missing, misconfigured or broken, the work goes on and
  nothing is emitted. An exception from the traced work still propagates (and is recorded on the span).
- It costs little when it is off: with no provider installed OpenTelemetry's own no-op tracer is used. Measured
  on one machine, a traced decision adds about 12 microseconds with the library installed and no provider, about
  2 without the library, and about 49 when recording to memory, against a server decision of tens of milliseconds.
- The Claude Code hook does not trace: it is a new process per call, and exporting from a process that lives
  for one decision would add delay to every tool call.

On the agent loop, a call is traced again if LangGraph re-runs an interrupted node after an approval.
"""

import hashlib
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from typing import Any

try:
    from opentelemetry import trace
except ImportError:  # pragma: no cover  (the missing-library case is tested by setting `trace` to None)
    trace = None

ALLOWED = ("surface", "tenant", "tool", "run", "verdict", "rule", "status", "approved")
MAX_TEXT = 128


def _clean(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Only the allowed names, only plain scalars, long text cut, and the run name hashed."""
    cleaned: dict[str, Any] = {}
    for key, value in attributes.items():
        if key not in ALLOWED or value is None:
            continue
        if key == "run":
            value = hashlib.sha256(str(value).encode()).hexdigest()[:12]
        if isinstance(value, (bool, int, float)):
            cleaned[f"portcullis.{key}"] = value
        elif isinstance(value, str):
            cleaned[f"portcullis.{key}"] = value[:MAX_TEXT]
    return cleaned


def _tracer() -> Any:
    return trace.get_tracer("portcullis") if trace is not None else None


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """A span around the body, with the allowed attributes. Never raises on its own account."""
    stack = ExitStack()
    try:
        tracer = _tracer()
        if tracer is not None:
            stack.enter_context(tracer.start_as_current_span(name, attributes=_clean(attributes)))
    except Exception:  # tracing must never get in the way
        stack = ExitStack()
    with stack:  # passes an exception from the body to the span, which records it and lets it continue
        yield


def annotate(**attributes: Any) -> None:
    """Add allowed attributes to the span that is open now, if any."""
    if trace is None:
        return
    try:
        current = trace.get_current_span()
        if current.is_recording():
            for key, value in _clean(attributes).items():
                current.set_attribute(key, value)
    except Exception:  # tracing must never get in the way
        pass


def setup_from_env(env: Mapping[str, str]) -> str:
    """Export spans over OTLP/HTTP when OTEL_EXPORTER_OTLP_ENDPOINT is set. Returns what happened.

    "disabled": no endpoint, nothing changed. "unavailable": the tracing packages are not installed.
    "configured": a provider that exports to the endpoint is installed (the standard OTEL_* variables apply)."""
    if not env.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return "disabled"
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return "unavailable"
    if trace is None:  # pragma: no cover  (the sdk imported, so the api is there)
        return "unavailable"
    provider = TracerProvider(resource=Resource.create({"service.name": env.get("OTEL_SERVICE_NAME") or "portcullis"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return "configured"
