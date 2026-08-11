"""L1 alternative: Arize Phoenix (Apache-2.0), self-hosted, OpenTelemetry-native.

What this option buys you: no proprietary tracing API. Langfuse gives you a good
UI behind an SDK of its own; Phoenix gives you plain OpenTelemetry with
OpenInference semantic conventions on top, so the same spans this adapter emits
can be fanned out to Jaeger, Tempo or any OTLP collector without rewriting a
line. If you already run OTel, Phoenix slots in beside it rather than beside-and-
duplicating it. It also ships evaluation and dataset tooling in the same UI,
which is the pitch for keeping L1 tracing and L1 eval in one place.

The gotcha: `phoenix.otel.register()` sets a *global* tracer provider by default,
which quietly hijacks any OTel setup already present in the process -- the classic
"why did my other traces vanish" bug. This adapter defaults
`set_global_tracer_provider=False` and keeps its own provider handle, so Phoenix
is additive. Flip it to True only if Phoenix is the only tracer in the process
and you want auto-instrumentation of other libraries to land here.

The other thing worth knowing: Phoenix buffers spans. A short-lived CLI run can
exit before anything is sent, so `flush()` is not optional here the way it feels
optional in a long-running service.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from ...interfaces import SpanHandle, Tracer
from ...registry import require

# OpenInference conventions, so the Phoenix UI renders these fields as it expects.
ATTRIBUTE_ALIASES: dict[str, str] = {
    "input": "input.value",
    "output": "output.value",
    "model": "llm.model_name",
    "metadata": "metadata",
}


def _scalar(value: Any) -> Any:
    """OTel attributes take str/bool/int/float only. Everything else is stringified."""
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(v, (str, bool, int, float)) for v in value
    ):
        return list(value)
    return repr(value)


class _PhoenixSpan(SpanHandle):
    def __init__(self, span: Any) -> None:
        self._span = span

    def update(self, **fields: Any) -> None:
        for key, value in fields.items():
            if value is None:
                continue
            try:
                self._span.set_attribute(ATTRIBUTE_ALIASES.get(key, key), _scalar(value))
            except Exception:
                # Telemetry must never break the request it is observing.
                pass


class _DeadSpan(SpanHandle):
    """Stand-in used when Phoenix is unreachable, so spans stay no-ops."""

    def update(self, **fields: Any) -> None:
        pass


class PhoenixTracer(Tracer):
    name = "phoenix"

    def __init__(
        self,
        *,
        host: str = "http://localhost:6006",
        endpoint: str = "",
        project_name: str = "arc-rector",
        batch: bool = True,
        auto_instrument: bool = False,
        set_global_tracer_provider: bool = False,
        enabled: bool = True,
        **_: Any,
    ) -> None:
        self.host = os.environ.get("PHOENIX_HOST", host).rstrip("/")
        self.endpoint = (endpoint or f"{self.host}/v1/traces").rstrip("/")
        self.project_name = project_name
        self.batch = batch
        self.auto_instrument = auto_instrument
        self.set_global_tracer_provider = set_global_tracer_provider
        self.enabled = enabled
        self._provider: Any = None
        self._tracer: Any = None
        self._failed = False
        self._trace_id = ""

    @property
    def tracer(self) -> Any:
        """Register with Phoenix on first span so import never needs the package."""
        if self._tracer is None and not self._failed and self.enabled:
            try:
                otel = require("arize-phoenix-otel", self.name, import_name="phoenix.otel")
                self._provider = otel.register(
                    project_name=self.project_name,
                    endpoint=self.endpoint,
                    batch=self.batch,
                    auto_instrument=self.auto_instrument,
                    set_global_tracer_provider=self.set_global_tracer_provider,
                )
                self._tracer = self._provider.get_tracer(__name__)
            except Exception:
                self._failed = True
                self._tracer = None
        return self._tracer

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[SpanHandle]:
        tracer = self.tracer
        if tracer is None:
            yield _DeadSpan()
            return

        try:
            ctx = tracer.start_as_current_span(name)
        except Exception:
            yield _DeadSpan()
            return

        with ctx as raw_span:
            handle = _PhoenixSpan(raw_span)
            # Span kind tells the Phoenix UI how to render the row.
            attrs.setdefault("openinference.span.kind", attrs.pop("as_type", "chain").upper())
            handle.update(**attrs)
            self._remember_trace_id(raw_span)
            try:
                yield handle
            except Exception as exc:
                try:
                    raw_span.record_exception(exc)
                    raw_span.set_attribute("error", True)
                except Exception:
                    pass
                raise

    def _remember_trace_id(self, span: Any) -> None:
        try:
            trace_id = span.get_span_context().trace_id
            if trace_id:
                self._trace_id = format(trace_id, "032x")
        except Exception:
            pass

    def flush(self) -> None:
        """Not optional for a CLI: batched spans are lost if the process just exits."""
        if self._provider is not None:
            try:
                self._provider.force_flush()
            except Exception:
                pass

    def last_trace_id(self) -> str:
        return self._trace_id

    def trace_url(self) -> str:
        base = f"{self.host}/projects"
        if not self._trace_id:
            return base
        return f"{base}?traceId={self._trace_id}"

    def reachable(self) -> bool:
        """Health probe used by `arc-rector doctor`."""
        try:
            requests = require("requests", self.name)
            return requests.get(f"{self.host}/healthz", timeout=5).status_code == 200
        except Exception:
            return False
