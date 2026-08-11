"""L1 offline option: tracing turned off.

Used by the test suite and by `make demo-offline`. Every span is a no-op context
manager, so the framework adapters keep their `with tracer.span(...)` structure
unchanged whether or not an observability backend exists.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from ...interfaces import SpanHandle, Tracer


class _NullSpan(SpanHandle):
    def update(self, **fields: Any) -> None:
        pass


class NoopTracer(Tracer):
    name = "none"

    def __init__(self, *, record: bool = False, **_: Any) -> None:
        # `record=True` keeps span names in memory, which the tests assert on to
        # prove each framework adapter really instruments every step.
        self.record = record
        self.spans: list[str] = []

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[SpanHandle]:
        if self.record:
            self.spans.append(name)
        yield _NullSpan()

    def flush(self) -> None:
        pass

    def last_trace_id(self) -> str:
        return ""

    def trace_url(self) -> str:
        return ""
