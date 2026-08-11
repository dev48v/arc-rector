"""L1 default: Langfuse (MIT), self-hosted in this repo's docker-compose.

Langfuse is the default observability layer because the whole thing -- UI,
ingestion, storage -- runs in containers you own, and the compose file seeds an
organisation, project and API key pair via `LANGFUSE_INIT_*` so tracing works on
first boot with nothing to click and no account to create.

SDK note: Langfuse v3 renamed the tracing API. v2's `langfuse.trace()` /
`trace.span()` are gone; v4 is OpenTelemetry-based and uses
`start_as_current_observation(as_type="span"|"generation")`. This adapter targets
v4 and falls back to the v3 spelling, because that rename is the single most
common reason a copied Langfuse snippet fails.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any, Iterator

from ...interfaces import SpanHandle, Tracer
from ...registry import require


class _LangfuseSpan(SpanHandle):
    def __init__(self, span: Any) -> None:
        self._span = span

    def update(self, **fields: Any) -> None:
        try:
            self._span.update(**fields)
        except Exception:
            # Telemetry must never break the request it is observing.
            pass


class _DeadSpan(SpanHandle):
    """Stand-in used when Langfuse is unreachable, so spans stay no-ops."""

    def update(self, **fields: Any) -> None:
        pass


class LangfuseTracer(Tracer):
    name = "langfuse"

    def __init__(
        self,
        *,
        host: str = "http://localhost:3000",
        public_key: str = "pk-lf-arc-rector-local",
        secret_key: str = "sk-lf-arc-rector-local",
        enabled: bool = True,
        **_: Any,
    ) -> None:
        self.host = os.environ.get("LANGFUSE_HOST", host).rstrip("/")
        self.public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", public_key)
        self.secret_key = os.environ.get("LANGFUSE_SECRET_KEY", secret_key)
        self.enabled = enabled
        self._client: Any = None
        self._failed = False
        self._trace_id = ""

    @property
    def client(self) -> Any:
        if self._client is None and not self._failed and self.enabled:
            try:
                lf = require("langfuse", "langfuse")
                # The SDK reads these from the environment on construction.
                os.environ.setdefault("LANGFUSE_HOST", self.host)
                os.environ.setdefault("LANGFUSE_PUBLIC_KEY", self.public_key)
                os.environ.setdefault("LANGFUSE_SECRET_KEY", self.secret_key)
                self._client = lf.Langfuse(
                    host=self.host, public_key=self.public_key, secret_key=self.secret_key
                )
            except Exception:
                self._failed = True
                self._client = None
        return self._client

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[SpanHandle]:
        client = self.client
        if client is None:
            yield _DeadSpan()
            return

        as_type = attrs.pop("as_type", "span")
        span_input = attrs.pop("input", None)
        payload: dict[str, Any] = {"name": name}
        if span_input is not None:
            payload["input"] = span_input
        if attrs:
            payload["metadata"] = attrs
        if as_type == "generation" and "model" in attrs:
            payload["model"] = attrs.get("model")

        try:
            opener = getattr(client, "start_as_current_observation", None)
            if opener is not None:
                ctx = opener(as_type=as_type, **payload)
            else:  # Langfuse v3 spelling
                fallback = (
                    client.start_as_current_generation
                    if as_type == "generation"
                    else client.start_as_current_span
                )
                ctx = fallback(**payload)
        except Exception:
            yield _DeadSpan()
            return

        # Enter and exit the SDK's context manager by hand rather than with a
        # `try: with ctx: yield ... except Exception: yield _DeadSpan()`.
        # That shape looks equivalent and is not: when the *caller's* block
        # raises, the exception is thrown in at the yield, the except catches
        # it, and yielding a second time turns a real failure -- an Ollama read
        # timeout, say -- into `RuntimeError: generator didn't stop after
        # throw()` with the original traceback gone. Telemetry must never break
        # the request it observes, and it must never hide why the request broke.
        try:
            raw_span = ctx.__enter__()
        except Exception:
            yield _DeadSpan()
            return

        try:
            self._trace_id = str(getattr(raw_span, "trace_id", "") or self._trace_id)
        except Exception:
            pass

        try:
            yield _LangfuseSpan(raw_span)
        finally:
            try:
                ctx.__exit__(*sys.exc_info())
            except Exception:
                pass

    def flush(self) -> None:
        client = self.client
        if client is not None:
            try:
                client.flush()
            except Exception:
                pass

    def last_trace_id(self) -> str:
        return self._trace_id

    def trace_url(self) -> str:
        if not self._trace_id:
            return self.host
        return f"{self.host}/trace/{self._trace_id}"

    def reachable(self) -> bool:
        """Health probe used by `arc-rector doctor`."""
        try:
            requests = require("requests", "langfuse")
            return requests.get(f"{self.host}/api/public/health", timeout=5).status_code == 200
        except Exception:
            return False
