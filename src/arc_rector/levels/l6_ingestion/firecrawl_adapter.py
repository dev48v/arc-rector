"""L6 alternative: Firecrawl, pointed at your own container (AGPL-3.0 core).

What this buys you is the messy half of the web. A plain HTTP GET returns the
server's HTML; modern documentation sites return an empty shell and fill it in
with JavaScript. Firecrawl runs a real browser, waits for the page, strips nav
and cookie banners, and hands back clean markdown -- which is already the shape a
chunker wants. For scraping docs sites into a corpus it is hard to beat.

This adapter deliberately targets the SELF-HOSTED instance (`localhost:3002` by
default), not `api.firecrawl.dev`. Arc Rector is a zero-key stack, and Firecrawl
self-hosts from the same repo as the cloud product with `USE_DB_AUTHENTICATION=
false`, so no key is needed at all. An `api_key` is still accepted (falling back
to the FIRECRAWL_API_KEY env var) for anyone pointing `base_url` at the cloud;
it is sent as a bearer token and is never logged, printed, or put in an error.

The real gotcha is version drift between the two. The cloud API has moved on to
`/v2/scrape`, but a self-hosted image built from the OSS repo still serves
`/v1/scrape` -- both route files exist in the repo, and which one your container
answers depends on when you pulled it. Hence `api_version`, defaulting to "v1"
for self-host. If scraping 404s, that knob is the first thing to flip.

Second gotcha: the self-hosted stack is several containers. If you only started
the API and not the browser/worker services, JS-heavy pages return an empty
markdown string with a cheerful `success: true`.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Sequence
from urllib.parse import urlparse

from ...interfaces import Loader
from ...registry import require
from ...types import Document


def _doc_id(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8", "replace")).hexdigest()[:16]


class FirecrawlLoader(Loader):
    name = "firecrawl"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:3002",
        api_version: str = "v1",
        api_key: str | None = None,
        formats: Sequence[str] | None = None,
        only_main_content: bool = True,
        wait_for: int = 0,
        timeout: int = 120,
        **_: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version.strip("/")
        # Env fallback only; the value is never rendered anywhere.
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY") or ""
        self.formats = list(formats) if formats else ["markdown"]
        self.only_main_content = only_main_content
        self.wait_for = wait_for
        self.timeout = timeout
        self._requests: Any = None

    @property
    def _http(self) -> Any:
        if self._requests is None:
            self._requests = require("requests", "firecrawl")
        return self._requests

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url}/{self.api_version}/scrape"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        # Self-hosted with auth disabled needs no header at all.
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def supports(self, source: str) -> bool:
        return urlparse(source).scheme in ("http", "https") if source else False

    def load(self, source: str) -> Document:
        if not self.supports(source):
            raise ValueError(f"Firecrawl loader only handles http/https URLs, got: {source}")

        payload: dict[str, Any] = {
            "url": source,
            "formats": self.formats,
            "onlyMainContent": self.only_main_content,
        }
        if self.wait_for:
            payload["waitFor"] = self.wait_for

        response = self._http.post(
            self._endpoint,
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        if response.status_code == 404:
            raise RuntimeError(
                f"Firecrawl returned 404 for {self._endpoint}. Self-hosted images serve "
                f"/v1/scrape while the cloud has moved to /v2/scrape -- try "
                f"api_version='v2' (or 'v1') for this instance."
            )
        if response.status_code != 200:
            # response.text is the server body; no request headers are echoed here.
            raise RuntimeError(
                f"Firecrawl scrape failed ({response.status_code}) at {self._endpoint}: "
                f"{response.text[:300]}\n"
                f"Is the self-hosted stack up?  docker compose up -d  in the firecrawl repo."
            )

        body = response.json()
        if not body.get("success", True):
            raise RuntimeError(f"Firecrawl reported failure for {source}: {body.get('error', '')}")

        data = body.get("data") or {}
        metadata = data.get("metadata") or {}
        text = str(data.get("markdown") or data.get("html") or data.get("rawHtml") or "")

        return Document(
            doc_id=_doc_id(source),
            text=text,
            source=str(metadata.get("sourceURL") or source),
            title=_first_string(metadata.get("title")) or urlparse(source).netloc,
            metadata={
                "loader": self.name,
                "api_version": self.api_version,
                "status_code": metadata.get("statusCode"),
                "description": _first_string(metadata.get("description")),
                "self_hosted": not self.api_key,
            },
        )


def _first_string(value: Any) -> str:
    """Firecrawl metadata fields are typed `string | string[]`."""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value) if value else ""
