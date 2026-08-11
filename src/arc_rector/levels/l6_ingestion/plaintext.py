"""L6 zero-dependency loader: text, Markdown, and plain HTML.

Docling is the default because it understands PDF layout, tables and reading
order. But most demo corpora are Markdown, and running a transformer-based
layout model over a .md file is pure waste -- so this loader handles the plain
cases, and the Docling adapter delegates to it for those extensions.

It also fetches URLs and strips tags with a small HTML parser, which is enough
for clean article pages and honestly not enough for JavaScript-rendered ones
(that is what the Firecrawl and Scrapy adapters are for).

A URL loader is a server-side fetcher, so it is an SSRF primitive: whoever
supplies the URL chooses which host this process connects to. `assert_fetchable`
is the gate -- http(s) only, no credentials in the URL, and every resolved
address must be public. Redirects are followed by hand so each hop is re-checked
rather than trusted, and the body is capped so one URL cannot exhaust memory.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

from ...interfaces import Loader
from ...registry import require
from ...types import Document

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".yaml", ".yml", ".log"}
HTML_SUFFIXES = {".html", ".htm"}

_SKIP_TAGS = {"script", "style", "noscript", "head", "meta", "link"}

MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024
MAX_REDIRECTS = 5
ALLOWED_SCHEMES = ("http", "https")


class UnsafeURLError(ValueError):
    """A URL that a document loader must not fetch."""


class _TextExtractor(HTMLParser):
    """Collect visible text, skipping script/style, tracking <title>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "section", "article", "br", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data.strip()
            return
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        joined = " ".join(self.parts)
        joined = re.sub(r"[ \t]+", " ", joined)
        return re.sub(r"\s*\n\s*", "\n", joined).strip()


def is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _address_is_public(host: str) -> tuple[bool, str]:
    """Resolve `host` and report whether EVERY address it maps to is public.

    Every address, not the first: a name that returns one public and one
    loopback address would otherwise pass the check and connect to the loopback.
    """
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        return False, f"cannot resolve {host!r}: {exc}"
    if not infos:
        return False, f"cannot resolve {host!r}"

    for info in infos:
        raw = info[4][0]
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            return False, f"{host!r} resolved to an unparseable address {raw!r}"
        # link_local covers 169.254.0.0/16, which is the cloud metadata service
        # on EC2, GCE and Oracle Cloud -- the single most valuable SSRF target.
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False, f"{host!r} resolves to the non-public address {ip}"
    return True, ""


def assert_fetchable(url: str, *, allow_private_hosts: bool = False) -> None:
    """Raise `UnsafeURLError` unless `url` is safe for a server to fetch.

    Residual risk: DNS can change between this check and the connection
    (rebinding). Closing that needs a connect-to-pinned-IP transport; see
    PRODUCTION.md.
    """
    parts = urlsplit(url)
    if parts.scheme not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"only http(s) URLs can be ingested, got {parts.scheme or url!r}")
    if parts.username or parts.password:
        raise UnsafeURLError("credentials in the URL are not accepted")
    host = parts.hostname
    if not host:
        raise UnsafeURLError(f"no host in {url!r}")
    if allow_private_hosts:
        return
    public, reason = _address_is_public(host)
    if not public:
        raise UnsafeURLError(
            f"refusing to fetch {url!r}: {reason}. "
            "Set allow_private_hosts on the loader to ingest from an intranet."
        )


def html_to_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text(), parser.title


def _title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped:
            break
    return fallback


class PlaintextLoader(Loader):
    name = "plaintext"

    def __init__(
        self,
        *,
        timeout: int = 30,
        user_agent: str = "arc-rector/0.1",
        max_bytes: int = MAX_DOWNLOAD_BYTES,
        max_redirects: int = MAX_REDIRECTS,
        allow_private_hosts: bool = False,
        **_: Any,
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent
        self.max_bytes = int(max_bytes)
        self.max_redirects = int(max_redirects)
        self.allow_private_hosts = bool(allow_private_hosts)

    def supports(self, source: str) -> bool:
        if is_url(source):
            return True
        suffix = Path(source).suffix.lower()
        return suffix in TEXT_SUFFIXES or suffix in HTML_SUFFIXES

    def load(self, source: str) -> Document:
        if is_url(source):
            return self._load_url(source)
        return self._load_file(source)

    def _fetch(self, url: str) -> tuple[str, str]:
        """Fetch `url`, re-checking every redirect hop. Returns (body, content_type)."""
        requests = require("requests", "plaintext")
        current = url
        for _ in range(self.max_redirects + 1):
            assert_fetchable(current, allow_private_hosts=self.allow_private_hosts)
            response = requests.get(
                current,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
                allow_redirects=False,
                stream=True,
            )
            if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location", "")
                response.close()
                if not location:
                    raise UnsafeURLError(f"redirect from {current!r} with no Location header")
                # Re-checked on the next pass, which is the point of not letting
                # requests follow these itself.
                current = urljoin(current, location)
                continue

            response.raise_for_status()
            content_type = response.headers.get("content-type", "")

            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > self.max_bytes:
                response.close()
                raise UnsafeURLError(
                    f"{current!r} declares {declared} bytes, over the {self.max_bytes}-byte cap"
                )

            body = bytearray()
            for block in response.iter_content(chunk_size=65536):
                body.extend(block)
                # Checked while reading: Content-Length is a claim, not a limit.
                if len(body) > self.max_bytes:
                    response.close()
                    raise UnsafeURLError(
                        f"{current!r} exceeded the {self.max_bytes}-byte download cap"
                    )
            encoding = response.encoding or response.apparent_encoding or "utf-8"
            response.close()
            return bytes(body).decode(encoding, errors="replace"), content_type

        raise UnsafeURLError(f"more than {self.max_redirects} redirects starting at {url!r}")

    def _load_url(self, url: str) -> Document:
        raw, content_type = self._fetch(url)
        if "html" in content_type:
            text, title = html_to_text(raw)
        else:
            text, title = raw, url
        return Document(
            doc_id=url,
            text=text,
            source=url,
            title=title or url,
            metadata={"loader": self.name, "content_type": content_type},
        )

    def _load_file(self, path_str: str) -> Document:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        raw = path.read_text(encoding="utf-8", errors="replace")
        suffix = path.suffix.lower()
        if suffix in HTML_SUFFIXES:
            text, title = html_to_text(raw)
            title = title or path.stem
        else:
            text = raw
            title = _title_from_markdown(raw, path.stem)
        return Document(
            doc_id=path.name,
            text=text,
            source=str(path),
            title=title,
            metadata={"loader": self.name, "suffix": suffix},
        )
