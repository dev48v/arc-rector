"""L6 zero-dependency loader: text, Markdown, and plain HTML.

Docling is the default because it understands PDF layout, tables and reading
order. But most demo corpora are Markdown, and running a transformer-based
layout model over a .md file is pure waste -- so this loader handles the plain
cases, and the Docling adapter delegates to it for those extensions.

It also fetches URLs and strips tags with a small HTML parser, which is enough
for clean article pages and honestly not enough for JavaScript-rendered ones
(that is what the Firecrawl and Scrapy adapters are for).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ...interfaces import Loader
from ...registry import require
from ...types import Document

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".yaml", ".yml", ".log"}
HTML_SUFFIXES = {".html", ".htm"}

_SKIP_TAGS = {"script", "style", "noscript", "head", "meta", "link"}


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

    def __init__(self, *, timeout: int = 30, user_agent: str = "arc-rector/0.1", **_: Any) -> None:
        self.timeout = timeout
        self.user_agent = user_agent

    def supports(self, source: str) -> bool:
        if is_url(source):
            return True
        suffix = Path(source).suffix.lower()
        return suffix in TEXT_SUFFIXES or suffix in HTML_SUFFIXES

    def load(self, source: str) -> Document:
        if is_url(source):
            return self._load_url(source)
        return self._load_file(source)

    def _load_url(self, url: str) -> Document:
        requests = require("requests", "plaintext")
        response = requests.get(url, timeout=self.timeout, headers={"User-Agent": self.user_agent})
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "html" in content_type:
            text, title = html_to_text(response.text)
        else:
            text, title = response.text, url
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
