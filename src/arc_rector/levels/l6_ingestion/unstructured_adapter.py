"""L6 alternative: Unstructured (Apache-2.0).

What this buys you is breadth. `unstructured` routes 60+ file types through one
`partition()` call -- PDF, DOCX, PPTX, XLSX, EML, MSG, EPUB, HTML, images -- and
hands back a list of typed elements (Title, NarrativeText, Table, ListItem)
rather than a wall of text. If your corpus is a shared drive full of mixed
formats, this is the loader that will not make you write per-format branches.

The gotcha, and the reason Docling is the default here, is the install. The base
`pip install unstructured` parses almost nothing on its own: PDF needs
`unstructured[pdf]`, and the real work happens in *system* packages that pip
cannot give you -- poppler-utils for PDF rendering, tesseract-ocr for scanned
pages, libmagic for filetype sniffing, libreoffice for Office formats. On
Windows in particular that is a manual afternoon. Docling ships its layout model
as a pip wheel and needs none of it. Breadth costs you a reproducible install.

Second-order gotcha: `strategy` decides how much of that heavy machinery gets
used. This adapter defaults to "fast" (text-layer extraction only, no OCR, no
layout model) so that a plain `pip install unstructured[pdf]` actually works. A
scanned PDF has no text layer, so "fast" returns empty text for it -- that is not
a bug, it is the strategy. Set strategy="hi_res" once tesseract is on PATH.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Sequence
from urllib.parse import urlparse

from ...interfaces import Loader
from ...registry import require
from ...types import Document

# Formats partition() can route without extra system packages beyond the extras.
_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".odt", ".rtf", ".pptx", ".ppt", ".xlsx", ".xls",
    ".csv", ".tsv", ".html", ".htm", ".xml", ".md", ".rst", ".org", ".txt",
    ".text", ".log", ".json", ".eml", ".msg", ".epub", ".png", ".jpg", ".jpeg",
    ".tiff", ".tif", ".bmp", ".heic",
}


def _doc_id(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8", "replace")).hexdigest()[:16]


class UnstructuredLoader(Loader):
    name = "unstructured"

    def __init__(
        self,
        *,
        strategy: str = "fast",
        languages: Sequence[str] | None = None,
        separator: str = "\n\n",
        request_timeout: int = 60,
        include_element_types: bool = False,
        min_element_chars: int = 0,
        **_: Any,
    ) -> None:
        self.strategy = strategy
        self.languages = list(languages) if languages else None
        self.separator = separator
        self.request_timeout = request_timeout
        self.include_element_types = include_element_types
        self.min_element_chars = min_element_chars

    def _partition(self) -> Any:
        """Import lazily: selecting this adapter must not require the install."""
        module = require("unstructured", "unstructured", "unstructured.partition.auto")
        return module.partition

    def supports(self, source: str) -> bool:
        if not source:
            return False
        if _is_url(source):
            return True
        return os.path.splitext(source)[1].lower() in _EXTENSIONS

    def load(self, source: str) -> Document:
        partition = self._partition()

        kwargs: dict[str, Any] = {"strategy": self.strategy}
        if self.languages:
            kwargs["languages"] = self.languages

        # partition() takes url= or filename=, never both.
        if _is_url(source):
            kwargs["url"] = source
            kwargs["request_timeout"] = self.request_timeout
        else:
            if not os.path.exists(source):
                raise FileNotFoundError(f"Unstructured loader: no such file: {source}")
            kwargs["filename"] = source

        elements = partition(**kwargs)

        parts: list[str] = []
        title = ""
        counts: dict[str, int] = {}
        for element in elements:
            kind = type(element).__name__
            counts[kind] = counts.get(kind, 0) + 1
            text = _element_text(element)
            if len(text) < self.min_element_chars:
                continue
            if not text:
                continue
            if not title and kind == "Title":
                title = text
            parts.append(f"[{kind}] {text}" if self.include_element_types else text)

        return Document(
            doc_id=_doc_id(source),
            text=self.separator.join(parts),
            source=source,
            title=title or _fallback_title(source),
            metadata={
                "loader": self.name,
                "strategy": self.strategy,
                "element_count": len(elements),
                "element_types": counts,
            },
        )


def _element_text(element: Any) -> str:
    """Elements expose .text; str() is the documented fallback for exotic types."""
    text = getattr(element, "text", None)
    if text is None:
        text = str(element)
    return str(text).strip()


def _fallback_title(source: str) -> str:
    if _is_url(source):
        return urlparse(source).netloc or source
    return os.path.splitext(os.path.basename(source))[0]


def _is_url(source: str) -> bool:
    return urlparse(source).scheme in ("http", "https")
