"""L6 default: Docling (MIT, IBM Research).

Docling is the default parser because it is the best open-source answer to the
part of RAG nobody demos: a real PDF. It reconstructs reading order, keeps table
structure, and exports clean Markdown -- which matters enormously downstream,
because a two-column PDF flattened naively produces chunks that interleave two
unrelated sentences and quietly wreck retrieval.

Two practical notes:
  * It is heavy. Docling pulls torch and layout models (~1 GB+ on first run,
    which it downloads on first parse). Everything else in this repo is light, so
    this is the one install worth knowing about before you run it.
  * It is wasted on Markdown. `.md`/`.txt` files are delegated to
    `PlaintextLoader` -- same result, no model load. `fallback_to_plaintext`
    (default on) extends that to any format Docling cannot handle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...interfaces import Loader
from ...registry import require
from ...types import Document
from .plaintext import TEXT_SUFFIXES, PlaintextLoader, assert_fetchable, is_url

DOCLING_SUFFIXES = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".epub",
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".adoc", ".xml",
}


class DoclingLoader(Loader):
    name = "docling"

    def __init__(self, *, fallback_to_plaintext: bool = True, ocr: bool = False, **_: Any) -> None:
        self.fallback_to_plaintext = fallback_to_plaintext
        self.ocr = ocr
        self._converter: Any = None
        self._plain = PlaintextLoader()

    @property
    def converter(self) -> Any:
        """Built on first use -- constructing it loads the layout models."""
        if self._converter is None:
            dc = require("docling", "docling", "docling.document_converter")
            self._converter = dc.DocumentConverter()
        return self._converter

    def supports(self, source: str) -> bool:
        if is_url(source):
            return True
        return Path(source).suffix.lower() in DOCLING_SUFFIXES or self._plain.supports(source)

    def load(self, source: str) -> Document:
        suffix = Path(source).suffix.lower()

        # Docling fetches URLs through its own machinery, which does not go
        # through the loader's SSRF gate -- so check before handing it over.
        if is_url(source):
            assert_fetchable(source, allow_private_hosts=self._plain.allow_private_hosts)

        # Plain text formats gain nothing from a layout model.
        if not is_url(source) and suffix in TEXT_SUFFIXES:
            doc = self._plain.load(source)
            doc.metadata["loader"] = "plaintext (delegated by docling)"
            return doc

        try:
            result = self.converter.convert(source)
            text = result.document.export_to_markdown()
            name = Path(source).name if not is_url(source) else source
            title = getattr(result.document, "name", "") or Path(source).stem
            return Document(
                doc_id=name,
                text=text,
                source=str(source),
                title=str(title),
                metadata={"loader": self.name, "suffix": suffix},
            )
        except Exception as exc:
            if not self.fallback_to_plaintext or not self._plain.supports(source):
                raise
            doc = self._plain.load(source)
            doc.metadata["loader"] = "plaintext (docling fallback)"
            doc.metadata["docling_error"] = str(exc)[:200]
            return doc
