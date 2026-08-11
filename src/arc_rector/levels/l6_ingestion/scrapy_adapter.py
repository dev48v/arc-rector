"""L6 alternative: Scrapy (BSD-3-Clause) -- the crawler, not the fetcher.

What this buys you is everything that shows up once you need more than one page:
a scheduler with de-duplication, concurrency limits, autothrottle, retry
middleware, robots.txt obedience, and resumable jobs. If the task is "pull 4,000
pages off a docs site without getting banned", Scrapy is the right tool and
Firecrawl is the expensive one.

Be warned about the shape mismatch, because this file is where it bites. Arc
Rector's `Loader` contract is one-shot and synchronous: `load(source) ->
Document`. Scrapy is a Twisted application. `CrawlerProcess.run()` installs and
then *tears down* a global reactor in your process, and Twisted reactors cannot
be restarted -- calling it a second time raises `ReactorNotRestartable`. A
`load()` built on CrawlerProcess would therefore work exactly once per Python
process and blow up on the second document. That is not a hypothetical; it is
the single most reported Scrapy-in-a-library problem.

So this adapter is honest about doing two different things:

  * `load(url)` does NOT use the reactor. It fetches with `requests` and parses
    with Scrapy's own `Selector` (parsel), so single-page extraction is the same
    selector engine you would write in a spider, minus the framework.
  * `crawl(start_url, max_pages)` is the real thing: an actual `scrapy.Spider`
    driven by a real `CrawlerProcess`, returning a list of Documents. Call it
    once per process, and prefer running it as its own ingest step.

If you only ever load single pages, this adapter buys you nothing over the
plaintext loader plus an HTTP call -- the value is entirely in `crawl()`.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable
from urllib.parse import urlparse

from ...interfaces import Loader
from ...registry import require
from ...types import Document

_DROP_ANCESTORS = "not(ancestor::script) and not(ancestor::style) and not(ancestor::noscript)"


def _doc_id(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8", "replace")).hexdigest()[:16]


class ScrapyLoader(Loader):
    name = "scrapy"

    def __init__(
        self,
        *,
        user_agent: str = "arc-rector/1.0 (+https://github.com/arc-rector)",
        timeout: int = 30,
        obey_robots: bool = True,
        concurrent_requests: int = 8,
        download_delay: float = 0.25,
        depth_limit: int = 2,
        same_domain_only: bool = True,
        log_level: str = "ERROR",
        **_: Any,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.obey_robots = obey_robots
        self.concurrent_requests = concurrent_requests
        self.download_delay = download_delay
        self.depth_limit = depth_limit
        self.same_domain_only = same_domain_only
        self.log_level = log_level
        self._scrapy: Any = None
        self._requests: Any = None

    @property
    def _lib(self) -> Any:
        """Scrapy itself, imported lazily. `scrapy.Selector` is re-exported here."""
        if self._scrapy is None:
            self._scrapy = require("scrapy", "scrapy")
        return self._scrapy

    @property
    def _http(self) -> Any:
        if self._requests is None:
            self._requests = require("requests", "scrapy")
        return self._requests

    def supports(self, source: str) -> bool:
        return urlparse(source).scheme in ("http", "https") if source else False

    def load(self, source: str) -> Document:
        """Single page, no reactor. See the module docstring for why."""
        if not self.supports(source):
            raise ValueError(f"Scrapy loader only handles http/https URLs, got: {source}")

        response = self._http.get(
            source,
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Scrapy loader fetch failed ({response.status_code}) for {source}: "
                f"{response.text[:200]}"
            )

        title, text = self._extract(response.text)
        return Document(
            doc_id=_doc_id(source),
            text=text,
            source=source,
            title=title or urlparse(source).netloc,
            metadata={"loader": self.name, "mode": "single-page", "reactor": False},
        )

    def crawl(self, start_url: str, max_pages: int = 25) -> list[Document]:
        """Real multi-page crawl through CrawlerProcess. Once per Python process.

        Raises ReactorNotRestartable on a second call -- that is Twisted, not a
        bug here. Run each crawl in a fresh process (or a subprocess) if you need
        more than one.
        """
        if not self.supports(start_url):
            raise ValueError(f"Scrapy crawl needs an http/https URL, got: {start_url}")

        scrapy = self._lib
        crawler = require("scrapy", "scrapy", "scrapy.crawler")
        collected: list[Document] = []
        extract = self._extract
        allowed = urlparse(start_url).netloc if self.same_domain_only else ""

        class _OneShotSpider(scrapy.Spider):  # type: ignore[misc, name-defined]
            name = "arc_rector_oneshot"

            def parse(self, response: Any, **kwargs: Any) -> Iterable[Any]:
                if len(collected) >= max_pages:
                    return
                title, text = extract(response.text)
                collected.append(
                    Document(
                        doc_id=_doc_id(response.url),
                        text=text,
                        source=response.url,
                        title=title or urlparse(response.url).netloc,
                        metadata={
                            "loader": "scrapy",
                            "mode": "crawl",
                            "depth": response.meta.get("depth", 0),
                        },
                    )
                )
                for href in response.css("a::attr(href)").getall():
                    if len(collected) >= max_pages:
                        return
                    target = response.urljoin(href)
                    if urlparse(target).scheme not in ("http", "https"):
                        continue
                    if allowed and urlparse(target).netloc != allowed:
                        continue
                    yield scrapy.Request(target, callback=self.parse)

        process = crawler.CrawlerProcess(
            settings={
                "USER_AGENT": self.user_agent,
                "ROBOTSTXT_OBEY": self.obey_robots,
                "CONCURRENT_REQUESTS": self.concurrent_requests,
                "DOWNLOAD_DELAY": self.download_delay,
                "DOWNLOAD_TIMEOUT": self.timeout,
                "DEPTH_LIMIT": self.depth_limit,
                "CLOSESPIDER_PAGECOUNT": max_pages,
                "LOG_LEVEL": self.log_level,
                "TELNETCONSOLE_ENABLED": False,
            }
        )
        process.crawl(_OneShotSpider, start_urls=[start_url])
        # Blocks until the crawl finishes, then tears the reactor down for good.
        process.start()
        return collected[:max_pages]

    def _extract(self, html: str) -> tuple[str, str]:
        """Title and visible text via parsel, the selector engine Scrapy ships."""
        selector = self._lib.Selector(text=html)
        title = (selector.xpath("//title/text()").get() or "").strip()
        nodes = selector.xpath(f"//body//text()[{_DROP_ANCESTORS}]").getall()
        if not nodes:
            nodes = selector.xpath(f"//text()[{_DROP_ANCESTORS}]").getall()
        lines = [line for line in (n.strip() for n in nodes) if line]
        return title, "\n".join(lines)
