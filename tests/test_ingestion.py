"""L6 URL safety. A server-side fetcher is an SSRF primitive, so it is gated.

Offline like the rest of the suite: every case here is either a literal IP
(no DNS) or a monkeypatched resolver.
"""

from __future__ import annotations

import socket

import pytest

from arc_rector.levels.l6_ingestion.plaintext import (
    PlaintextLoader,
    UnsafeURLError,
    assert_fetchable,
    is_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:6333/collections",
        "http://localhost:3000/api",
        "https://10.0.0.5/internal",
        "http://192.168.1.1/admin",
        "http://172.16.0.1/",
        # The one that actually pays on a cloud VM: the instance metadata
        # service, which this project's own DEPLOY.md targets (Oracle A1).
        "http://169.254.169.254/opc/v2/instance/",
        "http://[::1]:8800/api/config",
    ],
)
def test_non_public_addresses_are_refused(url: str) -> None:
    with pytest.raises(UnsafeURLError):
        assert_fetchable(url)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://127.0.0.1:11211/_stats",
        "ftp://example.com/x",
        "data:text/plain,hello",
    ],
)
def test_only_http_schemes_are_accepted(url: str) -> None:
    with pytest.raises(UnsafeURLError):
        assert_fetchable(url)


def test_credentials_in_the_url_are_refused() -> None:
    with pytest.raises(UnsafeURLError, match="credentials"):
        assert_fetchable("https://user:pass@example.com/doc")


def test_a_public_address_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )
    assert_fetchable("https://example.com/article")


def test_a_name_resolving_to_both_public_and_loopback_is_refused(monkeypatch) -> None:
    """Checking only the first address is how this gate usually gets bypassed."""
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0)), (2, 1, 6, "", ("127.0.0.1", 0))],
    )
    with pytest.raises(UnsafeURLError):
        assert_fetchable("https://rebind.example.com/x")


def test_an_unresolvable_host_is_refused(monkeypatch) -> None:
    def boom(*a, **k):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(UnsafeURLError, match="cannot resolve"):
        assert_fetchable("https://nope.invalid/x")


def test_the_intranet_escape_hatch_is_explicit() -> None:
    assert_fetchable("http://10.0.0.5/wiki", allow_private_hosts=True)


def test_the_loader_defaults_to_refusing_private_hosts() -> None:
    loader = PlaintextLoader()
    assert loader.allow_private_hosts is False
    assert loader.max_bytes == 10 * 1024 * 1024
    with pytest.raises(UnsafeURLError):
        loader.load("http://169.254.169.254/latest/meta-data/")


def test_url_detection_is_unchanged() -> None:
    assert is_url("https://example.com")
    assert not is_url("corpus/01-overview.md")


def test_every_url_fetching_adapter_calls_the_gate() -> None:
    """Docling, unstructured and scrapy fetch through their own machinery.

    Gating only the plaintext loader would leave the DEFAULT adapter open, so
    each of them has to call `assert_fetchable` before handing a URL over.
    """
    import inspect

    from arc_rector.levels.l6_ingestion import (
        docling_adapter,
        scrapy_adapter,
        unstructured_adapter,
    )

    for module in (docling_adapter, unstructured_adapter, scrapy_adapter):
        source = inspect.getsource(module)
        assert "assert_fetchable(" in source, f"{module.__name__} does not gate URLs"
