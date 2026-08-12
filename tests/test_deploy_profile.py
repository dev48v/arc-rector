"""The A1 deployment profile, checked the way a boot would check it.

Every assertion here corresponds to something that actually went wrong on the
reference box. None of it needs Docker: these are the properties whose failure
mode is a container that never becomes healthy, which is the slowest possible
way to find a typo.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
A1_COMPOSE = ROOT / "docker-compose.a1.yml"
LOCAL_COMPOSE = ROOT / "docker-compose.yml"
CLICKHOUSE_XML = ROOT / "deploy" / "clickhouse-low-mem.xml"
SETUP_SH = ROOT / "deploy" / "a1-setup.sh"
DEPLOY_MD = ROOT / "DEPLOY.md"

# The box is shared. This is the number the whole profile is designed around.
MEM_BUDGET_MB = 8 * 1024


def _compose(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _mem_mb(value: str) -> int:
    """`900m` / `2g` -> megabytes."""
    match = re.fullmatch(r"(\d+)([mg])", str(value).strip().lower())
    assert match, f"unparseable mem_limit: {value!r}"
    size = int(match.group(1))
    return size * 1024 if match.group(2) == "g" else size


# -- ClickHouse ------------------------------------------------------------
def test_clickhouse_config_is_well_formed_xml():
    """It has been broken before, by a `--` inside an XML comment.

    ClickHouse does not merge a file it cannot parse, so the effect is not an
    error but a silent revert to defaults sized for a machine with 11 GB.
    """
    ET.parse(CLICKHOUSE_XML)


def test_clickhouse_mutation_threshold_fits_the_background_pool():
    """Regression: ClickHouse refused to start, so Langfuse never started.

    The merges/mutations pool holds background_pool_size x concurrency_ratio
    entries. MergeTree will not run if it is told to keep more free entries in
    that pool than the pool can hold -- the server exits with BAD_ARGUMENTS
    before it listens on 8123, which reads as a healthcheck failure until you
    open the log. Shrinking the pool without shrinking the thresholds is the
    trap; this test is here so the two cannot drift apart again.
    """
    root = ET.parse(CLICKHOUSE_XML).getroot()

    def value(path: str) -> int:
        node = root.find(path)
        assert node is not None and node.text, f"{path} is not set in {CLICKHOUSE_XML.name}"
        return int(node.text)

    capacity = value("background_pool_size") * value(
        "background_merges_mutations_concurrency_ratio"
    )
    for threshold in (
        "number_of_free_entries_in_pool_to_execute_mutation",
        "number_of_free_entries_in_pool_to_lower_max_size_of_merge",
        "number_of_free_entries_in_pool_to_execute_optimize_entire_partition",
    ):
        assert value(f"merge_tree/{threshold}") <= capacity, (
            f"{threshold} exceeds the {capacity}-entry pool; ClickHouse will not boot"
        )


def test_clickhouse_server_ceiling_is_below_its_cgroup_limit():
    """A ceiling above the cgroup means it plans queries it cannot finish."""
    limit_bytes = _mem_mb(_compose(A1_COMPOSE)["services"]["clickhouse"]["mem_limit"]) * 1024 * 1024
    root = ET.parse(CLICKHOUSE_XML).getroot()
    server = int(root.findtext("max_server_memory_usage", "0"))
    query = int(root.findtext("profile/default/max_memory_usage", "0"))
    assert 0 < server < limit_bytes
    assert 0 < query <= server


# -- healthchecks ----------------------------------------------------------
@pytest.mark.parametrize("path", [A1_COMPOSE, LOCAL_COMPOSE], ids=lambda p: p.name)
def test_no_healthcheck_probes_localhost(path: Path):
    """Regression: `localhost` resolves to ::1 first on a host without IPv6.

    The container is listening on 0.0.0.0 and answering, the probe gets
    ECONNREFUSED, and the service never goes healthy -- so everything waiting on
    `condition: service_healthy` waits forever, for no visible reason.
    """
    for name, service in _compose(path)["services"].items():
        test = service.get("healthcheck", {}).get("test")
        if not test:
            continue
        text = " ".join(test) if isinstance(test, list) else str(test)
        assert "localhost" not in text, f"{name} healthcheck probes localhost; use 127.0.0.1"


# -- memory budget ---------------------------------------------------------
def test_a1_memory_limits_are_all_set_and_fit_the_budget():
    services = _compose(A1_COMPOSE)["services"]
    limits = {}
    for name, service in services.items():
        assert "mem_limit" in service, f"{name} has no mem_limit; the box is shared"
        limits[name] = _mem_mb(service["mem_limit"])
    total = sum(limits.values())
    assert total <= MEM_BUDGET_MB, f"mem_limits total {total} MB, over the {MEM_BUDGET_MB} MB budget"


def test_deploy_md_quotes_the_real_memory_total():
    """The doc claimed 7.9 GB while the file totalled 10.9 GB. Once is enough."""
    total = sum(_mem_mb(s["mem_limit"]) for s in _compose(A1_COMPOSE)["services"].values())
    assert f"{total} MB" in DEPLOY_MD.read_text(encoding="utf-8"), (
        f"DEPLOY.md does not mention the actual total of {total} MB"
    )


# -- ports -----------------------------------------------------------------
def test_every_published_port_is_loopback_and_configurable():
    """Two rules at once: nothing is exposed publicly, and nothing is hardcoded.

    The box's ingress is TCP 22 only, and it is shared -- a neighbour already
    owns 3000 there, so our host ports have to be movable without editing a
    tracked file.
    """
    for name, service in _compose(A1_COMPOSE)["services"].items():
        for mapping in service.get("ports", []):
            assert str(mapping).startswith("127.0.0.1:"), f"{name} publishes {mapping} beyond loopback"
            host_port = str(mapping).split(":")[1]
            assert host_port.startswith("${ARC_PORT_"), (
                f"{name} hardcodes host port {host_port}; make it an ARC_PORT_* variable"
            )


# -- setup script ----------------------------------------------------------
def test_setup_script_does_not_try_to_pull_a_built_image():
    """Regression: `docker compose pull` aborted on the buildable `ui` service,
    so nothing was pulled and setup stopped before it started anything."""
    script = SETUP_SH.read_text(encoding="utf-8")
    buildable = [n for n, s in _compose(A1_COMPOSE)["services"].items() if "build" in s]
    assert buildable, "expected at least one buildable service in the A1 profile"
    assert "--ignore-buildable" in script
    assert "build ui" in script, "the buildable service must be built explicitly"
