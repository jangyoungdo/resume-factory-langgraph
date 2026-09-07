import json
import sys
from pathlib import Path

from resume_factory.mcp import dual_brain_server, notion_intake_server
from resume_factory.mcp.client import load_tools_isolated, server_configs


def test_dual_brain_curated_loader(monkeypatch, tmp_path: Path) -> None:
    private = tmp_path / ".resume_factory"
    private.mkdir()
    (private / "evidence.jsonl").write_text(
        json.dumps({"event_id": "SYNTH-01", "capabilities": ["analysis"]}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RF_DUAL_BRAIN_ROOT", str(tmp_path))
    assert dual_brain_server._load_records()[0]["event_id"] == "SYNTH-01"


def test_notion_snapshot_loader_rejects_traversal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RF_NOTION_SNAPSHOT_DIR", str(tmp_path))
    try:
        notion_intake_server._application_path("../secret")
    except ValueError:
        pass
    else:
        raise AssertionError("path traversal must be rejected")


def test_mcp_servers_keep_active_virtualenv_python() -> None:
    configs = server_configs()
    assert configs["dual_brain"]["command"] == sys.executable
    assert configs["notion_intake"]["command"] == sys.executable


def test_notion_property_normalization() -> None:
    properties = {
        "회사": {"type": "title", "title": [{"plain_text": "Sample Steel"}]},
        "직무": {"type": "select", "select": {"name": "Maintenance Engineer"}},
    }
    assert notion_intake_server._first_property(properties, ("회사",)) == "Sample Steel"
    assert (
        notion_intake_server._first_property(properties, ("직무",))
        == "Maintenance Engineer"
    )


async def test_one_mcp_failure_does_not_remove_healthy_tools(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RF_DUAL_BRAIN_ROOT", str(tmp_path))
    configs = {
        "dual_brain": server_configs()["dual_brain"],
        "broken": {
            "command": "/definitely/missing/python",
            "args": [],
            "transport": "stdio",
        },
    }
    tools, health = await load_tools_isolated(timeout_seconds=3, configs=configs)
    assert health["dual_brain"] == "healthy"
    assert health["broken"].startswith("unavailable:")
    assert any(tool.name == "dual_brain_search_evidence" for tool in tools)
