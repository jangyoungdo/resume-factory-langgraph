import json
import sys
from pathlib import Path
from types import SimpleNamespace

from resume_factory.mcp import dual_brain_server, notion_intake_server
from resume_factory.mcp.client import load_tools_isolated, open_tools_persistent, server_configs


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
    assert notion_intake_server._first_property(properties, ("직무",)) == "Maintenance Engineer"


async def test_one_mcp_failure_does_not_remove_healthy_tools(monkeypatch, tmp_path: Path) -> None:
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


async def test_persistent_mcp_opens_each_server_once(monkeypatch) -> None:
    counts = {"a": [0, 0], "b": [0, 0]}

    class SessionManager:
        def __init__(self, name: str) -> None:
            self.name = name

        async def __aenter__(self):  # type: ignore[no-untyped-def]
            counts[self.name][0] += 1
            return SimpleNamespace(name=self.name)

        async def __aexit__(self, *args):  # type: ignore[no-untyped-def]
            counts[self.name][1] += 1

    class FakeClient:
        def __init__(self, connections, **kwargs):  # type: ignore[no-untyped-def]
            self.connections = connections

        def session(self, name: str) -> SessionManager:
            return SessionManager(name)

    async def fake_load(session, **kwargs):  # type: ignore[no-untyped-def]
        return [SimpleNamespace(name=f"{kwargs['server_name']}_tool")]

    monkeypatch.setattr("resume_factory.mcp.client.MultiServerMCPClient", FakeClient)
    monkeypatch.setattr("resume_factory.mcp.client.load_mcp_tools", fake_load)
    configs = {
        "a": {"command": "a", "transport": "stdio"},
        "b": {"command": "b", "transport": "stdio"},
    }
    async with open_tools_persistent(configs=configs) as (tools, health):
        assert len(tools) == 2
        assert health == {"a": "healthy", "b": "healthy"}
    assert counts == {"a": [1, 1], "b": [1, 1]}
