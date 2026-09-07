from __future__ import annotations

import asyncio
import sys
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient


def server_configs() -> dict[str, dict[str, Any]]:
    # Do not resolve the venv symlink: doing so escapes the environment and loses packages.
    executable = sys.executable
    return {
        "dual_brain": {
            "command": executable,
            "args": ["-m", "resume_factory.mcp.dual_brain_server"],
            "transport": "stdio",
        },
        "notion_intake": {
            "command": executable,
            "args": ["-m", "resume_factory.mcp.notion_intake_server"],
            "transport": "stdio",
        },
    }


async def load_tools_isolated(
    timeout_seconds: float = 10,
    configs: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[Any], dict[str, str]]:
    tools: list[Any] = []
    health: dict[str, str] = {}
    for server_name, config in (configs or server_configs()).items():
        try:
            client = MultiServerMCPClient({server_name: config}, tool_name_prefix=True)
            server_tools = await asyncio.wait_for(client.get_tools(), timeout_seconds)
            tools.extend(server_tools)
            health[server_name] = "healthy"
        except Exception as exc:  # server isolation is the contract
            health[server_name] = f"unavailable:{type(exc).__name__}"
    return tools, health
