from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


def server_configs() -> dict[str, dict[str, Any]]:
    # Do not resolve the venv symlink: doing so escapes the environment and loses packages.
    executable = sys.executable
    environment = {
        key: value
        for key in ("RF_DUAL_BRAIN_ROOT", "RF_NOTION_SNAPSHOT_DIR", "RF_NOTION_TOKEN")
        if (value := os.getenv(key))
    }
    return {
        "dual_brain": {
            "command": executable,
            "args": ["-m", "resume_factory.mcp.dual_brain_server"],
            "transport": "stdio",
            "env": environment,
        },
        "notion_intake": {
            "command": executable,
            "args": ["-m", "resume_factory.mcp.notion_intake_server"],
            "transport": "stdio",
            "env": environment,
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
            client = MultiServerMCPClient(cast(Any, {server_name: config}), tool_name_prefix=True)
            server_tools = await asyncio.wait_for(client.get_tools(), timeout_seconds)
            tools.extend(server_tools)
            health[server_name] = "healthy"
        except Exception as exc:  # server isolation is the contract
            health[server_name] = f"unavailable:{type(exc).__name__}"
    return tools, health


@asynccontextmanager
async def open_tools_persistent(
    timeout_seconds: float = 10,
    configs: dict[str, dict[str, Any]] | None = None,
    startup_metrics: list[dict[str, Any]] | None = None,
) -> AsyncIterator[tuple[list[Any], dict[str, str]]]:
    """Keep one initialized stdio session per MCP server for the whole intake."""
    resolved = configs or server_configs()
    client = MultiServerMCPClient(cast(Any, resolved), tool_name_prefix=True)
    managers = {name: client.session(name) for name in resolved}
    async def enter(name: str, manager: Any) -> Any:
        started = time.perf_counter()
        try:
            return await asyncio.wait_for(manager.__aenter__(), timeout_seconds)
        finally:
            if startup_metrics is not None:
                startup_metrics.append(
                    {"server": name, "duration_ms": int((time.perf_counter() - started) * 1000)}
                )

    entered = await asyncio.gather(
        *(enter(name, manager) for name, manager in managers.items()),
        return_exceptions=True,
    )
    sessions: dict[str, Any] = {}
    health: dict[str, str] = {}
    for (name, _), result in zip(managers.items(), entered, strict=True):
        if isinstance(result, BaseException):
            health[name] = f"unavailable:{type(result).__name__}"
        else:
            sessions[name] = result
            health[name] = "healthy"
    try:
        loaded = await asyncio.gather(
            *(
                load_mcp_tools(session, server_name=name, tool_name_prefix=True)
                for name, session in sessions.items()
            )
        )
        yield [tool for group in loaded for tool in group], health
    finally:
        await asyncio.gather(
            *(
                managers[name].__aexit__(None, None, None)
                for name in sessions
            ),
            return_exceptions=True,
        )
