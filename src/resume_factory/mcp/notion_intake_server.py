from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

from mcp.server.fastmcp import FastMCP
from notion_client import Client

from resume_factory.security import safe_path

mcp = FastMCP("notion-intake-readonly")


def _snapshot_root() -> Path:
    configured = os.getenv("RF_NOTION_SNAPSHOT_DIR")
    if not configured:
        raise RuntimeError(
            "RF_NOTION_SNAPSHOT_DIR is required. Export a private snapshot or configure a "
            "read-only Notion integration."
        )
    return Path(configured).expanduser().resolve()


def _application_path(application_id: str) -> Path:
    if not application_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("application_id contains unsupported characters")
    return safe_path(_snapshot_root(), _snapshot_root() / f"{application_id}.json")


def _load(application_id: str) -> dict[str, Any]:
    if os.getenv("RF_NOTION_SNAPSHOT_DIR"):
        path = _application_path(application_id)
        if not path.exists():
            return {"error": "not_found", "application_id": application_id}
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    token = os.getenv("RF_NOTION_TOKEN")
    if token:
        return _load_from_notion(application_id, token)
    raise RuntimeError("configure RF_NOTION_SNAPSHOT_DIR or RF_NOTION_TOKEN")


def _load_from_notion(application_id: str, token: str) -> dict[str, Any]:
    """Normalize a read-only Notion page without persisting its private content."""
    client = Client(auth=token)
    page = cast(dict[str, Any], client.pages.retrieve(page_id=application_id))
    block_response = cast(
        dict[str, Any], client.blocks.children.list(block_id=application_id, page_size=100)
    )
    blocks = cast(list[dict[str, Any]], block_response.get("results", []))
    properties = page.get("properties", {})
    return {
        "application_id": application_id,
        "company": _first_property(properties, ("회사", "기업명", "Company")),
        "job": _first_property(properties, ("직무", "지원부서", "Job")),
        "job_description": _first_property(properties, ("직무기술서", "JD")),
        "company_context": _first_property(properties, ("기업분석", "회사분석")),
        "questions": [],
        "existing_draft": _blocks_to_text(blocks),
        "source": "notion-readonly-api",
    }


def _first_property(properties: dict[str, Any], names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in properties:
            value = _property_text(properties[name])
            if value:
                return value
    return None


def _property_text(prop: dict[str, Any]) -> str | None:
    prop_type = prop.get("type")
    if not isinstance(prop_type, str):
        return None
    value = prop.get(prop_type, {})
    if prop_type in {"title", "rich_text"}:
        return "".join(item.get("plain_text", "") for item in value)
    if prop_type in {"select", "status"}:
        selected = (value or {}).get("name")
        return str(selected) if selected is not None else None
    if prop_type == "url":
        return str(value) if value is not None else None
    if prop_type == "number":
        return str(value) if value is not None else None
    return None


def _blocks_to_text(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for block in blocks:
        block_type = block.get("type")
        if not isinstance(block_type, str):
            continue
        content = block.get(block_type, {})
        rich_text = content.get("rich_text", []) if isinstance(content, dict) else []
        text = "".join(item.get("plain_text", "") for item in rich_text).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


@mcp.tool()
def get_application(application_id: str) -> dict[str, Any]:
    """Return a private application snapshot without mutating Notion."""
    return _load(application_id)


@mcp.tool()
def get_job_description(application_id: str) -> dict[str, Any]:
    """Return company, job, and JD fields from a snapshot."""
    record = _load(application_id)
    if "error" in record:
        return record
    return {
        "company": record.get("company"),
        "job": record.get("job"),
        "job_description": record.get("job_description"),
        "company_context": record.get("company_context"),
    }


@mcp.tool()
def get_application_questions(application_id: str) -> dict[str, Any]:
    """Return application questions and character limits."""
    record = _load(application_id)
    if "error" in record:
        return record
    return {"questions": record.get("questions", [])}


@mcp.tool()
def get_existing_draft(application_id: str) -> dict[str, Any]:
    """Return an existing draft as read-only context."""
    record = _load(application_id)
    if "error" in record:
        return record
    return {"existing_draft": record.get("existing_draft")}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
