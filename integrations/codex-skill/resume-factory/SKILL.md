---
name: resume-factory
description: Run the local Resume Factory LangGraph pipeline for evidence-grounded cover letters. Use whenever the user asks to create, revise, resume, deliver, or inspect usage for a job application with Resume Factory, including Korean requests naming a company and role.
---

# Resume Factory

Run the installed `rf` CLI; do not imitate the pipeline by writing a draft directly.

## Run

1. Identify the requested private application ID. If the user names LG에너지솔루션 설비기술 FA, use `lges-2026h2-fa-cheongju` unless they specify another ID.
2. Run `rf doctor --backend codex`. Stop and report the exact failed check if ChatGPT authentication or either required MCP server is unavailable.
3. Run `rf run --application-id <id> --backend codex --mode balanced`.
4. Read the saved run JSON and verify:
   - `telemetry.provider` is `codex_cli`;
   - `telemetry.graph_version` is `v0.5`;
   - all seven graph nodes are present in `graph_nodes_completed`;
   - Codex model calls are between the base budget and hard cap;
   - final validation status and per-question character counts are explicit.
5. Run `rf deliver <run-id>` only after the graph finishes. Never submit an application.
6. Open the generated Markdown in the Codex app and summarize QA, call count, token use, and character counts.

If the CLI fails, do not silently create a replacement cover letter. Preserve the failed run evidence and explain the blocker.

## Privacy

Never print or commit the private intake snapshot, evidence ledger, run JSON, or deliverable contents into the public repository. Use only the read-only domain MCP tools exposed by Resume Factory.
