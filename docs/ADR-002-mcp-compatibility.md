# ADR-002: MCP v1 compatibility lane

- Status: Accepted
- Date: 2026-09-07

## Decision

첫 릴리스는 `langchain-mcp-adapters==0.3.2`와 `mcp>=1.28,<2`를 함께 고정한다. MCP 2로의
변경은 LangChain 정식 지원, MCP 계약 테스트, 골든 품질 회귀 테스트가 모두 통과할 때만
진행한다.

## Consequences

- 현재 안정형 LangChain 어댑터와의 import 호환성을 보장한다.
- 최신 MCP SDK 기능은 업그레이드 게이트가 열릴 때까지 사용하지 않는다.

