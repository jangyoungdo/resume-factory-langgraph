# ADR-001: Public code, private career data

- Status: Accepted
- Date: 2026-09-07

## Decision

코드와 합성 fixture는 공개하고 개인 이력, 회사별 지원서, Notion snapshot, 실행 결과,
SQLite와 MLflow 데이터는 로컬에만 둔다. Dual Brain과 Notion은 읽기 전용 MCP 계약으로
연결하며 범용 파일시스템 MCP는 사용하지 않는다.

## Consequences

- 공개 CI가 실제 개인정보 없이 전체 워크플로를 검증할 수 있다.
- 로컬 데이터 구조가 바뀌어도 MCP 계약을 유지하면 그래프는 영향을 받지 않는다.
- 실제 데이터 기반 품질 평가는 공개 회귀 테스트와 별도로 실행해야 한다.

