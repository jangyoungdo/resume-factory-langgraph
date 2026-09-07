# Architecture

## Boundary

Resume Factory는 초안과 검토 자료만 만든다. 최종 제출, 사용자가 승인한 원문 수정, 범용
파일 쓰기 도구는 제공하지 않는다. 실제 데이터는 공개 저장소 밖에 있고 MCP가 구조화된
읽기 결과만 제공한다.

## Multi-agent contract

각 팀은 독립 제안, 익명 비평, 팀장 결정, 신뢰도 게이트 순서로 실행한다. 에이전트는
`AgentProposal`을 반환하고 팀 사이에는 `TeamDecision`과 도메인 스키마만 전달한다.
사실 충돌은 투표하지 않고 `EvidencePacket` 원본으로 판정한다.

## Model routing

- Local/BGE-M3: 검색과 결정론 검사
- Luna: 후보 생성과 짧은 비평
- Terra: 분석, 선택, 통합
- Sol: 낮은 신뢰도 또는 판단 충돌의 1회 심판

정적 규칙이 먼저 실패를 찾고, 실패한 블록만 재실행한다.

## Data lineage

모든 경험은 불변 `event_id`, 출처 hash, 수치 권위 ID, 수행 범위, 금지 조합을 가진다.
모든 자소서 문장은 역할, selling point, 근거 ID, claim ID, 회사 연결과 면접 방어 가능성을
기록한다.

