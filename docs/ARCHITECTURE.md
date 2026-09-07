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

## Human editorial policy

- 교육용·가상·프로젝트 등 사실 범위는 사건을 처음 소개할 때 긍정형 맥락으로 한 번만 쓴다.
- 미측정 값과 수행 경계는 근거 원장과 `interview_defense`에 계속 보존한다. 다만 이미 정확한
  맥락을 밝힌 뒤 독자에게 새 정보를 주지 않는 보험 문장은 제출 본문에서 제거한다.
- 판단과 관점은 구체적인 기술 구조, 입출력, 제어 조건 또는 검증 행동으로 이어져야 하며,
  뒤에는 관찰 가능한 결과가 있어야 한다. `좋은 결과` 같은 자체 평가는 통과하지 않는다.
- 배움을 묻는 문항은 섹션명이나 영문 라벨을 흉내 내지 않는다. 선택적으로 승인된 보조
  evidence를 사용해 후행 프로젝트에서 같은 판단 방식을 재적용한 행동과 결과를 1~2문장으로
  증명한다. 다른 문항의 기술 설명은 반복하지 않는다.
- 위반은 `LOW_VALUE_DEFENSIVE_CAVEAT`, `REPEATED_SCOPE_QUALIFIER`, `VAGUE_RESULT`,
  `NO_ACTION_RESULT_CHAIN`, `LEARNING_TRANSFER_MISSING` 품질 코드로 기록한다.
