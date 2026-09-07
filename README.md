# Resume Factory LangGraph

회사·직무의 채용 수요와 검증된 경험을 연결해 근거 기반 자기소개서 초안을 만드는
LangGraph 멀티에이전트 워크플로입니다.

## 안전 경계

- 실제 지원서 제출과 최종본 덮어쓰기는 지원하지 않습니다.
- MCP 서버는 읽기 전용 도메인 도구만 노출합니다.
- 실제 이력서, 자기소개서, Notion 데이터와 실행 로그는 Git에서 제외합니다.
- 저장소의 fixture는 모두 가상의 회사와 합성 경험입니다.

## 빠른 시작

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
rf doctor
rf run --input tests/fixtures/sample_application.json --offline
```

API 키 없이 `--offline`으로 회귀 테스트를 재현할 수 있습니다. 기본 실제 실행은 ChatGPT로
로그인한 `codex exec`를 사용하며 API 키를 자식 프로세스에서 제거합니다. 구독 실행의 토큰은
기록하지만 달러 비용은 `N/A`입니다.

```bash
codex login
rf doctor --backend codex
rf run --application-id <private-id> --backend codex --mode balanced
rf deliver <run-id>
rf install-codex-skill
```

## 글자 수 품질 게이트

제출 문자열은 `[소제목]\n본문`으로 정규화하며 줄바꿈 한 글자를 포함합니다. 문항 제한의
95% 미만 또는 100% 초과는 hard fail이고, 생성기는 97~98%를 목표로 한 번만 선택적으로
보강합니다. 보강 문장도 기존 evidence ID와 문장 역할을 유지하며 일반론으로 분량을 채우지
않습니다.

## CLI

```text
rf doctor --backend codex
rf index --source <private-evidence-directory>
rf run --application-id <private-id> --backend codex --mode balanced
rf run --input <private-application-snapshot.json> --backend offline
rf resume <run-id>  # .local/checkpoints.sqlite의 중단 노드부터 재개
rf deliver <run-id>
rf review <run-id>
rf render-draft --input <private-submission.json> --output <new-private-draft.md>
rf usage list
rf usage show <run-id> --group-by question
rf usage compare <run-a> <run-b> --group-by model
rf feedback <run-id> --decision revised --rating 4 --final-draft <private-final.json>
rf eval --suite golden
rf diagram
rf install-codex-skill
```

v0.5의 3문항 balanced 실행은 회사·직무 3회, 질문 전략 3회, 경험·브랜딩 3회,
문항별 작성위원회 12회, 통합 편집 1회로 기본 22회입니다. 글자 수 재작성과 Sol 판정은
합계 6회 이내이며 전체 상한은 28회입니다. 최종 문장은 작성위원회의
`DraftProposal.sentence_plans`에서 직접 조립되어 근거 계보를 유지합니다.

토큰·비용·사용자 수정률 모니터링은 `docs/LLMOPS.md`, 아키텍처와 운영 결정은 `docs/`를
참고하십시오.
