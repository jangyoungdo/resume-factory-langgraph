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
rf execute --application-id <private-id> --backend codex --mode balanced --output json
rf install-codex-skill
```

## 글자 수 품질 게이트

제출 문자열은 `[소제목]\n본문`으로 정규화하며 줄바꿈 한 글자를 포함합니다. 문항 제한의
95% 미만 또는 100% 초과만 재작성 대상입니다. 97~98%는 생성 목표이지만 Hard Gate 안의
문항을 다시 쓰는 조건은 아닙니다. 여러 문항이 Hard Gate를 벗어나도 Terra의 배치 호출
한 번으로만 복구하며, 기존 evidence ID와 문장 역할을 유지합니다.

## CLI

```text
rf doctor --backend codex
rf execute --application-id <private-id> --backend codex --mode balanced --output json
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
rf performance show <run-id> --timeline
rf performance summary --limit 10
rf feedback <run-id> --decision revised --rating 4 --final-draft <private-final.json>
rf eval --suite golden
rf diagram
rf install-codex-skill
```

v0.7은 회사별 근거 ID와 별도로 불변 `experience_key`를 사용합니다. 모든 문항의 주 소재를
전역 배정해 중복을 막고, 소재가 부족하면 모델 호출 전에 중단합니다. 배움형 문항만 하나의
보조 경험을 1~2문장으로 사용할 수 있습니다.

3문항 balanced 실행은 공통 분석·소재팀 9회, 작성위원회 9회, 통합 편집 1회, 문항별 독자
비평 3회로 기본 22회입니다. 결함 문장만 최대 2회 보정하며 전체 호출 상한은 30회입니다.
최종 문장은 `DraftProposal.sentence_plans`에서 조립되고 국소 보정도 문장 ID 계보를
유지합니다.

`rf execute`에는 Git·push·PR·CI가 없으며 비공개 산출물 생성까지 한 프로세스에서 끝납니다.
코드 버전 관리는 기능 릴리스 때만 별도 수행합니다.

토큰·비용·사용자 수정률 모니터링은 `docs/LLMOPS.md`, 아키텍처와 운영 결정은 `docs/`를
참고하십시오.
