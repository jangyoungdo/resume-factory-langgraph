from __future__ import annotations

import re

LOW_VALUE_CAVEAT_RE = re.compile(
    r"일 뿐|에 불과하|보장(?:한|하는|된)? 값이 아니|측정하지 못(?:했|한)|"
    r"실제 .{0,24}(?:경험이 아니|수행한 .*아니)|오해할 수"
)
SCOPE_QUALIFIER_RE = re.compile(
    r"교육용|가상 (?:생산라인|공정|환경)|프로젝트 기반|\bPoC\b|실제 양산"
)
VAGUE_RESULT_RE = re.compile(r"좋은 결과|긍정적인 결과|좋은 성과|성과를 얻(?:었|었습니다)")


def has_low_value_caveat(text: str) -> bool:
    return LOW_VALUE_CAVEAT_RE.search(text) is not None


def has_scope_qualifier(text: str) -> bool:
    return SCOPE_QUALIFIER_RE.search(text) is not None


def has_vague_result(text: str) -> bool:
    return VAGUE_RESULT_RE.search(text) is not None
