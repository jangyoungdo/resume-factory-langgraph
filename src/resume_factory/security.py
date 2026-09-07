from __future__ import annotations

import hashlib
import re
from pathlib import Path

SECRET_PATTERNS = (
    re.compile(r"(?:sk|ntn|secret)_[A-Za-z0-9_-]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{16,}", re.IGNORECASE),
)


def redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def safe_path(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise PermissionError(f"path escapes configured root: {resolved}")
    return resolved


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
