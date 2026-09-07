from pathlib import Path

import pytest

from resume_factory.security import redact, safe_path


def test_safe_path_blocks_escape(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        safe_path(tmp_path, tmp_path / ".." / "private.txt")


def test_redact_removes_tokens() -> None:
    assert "secret_" not in redact("Authorization secret_1234567890abcdef")

