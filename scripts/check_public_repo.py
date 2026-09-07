from __future__ import annotations

import re
import subprocess
from pathlib import Path

DENY = [
    re.compile(r"/Users/[^/]+/"),
    re.compile(r"(?:sk|ntn|secret)_[A-Za-z0-9_-]{16,}"),
    re.compile(r"3cee8df8-550e-80c8-81ff-c205f9f068c3"),
]


def main() -> None:
    root = Path(__file__).parents[1]
    tracked = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
    violations: list[str] = []
    for relative in tracked:
        path = root / relative
        if not path.is_file() or path.suffix.lower() in {".png", ".svg"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in DENY):
            violations.append(relative)
    if violations:
        raise SystemExit(f"public-repository privacy violation: {violations}")


if __name__ == "__main__":
    main()
