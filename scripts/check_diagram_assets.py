"""Validate committed diagram deliverables without platform-specific PNG byte checks."""

from __future__ import annotations

import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEMS = ("vertical-ai-overview", "multi-agent-team-flow")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    if len(data) != 24 or data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    return struct.unpack(">II", data[16:24])


def main() -> None:
    architecture = ROOT / "docs" / "architecture"
    for stem in STEMS:
        source = architecture / f"{stem}.mmd"
        svg = architecture / f"{stem}.svg"
        png = architecture / f"{stem}.png"
        for path in (source, svg, png):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty diagram asset: {path}")
        if "<svg" not in svg.read_text(encoding="utf-8"):
            raise ValueError(f"Invalid SVG document: {svg}")
        width, height = png_dimensions(png)
        if width != 1920 or height <= 0:
            raise ValueError(f"PNG must be 1920px wide: {png}")
    print("diagram assets: ok")


if __name__ == "__main__":
    main()
