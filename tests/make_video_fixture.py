#!/usr/bin/env python3
"""Build a synthetic StreamYard-like recording for the stacked-layout E2E.

Usage: uv run tests/make_video_fixture.py OUT.mp4

1280x720, 30 fps, the sample fixture's audio. Two coloured speaker tiles sit
where config.STACK_TILES expects them (teal = left/top panel, orange =
right/bottom panel) on a dark grey canvas, and a white box flashes in the
centre of the left tile for 0.2s at the start of every ground-truth phrase, so
tests/check_sync.py --flashes can verify picture/sound sync after re-framing
and hard cuts. Deterministic and quick (a few seconds), so it is generated at
test time instead of being committed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "fixtures" / "sample"
sys.path.insert(0, str(ROOT / "scripts"))
import config as C  # noqa: E402

W, H, FPS = 1280, 720, 30
TILE_COLORS = ("0x008080", "0xE6781E")  # teal, orange
FLASH_SECONDS = 0.2
FLASH_SIZE = 160


def build(out: Path) -> None:
    truth = json.loads((SAMPLE / "ground_truth.json").read_text(encoding="utf-8"))
    boxes = []
    for t, color in zip(C.STACK_TILES, TILE_COLORS):
        boxes.append(f"drawbox=x={round(t['x'] * W)}:y={round(t['y'] * H)}:w={round(t['w'] * W)}:"
                     f"h={round(t['h'] * H)}:color={color}:t=fill")
    t0 = C.STACK_TILES[0]
    cx, cy = round((t0["x"] + t0["w"] / 2) * W), round((t0["y"] + t0["h"] / 2) * H)
    enable = "+".join(f"between(t,{p['start']:.3f},{p['start'] + FLASH_SECONDS:.3f})" for p in truth["phrases"])
    boxes.append(f"drawbox=x={cx - FLASH_SIZE // 2}:y={cy - FLASH_SIZE // 2}:w={FLASH_SIZE}:h={FLASH_SIZE}:"
                 f"color=white:t=fill:enable='{enable}'")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-v", "error",
                    "-f", "lavfi", "-i", f"color=c=0x202020:s={W}x{H}:r={FPS}:d={truth['duration']}",
                    "-i", str(SAMPLE / "input.mp4"),
                    "-filter_complex", f"[0:v]{','.join(boxes)},format=yuv420p[v]",
                    "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                    "-g", str(2 * FPS), "-c:a", "aac", "-b:a", "128k", "-shortest", str(out)], check=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    build(Path(sys.argv[1]))
