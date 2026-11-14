#!/usr/bin/env python3
"""Step 0: check that everything the pipeline needs is in place.

Usage: uv run scripts/check_env.py

Checks ffmpeg/ffprobe (with the libass `ass` filter), the background image, the
caption font, and OPENAI_API_KEY (environment or <repo>/.env). Prints only
whether the key is set and where it came from, never the key itself.
Exit code 0 = ready, 1 = something is missing.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import config as C
from common import load_env
from render import font_available


def main() -> None:
    ok = True

    def report(good: bool, label: str, detail: str, required: bool = True) -> None:
        nonlocal ok
        mark = "ok  " if good else ("FAIL" if required else "warn")
        print(f"{mark}  {label:<16} {detail}")
        ok = ok and (good or not required)

    for tool in ("ffmpeg", "ffprobe"):
        report(shutil.which(tool) is not None, tool, shutil.which(tool) or "not on PATH (README: Setup)")
    if shutil.which("ffmpeg"):
        filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
        has_ass = any(line.split()[1:2] == ["ass"] for line in filters.splitlines() if line.strip())
        report(has_ass, "libass", "ffmpeg has the `ass` filter" if has_ass else "ffmpeg built without libass")

    report(C.BACKGROUND_IMAGE.is_file(), "background", str(C.BACKGROUND_IMAGE))
    report(font_available(), "font", C.FONT_NAME if font_available() else
           f"{C.FONT_NAME} not found; captions will use a fallback font", required=False)

    source = load_env("OPENAI_API_KEY")
    if source:
        report(True, "OPENAI_API_KEY", f"set (from {source if source == 'environment' else C.ENV_FILE})")
    else:
        hint = (f"{C.ENV_FILE} exists but has no value" if C.ENV_FILE.is_file()
                else f"create {C.ENV_FILE}: cp .env.example .env, then fill it in")
        report(False, "OPENAI_API_KEY", f"not set; {hint}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
