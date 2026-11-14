"""Shared helpers: episode paths, JSON IO, freshness checks, ffmpeg, text alignment."""

from __future__ import annotations

import difflib
import json
import os
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable


class StepError(RuntimeError):
    """A user-facing failure: printed without a traceback."""


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def load_env(key: str = "OPENAI_API_KEY") -> str | None:
    """Make `key` available from the environment or <repo>/.env.

    Returns where it came from ("environment" or ".env"), or None if unset.
    Never returns or prints the value itself.
    """
    if os.environ.get(key):
        return "environment"
    from config import ENV_FILE
    if ENV_FILE.is_file():
        from dotenv import load_dotenv  # lazy: only scripts that need secrets import it
        load_dotenv(ENV_FILE, override=False)
        if os.environ.get(key):
            return ".env"
    return None


def episode_dir(arg: str) -> Path:
    path = Path(arg).expanduser().resolve()
    if not path.is_dir():
        die(f"episode directory not found: {path}")
    return path


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """Atomic write so an interrupted step never leaves a half-written output."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def is_fresh(output: Path, *inputs: Path) -> bool:
    """True if `output` exists and is at least as new as every existing input."""
    if not output.exists():
        return False
    out_mtime = output.stat().st_mtime
    return all(not p.exists() or p.stat().st_mtime <= out_mtime for p in inputs)


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        die(f"`{name}` not found on PATH (see README: Setup)")


def run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-25:])
        raise StepError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc


def media_duration(path: Path) -> float:
    proc = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(proc.stdout.strip())


def detect_silences(path: Path, noise_db: float, min_dur: float) -> list[tuple[float, float]]:
    """(start, end) of every silence in a media file's audio, via ffmpeg silencedetect."""
    import re
    proc = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn",
                "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}", "-f", "null", "-"])
    out, open_at = [], None
    for kind, val in re.findall(r"silence_(start|end): (-?[\d.]+)", proc.stderr):
        if kind == "start":
            open_at = max(0.0, float(val))
        elif open_at is not None:
            out.append((open_at, float(val)))
            open_at = None
    if open_at is not None:
        out.append((open_at, media_duration(path)))
    return out


def fmt_ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}" if h >= 1 else f"{int(m):02d}:{s:05.2f}"


# --- Text helpers -----------------------------------------------------------------

def char_width(c: str) -> float:
    """Japanese caption width: full-width = 1, half-width = 0.5."""
    return 1.0 if unicodedata.east_asian_width(c) in ("W", "F", "A") else 0.5


def text_width(s: str) -> float:
    return sum(char_width(c) for c in s)


def _norm(c: str) -> str:
    n = unicodedata.normalize("NFKC", c).casefold()
    return n[:1] if n else c


def clip_spans(clip: dict) -> list[dict]:
    """The clip's spans in output order. Accepts v2 ({"spans": [...]}) and the
    original single-span v1 (start/end/segments at the top level)."""
    if "spans" in clip:
        return clip["spans"]
    return [{k: clip[k] for k in ("start", "end", "segments", "title", "reason") if k in clip}]


def clip_duration(clip: dict) -> float:
    return sum(s["end"] - s["start"] for s in clip_spans(clip))


def flatten_words(transcript: dict) -> list[dict]:
    """All words in order, each tagged with its segment id and index in that segment."""
    out = []
    for seg in transcript["segments"]:
        for i, w in enumerate(seg.get("words", [])):
            out.append({**w, "segment_id": seg["id"], "index": i})
    return out


def align_text(text: str, words: list[dict], seg_start: float, seg_end: float):
    """Map every character of (possibly corrected) `text` to a (start, end) time.

    Word timestamps belong to the raw recognition; the corrected text can differ.
    We spread each word's time span over its characters, then diff the raw word
    string against `text` and carry times across matching / replaced runs.

    Returns (char_times, word_offsets): word_offsets[k] is the character offset
    in `text` where words[k] begins (good caption breaks / trim points).
    """
    n = len(text)
    if n == 0:
        return [], [0] * len(words)

    wchars: list[str] = []
    wtimes: list[tuple[float, float]] = []
    wstarts: list[int] = []  # W offset of each word (whitespace-only words share the next)
    for w in words:
        wstarts.append(len(wchars))
        chars = [c for c in w["word"] if not c.isspace()]
        if not chars:
            continue
        s, e = float(w["start"]), float(w["end"])
        if e < s:
            e = s
        for k, c in enumerate(chars):
            wchars.append(c)
            wtimes.append((s + (e - s) * k / len(chars), s + (e - s) * (k + 1) / len(chars)))

    if not wchars:
        step = (seg_end - seg_start) / n
        return [(seg_start + i * step, seg_start + (i + 1) * step) for i in range(n)], [0] * len(words)

    times: list[tuple[float, float] | None] = [None] * n
    w2t = [0] * len(wchars)
    sm = difflib.SequenceMatcher(None, [_norm(c) for c in wchars], [_norm(c) for c in text], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                times[j1 + k] = wtimes[i1 + k]
                w2t[i1 + k] = j1 + k
        elif tag == "replace":
            s, e = wtimes[i1][0], wtimes[i2 - 1][1]
            m = j2 - j1
            for k in range(m):
                times[j1 + k] = (s + (e - s) * k / m, s + (e - s) * (k + 1) / m)
            for i in range(i1, i2):
                w2t[i] = j1 + (i - i1) * m // (i2 - i1)
        elif tag == "delete":
            for i in range(i1, i2):
                w2t[i] = min(j1, n - 1)
        # "insert": text chars with no audio counterpart (e.g. punctuation) — filled below

    # Inserted chars take a zero-length slot at the end of the previous char.
    last: tuple[float, float] | None = None
    for i in range(n):
        if times[i] is None:
            if last is not None:
                times[i] = (last[1], last[1])
        else:
            last = times[i]
    nxt: tuple[float, float] | None = None
    for i in range(n - 1, -1, -1):
        if times[i] is None:
            times[i] = (nxt[0], nxt[0]) if nxt else (seg_start, seg_start)
        else:
            nxt = times[i]

    # Enforce monotonic starts (diffs can produce small inversions).
    out: list[tuple[float, float]] = []
    prev_s = -1.0
    for s, e in times:  # type: ignore[misc]
        s = max(s, prev_s)
        out.append((s, max(e, s)))
        prev_s = s
    w2t.append(n)  # a trailing whitespace-only word maps to the end of the text
    return out, [w2t[p] for p in wstarts]


def iter_md_table_rows(md: str) -> Iterable[list[str]]:
    """Yield cell lists for each data row of every pipe table in a markdown string."""
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells, buf, esc = [], "", False
        for ch in line[1:]:
            if esc:
                buf += ch
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "|":
                cells.append(buf.strip())
                buf = ""
            else:
                buf += ch
        if buf.strip():
            cells.append(buf.strip())
        if all(set(c) <= set("-: ") for c in cells):
            continue  # separator row
        yield cells
