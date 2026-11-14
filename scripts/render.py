#!/usr/bin/env python3
"""Step 3: clip.json -> digest.ass + digest.mp4 (1080x1920, 30 fps).

Usage: uv run scripts/render.py episodes/<slug> [--force] [--audio-from-video]

- Spans: every span in clip.json is cut from the source and the spans are joined
  in order with hard cuts (tiny audio fades at each cut). Each span's length is
  snapped to whole frames so picture and sound stay in sync.
- Layout (episode.json "layout", default config.DEFAULT_LAYOUT):
  background: assets/background.png ("cover" or "contain", config.BACKGROUND_FIT)
              + audio from audio.mp3 (--audio-from-video: input.mp4).
  stacked:    the two speaker tiles of input.mp4 (episode.json "tiles", default
              config.STACK_TILES) top and bottom, captions in the band between.
              Picture and sound both come from input.mp4.
- Audio: joined spans -> stereo, gain + peak limiter iterated until integrated
  loudness is -14 LUFS (±0.15), 0.3s fades at the start and end.
- Captions: each span's segments re-split with word timestamps into cues of at
  most 2 lines x 16 full-width chars, >= 1s on screen, on the output timeline.
Skips when the inputs and render settings are unchanged since the last render
(fingerprint in .digest.stamp.json), unless --force.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import config as C
from common import (StepError, align_text, clip_spans, die, episode_dir, load_json, require_tool, run, save_json,
                    text_width)
from validate_clip import validate

LINE_W = float(C.MAX_CHARS_PER_LINE)
CUE_W = LINE_W * C.MAX_LINES

NO_LINE_START = set("、。，．,.！？!?」』）)]｝】〉》ーぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ々…‥・：:；;〜～")
NO_LINE_END = set("「『（([｛【〈《")
SENTENCE_END = set("。！？!?")
CLAUSE_END = set("、，,…")
DROP_AT_LINE_END = set("。、，,")


# --- Caption layout --------------------------------------------------------------------

def _script(c: str) -> str:
    if "぀" <= c <= "ゟ":
        return "hira"
    if "゠" <= c <= "ヿ":
        return "kata"
    if unicodedata.category(c).startswith("L") and ord(c) > 0x2E80:
        return "kanji"
    if c.isascii() and c.isalnum():
        return "ascii"
    return "other"


PARTICLES = set("はがをにでともへやてばか")
CUE_COST = 14.0  # higher = fewer, fuller (2-line) cues; lower = many short 1-line cues


def break_score(text: str, b: int, word_starts: set[int]) -> float:
    """How good it is to break `text` before index b (higher is better).

    No morphological analyser: punctuation is the strongest signal, then a
    particle followed by a new content word (the usual bunsetsu boundary).
    Breaks inside a script run (kanji compound, katakana word, okurigana) are bad.
    """
    prev, cur = text[b - 1], text[b]
    if cur in NO_LINE_START or prev in NO_LINE_END:
        return -100.0
    sp, sc = _script(prev), _script(cur)
    if sp == "ascii" and sc == "ascii":
        return -100.0  # never split an English word / number
    if prev in SENTENCE_END:
        s = 14.0
    elif prev in CLAUSE_END:
        s = 10.0
    elif prev.isspace() or cur.isspace():
        around = text[b - 2:b - 1] + text[b + 1:b + 2] if cur.isspace() else prev + cur
        # keep multi-word names like "Claude Code" on one line
        s = -4.0 if len(around) == 2 and all(_script(c) == "ascii" for c in around) else 6.0
    elif sp == "hira" and sc != "hira":
        s = 5.0 if prev in PARTICLES else 1.0
    elif sp == "hira":
        s = 1.0 if prev in PARTICLES else -3.0
    elif sp == sc:
        s = -6.0 if sp == "kata" else -3.0
    elif sc == "hira":
        s = -2.0  # content word -> its particle / okurigana
    else:
        s = 1.0
    if b in word_starts:
        s += 1.5
    return s


def eff_width(s: str) -> float:
    """Displayed width: trailing spaces and line-final 。、 are dropped when rendering."""
    s = s.strip()
    while s and s[-1] in DROP_AT_LINE_END:
        s = s[:-1]
    return text_width(s)


def best_line_break(text: str, a: int, b: int, word_starts: set[int]) -> tuple[int, float] | None:
    """Best index k to split text[a:b] into two lines of <= LINE_W each (None if impossible)."""
    best: tuple[int, float] | None = None
    for k in range(a + 1, b):
        left, right = eff_width(text[a:k]), eff_width(text[k:b])
        if left > LINE_W or right > LINE_W or not left or not right:
            continue
        sc = break_score(text, k, word_starts) - 0.3 * abs(left - right)
        if best is None or sc > best[1]:
            best = (k, sc)
    return best


def cue_quality(text: str, a: int, b: int, word_starts: set[int]) -> float:
    """How well text[a:b] lays out as one caption (single line, or a clean 2-line split)."""
    if eff_width(text[a:b]) <= LINE_W:
        return 4.0
    lb = best_line_break(text, a, b, word_starts)
    return -50.0 if lb is None else lb[1]


def split_cues(text: str, word_starts: set[int]) -> list[tuple[int, int]]:
    """Split one segment's text into [start, end) ranges that each fit 2 lines x LINE_W.

    Dynamic programming over break points: maximise break quality at cue
    boundaries plus line-layout quality inside each cue, with a small cost
    per cue so short fragments are only used when they read better.
    """
    n = len(text)
    lo = 0
    while lo < n and text[lo].isspace():
        lo += 1
    if lo >= n:
        return []
    INF = float("-inf")
    best = [INF] * (n + 1)
    back = [0] * (n + 1)
    best[lo] = 0.0
    for i in range(lo, n):
        if best[i] == INF:
            continue
        for j in range(i + 1, n + 1):
            w = eff_width(text[i:j])
            if w > CUE_W:
                break
            if j < n and text[j].isspace():
                continue  # break after the space instead
            inner_stops = sum(c in SENTENCE_END for c in text[i:j - 1])
            sc = (cue_quality(text, i, j, word_starts) - CUE_COST - 2.0 * inner_stops
                  - (4.0 if w < 4 else 0.0))
            if j < n:
                sc += break_score(text, j, word_starts)
            if best[i] + sc > best[j]:
                best[j], back[j] = best[i] + sc, i
    spans, j = [], n
    while j > lo:
        i = back[j]
        spans.append((i, j))
        j = i
    return [(a + len(text[a:b]) - len(text[a:b].lstrip()), b) for a, b in reversed(spans)]


def split_lines(text: str, word_starts: set[int]) -> list[str]:
    if eff_width(text) <= LINE_W:
        lines = [text]
    else:
        lb = best_line_break(text, 0, len(text), word_starts)
        if lb is None:  # hard cut at the line width
            k, w = 0, 0.0
            while k < len(text) and w + text_width(text[k]) <= LINE_W:
                w += text_width(text[k])
                k += 1
            lb = (k, 0.0)
        lines = [text[:lb[0]], text[lb[0]:]]
    out = []
    for line in lines:
        line = line.strip()
        while line and line[-1] in DROP_AT_LINE_END:
            line = line[:-1]
        if line:
            out.append(line)
    return out


@dataclass
class Cue:
    text: str
    times: list[tuple[float, float]]
    word_starts: set[int] = field(default_factory=set)

    @property
    def start(self) -> float:
        for c, t in zip(self.text, self.times):
            if not c.isspace():
                return t[0]
        return self.times[0][0]

    @property
    def speech_end(self) -> float:
        return max(t[1] for t in self.times)

    def __add__(self, other: "Cue") -> "Cue":
        joiner = " " if (_script(self.text[-1]) == "ascii" and _script(other.text[0]) == "ascii") else ""
        jt = [(self.speech_end, self.speech_end)] * len(joiner)
        shift = len(self.text) + len(joiner)
        return Cue(self.text + joiner + other.text, self.times + jt + other.times,
                   self.word_starts | {shift} | {p + shift for p in other.word_starts})


SNAP_MAX = 0.6  # never move a cue start by more than this (seconds)


def snap_start(t: float, silences: list[tuple[float, float]],
               prev_end: float = float("-inf")) -> float:
    """Align a cue start with the speech onset it belongs to.

    Whisper word timestamps drift by a few hundred ms around pauses, in either
    direction, so a start is moved to the end of a detected pause (= onset) when:
      - early: it falls *inside* the pause, or
      - late: the pause ended just before it AND the previous cue's speech had
        already ended when the pause began, i.e. the pause separates the two
        cues, so the audio between the onset and `t` belongs to this cue.
    Starts more than SNAP_MAX away from any onset are left alone.
    """
    for a, b in silences:
        if a <= t < b and b - t <= SNAP_MAX:
            return b
    late = [b for a, b in silences if b <= t and t - b <= SNAP_MAX and prev_end <= a + 0.35]
    return max(late) if late else t


def span_plan(clip: dict) -> list[tuple[float, float]]:
    """(source start, output duration) of each span, in output order.

    Durations are snapped to whole video frames so the audio and video of every
    span are exactly the same length and stay in sync across the hard cuts.
    """
    out = []
    for sp in clip_spans(clip):
        frames = max(1, round((sp["end"] - sp["start"]) * C.VIDEO_FPS))
        out.append((float(sp["start"]), frames / C.VIDEO_FPS))
    return out


def build_cues(clip: dict, transcript: dict, silences: list[tuple[float, float]] | None = None) -> list[dict]:
    """Return [{start, end, lines}] in output-timeline seconds (0 = start of the digest).

    `silences` are output-timeline (start, end) pauses detected in the rendered audio.
    Cues are built per span and never run across a cut into the next span.
    """
    out, offset = [], 0.0
    for span, (_, dur) in zip(clip_spans(clip), span_plan(clip)):
        local = [(max(a - offset, 0.0), min(b - offset, dur)) for a, b in silences or []
                 if b > offset and a < offset + dur]
        for c in span_cues(span, transcript, local, dur):
            out.append({"start": round(c["start"] + offset, 3), "end": round(c["end"] + offset, 3),
                        "lines": c["lines"]})
        offset += dur
    return out


def span_cues(span: dict, transcript: dict, silences: list[tuple[float, float]], span_len: float) -> list[dict]:
    """Cues of one span, relative to its start; `silences` are span-relative too."""
    by_id = {s["id"]: s for s in transcript["segments"]}
    segs = span["segments"]
    n = len(segs)
    cues: list[Cue] = []
    for i, cs in enumerate(segs):
        if not cs["text"].strip():
            continue
        t = by_id[cs["id"]] if "id" in cs else next(
            s for s in transcript["segments"] if s["start"] - 1e-3 <= cs["start"] <= s["end"] + 1e-3)
        full = t["text"]
        times, word_offsets = align_text(full, t["words"], t["start"], t["end"])
        wstarts = set(word_offsets)
        if n > 1 and i == 0:
            off = len(full) - len(cs["text"])
        elif n > 1 and i == n - 1:
            off = 0
        else:
            off = max(0, full.find(cs["text"]))
        text = cs["text"]
        times = times[off:off + len(text)]
        ws = {p - off for p in wstarts if off <= p < off + len(text)}
        for a, b in split_cues(text, ws):
            cues.append(Cue(text[a:b], times[a:b], {p - a for p in ws if a <= p < b}))

    # Merge cues that would flash by too quickly into their successor when they fit.
    merged: list[Cue] = []
    i = 0
    while i < len(cues):
        cur = cues[i]
        while (i + 1 < len(cues) and cues[i + 1].start - cur.start < C.MIN_CUE_SECONDS
               and cues[i + 1].start - cur.speech_end < 0.5
               and eff_width((cur + cues[i + 1]).text) <= CUE_W
               and cue_quality((m := cur + cues[i + 1]).text, 0, len(m.text), m.word_starts) >= 0):
            cur = cur + cues[i + 1]
            i += 1
        merged.append(cur)
        i += 1

    base = span["start"]
    starts = [snap_start(c.start - base, silences, merged[k - 1].speech_end - base if k else float("-inf"))
              for k, c in enumerate(merged)]
    if starts and starts[0] <= SNAP_MAX and not any(a < starts[0] for a, _ in silences):
        starts[0] = 0.0  # speech runs from the very start of the span: show the first caption at its first frame
    out = []
    for k, cue in enumerate(merged):
        start = starts[k]
        end = max(cue.speech_end - base, start)
        nxt = starts[k + 1] if k + 1 < len(merged) else span_len
        end = max(end, start + C.MIN_CUE_SECONDS)
        if nxt - end < C.MERGE_GAP_SECONDS:
            end = nxt  # close tiny gaps, and never overlap the next cue
        end = min(end, nxt, span_len)
        start = max(0.0, start)
        lines = split_lines(cue.text, cue.word_starts)
        if lines and end - start > 0.05:
            out.append({"start": start, "end": end, "lines": lines})
    return out


# --- ASS ----------------------------------------------------------------------------------

def ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(s: str) -> str:
    return s.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", " ")


def caption_style(layout: dict) -> tuple[int, int, int]:
    """(font size, ASS alignment, MarginV) for a layout.

    background: bottom-centre (2), CAPTION_MARGIN_V up from the bottom (lower third).
    stacked: middle-centre (5) = the vertical centre of the frame, which is the
    centre of the band between the two equal panels.
    """
    if layout["layout"] == "stacked":
        return C.STACK_FONT_SIZE, 5, 0
    return C.FONT_SIZE, 2, C.CAPTION_MARGIN_V


def write_ass(cues: list[dict], path: Path, title: str, layout: dict | None = None) -> None:
    size, align, margin_v = caption_style(layout or {"layout": "background"})
    style = ",".join(str(x) for x in [
        "Default", C.FONT_NAME, size, "&H00FFFFFF", "&H000000FF", "&H00101010", "&H80000000",
        -1 if C.FONT_BOLD else 0, 0, 0, 0, 100, 100, 0, 0, 1, C.CAPTION_OUTLINE, C.CAPTION_SHADOW,
        align, C.CAPTION_MARGIN_LR, C.CAPTION_MARGIN_LR, margin_v, 1])
    lines = [
        "[Script Info]",
        f"Title: {ass_escape(title)}",
        "ScriptType: v4.00+",
        f"PlayResX: {C.VIDEO_WIDTH}",
        f"PlayResY: {C.VIDEO_HEIGHT}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: {style}",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for c in cues:
        text = r"\N".join(ass_escape(line) for line in c["lines"])
        lines.append(f"Dialogue: 0,{ass_time(c['start'])},{ass_time(c['end'])},Default,,0,0,0,,{text}")
    tmp = path.with_suffix(".ass.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# --- ffmpeg ---------------------------------------------------------------------------------

def ff_escape(value: str) -> str:
    """Escape a filter option value for use inside -filter_complex (two levels)."""
    level1 = re.sub(r"([\\':])", r"\\\1", value)
    return re.sub(r"([\\'\[\],;])", r"\\\1", level1)


def font_available() -> bool:
    if C.FONTS_DIR.is_dir() and any("notosansjp" in p.name.lower().replace(" ", "").replace("-", "")
                                    for p in C.FONTS_DIR.iterdir()):
        return True
    if shutil.which("fc-list"):
        out = subprocess.run(["fc-list", f":family={C.FONT_NAME}", "family"], capture_output=True, text=True)
        if out.stdout.strip():
            return True
    pats = ["~/Library/Fonts/NotoSansJP*", "/Library/Fonts/NotoSansJP*", "/usr/share/fonts/**/NotoSansJP*",
            "~/.local/share/fonts/**/NotoSansJP*", "~/.fonts/**/NotoSansJP*"]
    return any(glob.glob(os.path.expanduser(p), recursive=True) for p in pats)


def span_inputs(src: Path, plan: list[tuple[float, float]]) -> list[str]:
    """One accurately seeked input per span (input seeking + re-encode is frame-accurate)."""
    args: list[str] = []
    for start, dur in plan:
        args += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src)]
    return args


def audio_chain(gain_db: float) -> str:
    """Stereo upmix -> gain -> peak limiter (measured and rendered identically)."""
    limit = 10 ** ((C.TRUE_PEAK_DB - 0.5) / 20)  # sample-peak limiter, margin for true peak
    return (f"aformat=sample_rates={C.OUTPUT_AUDIO_RATE}:channel_layouts=stereo,"
            f"volume={gain_db:.2f}dB,alimiter=limit={limit:.4f}:level=false:attack=5:release=50")


def audio_graph(plan: list[tuple[float, float]], gain_db: float, *, final_fades: bool) -> str:
    """filter_complex for inputs 0..n-1 (one per span) -> [a].

    Each span is trimmed/padded to exactly its frame-snapped duration, gets a tiny
    fade at every hard cut, then the spans are joined and normalised as one stream.
    """
    rate, n, j = C.OUTPUT_AUDIO_RATE, len(plan), C.JOIN_FADE_SECONDS
    parts = []
    for i, (_, dur) in enumerate(plan):
        f = f"[{i}:a]asetpts=PTS-STARTPTS,aresample={rate},apad,atrim=end_sample={round(dur * rate)}"
        if n > 1 and i > 0:
            f += f",afade=t=in:d={j}"
        if n > 1 and i < n - 1:
            f += f",afade=t=out:st={max(0.0, dur - j):.3f}:d={j}"
        parts.append(f"{f}[a{i}]")
    head = "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1," if n > 1 else "[a0]"
    chain = audio_chain(gain_db)
    if final_fades:
        total, fade = sum(d for _, d in plan), C.AUDIO_FADE_SECONDS
        chain += f",afade=t=in:st=0:d={fade},afade=t=out:st={max(0.0, total - fade):.3f}:d={fade}"
    parts.append(f"{head}{chain}[a]")
    return ";".join(parts)


def analyze_audio(src: Path, plan: list[tuple[float, float]], gain_db: float) -> tuple[float, float, list]:
    """Integrated loudness (LUFS), peak (dBFS) and output-timeline silences of the joined audio."""
    graph = (audio_graph(plan, gain_db, final_fades=False)
             + f";[a]silencedetect=noise={C.SILENCE_NOISE_DB}dB:d=0.12,ebur128=peak=true:framelog=quiet[out]")
    proc = run(["ffmpeg", "-hide_banner", "-nostats", *span_inputs(src, plan),
                "-filter_complex", graph, "-map", "[out]", "-f", "null", "-"])
    err = proc.stderr
    summary = err[err.rfind("Summary:"):]
    i_m = re.search(r"I:\s+(-?[\d.]+|-inf) LUFS", summary)
    p_m = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", summary)
    loud = float(i_m.group(1)) if i_m else float("-inf")
    peak = float(p_m.group(1)) if p_m else float("-inf")
    silences, open_at = [], None
    for kind, val in re.findall(r"silence_(start|end): (-?[\d.]+)", err):
        if kind == "start":
            open_at = max(0.0, float(val))
        elif open_at is not None:
            silences.append((open_at, float(val)))
            open_at = None
    if open_at is not None:
        silences.append((open_at, sum(d for _, d in plan)))
    return loud, peak, silences


def normalize(src: Path, plan: list[tuple[float, float]]) -> tuple[float, list]:
    """Find the gain that brings the limited, joined audio to TARGET_LUFS (iterative, <= 4 passes)."""
    gain, silences = 0.0, []
    for _ in range(4):
        loud, _, silences = analyze_audio(src, plan, gain)
        if loud == float("-inf") or loud < -70:
            return 0.0, silences  # silence: nothing to normalise
        if abs(loud - C.TARGET_LUFS) <= 0.15:
            break
        gain += C.TARGET_LUFS - loud
    return gain, silences


# --- Layout -----------------------------------------------------------------------------------

def load_layout(ep: Path) -> dict:
    """{"layout": "background"} or {"layout": "stacked", "tiles": [...]} from episode.json / config."""
    p = ep / C.EPISODE_CONFIG
    cfg = load_json(p) if p.exists() else {}
    if not isinstance(cfg, dict):
        raise StepError(f"{C.EPISODE_CONFIG} must be a JSON object")
    kind = cfg.get("layout", C.DEFAULT_LAYOUT)
    if kind == "background":
        return {"layout": kind}
    if kind != "stacked":
        raise StepError(f"{C.EPISODE_CONFIG}: layout must be 'background' or 'stacked', got {kind!r}")
    tiles = cfg.get("tiles", C.STACK_TILES)
    if not (isinstance(tiles, list) and len(tiles) == 2):
        raise StepError(f"{C.EPISODE_CONFIG}: tiles must list exactly 2 rectangles (top, bottom)")
    for i, t in enumerate(tiles):
        ok = isinstance(t, dict) and all(isinstance(t.get(k), (int, float)) for k in "xywh")
        if not ok or not (0 <= t["x"] < 1 and 0 <= t["y"] < 1 and 0 < t["w"] and 0 < t["h"]
                          and t["x"] + t["w"] <= 1.001 and t["y"] + t["h"] <= 1.001):
            raise StepError(f"{C.EPISODE_CONFIG}: tiles[{i}] must be {{x, y, w, h}} fractions inside the frame, "
                            f"got {t!r}")
    return {"layout": kind, "tiles": tiles}


def panel_height() -> int:
    return (C.VIDEO_HEIGHT - C.STACK_BAND_HEIGHT) // 2


def stacked_filter(src: str, out: str, tiles: list[dict]) -> str:
    """Two source tiles -> top and bottom panels (scaled + centre-cropped to fill), band between."""
    W, H, P = C.VIDEO_WIDTH, C.VIDEO_HEIGHT, panel_height()
    parts = [f"{src}split=2[s0][s1]"]
    for i, t in enumerate(tiles):
        parts.append(f"[s{i}]crop=w=iw*{t['w']}:h=ih*{t['h']}:x=iw*{t['x']}:y=ih*{t['y']},"
                     f"scale={W}:{P}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{P},setsar=1[t{i}]")
    parts.append(f"[t0]pad={W}:{H}:0:0:color={C.STACK_BAND_COLOR}[pad];"
                 f"[pad][t1]overlay=0:{H - P}:shortest=1,format=yuv420p{out}")
    return ";".join(parts)


def video_concat(plan: list[tuple[float, float]]) -> str:
    """Inputs 0..n-1 (one per span) -> [vc]: each exactly round(dur * fps) frames, joined by hard cuts."""
    fps, n, parts = C.VIDEO_FPS, len(plan), []
    for i, (_, dur) in enumerate(plan):
        parts.append(f"[{i}:v]setpts=PTS-STARTPTS,fps={fps},tpad=stop_mode=clone:stop_duration=1,"
                     f"trim=end_frame={round(dur * fps)},setpts=PTS-STARTPTS,setsar=1[v{i}]")
    joined = "".join(f"[v{i}]" for i in range(n))
    parts.append(f"{joined}concat=n={n}:v=1:a=0[vc]" if n > 1 else "[v0]null[vc]")
    return ";".join(parts)


def subtitles_filter(ass: Path) -> str:
    sub = f"ass=filename={ff_escape(ass.name)}"
    if C.FONTS_DIR.is_dir():
        sub += f":fontsdir={ff_escape(str(C.FONTS_DIR))}"
    return sub


def background_filter() -> str:
    W, H = C.VIDEO_WIDTH, C.VIDEO_HEIGHT
    if C.BACKGROUND_FIT == "contain":
        return (f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={C.BACKGROUND_PAD_COLOR}")
    if C.BACKGROUND_FIT == "cover":
        return f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H}"
    raise StepError(f"config.BACKGROUND_FIT must be 'cover' or 'contain', got {C.BACKGROUND_FIT!r}")


def input_stamp(*paths: Path) -> dict:
    """Fingerprint of everything that affects the render: inputs, settings and render code.

    Compared for equality (not "newer than"), so swapping in an asset with an
    older file date, or changing a caption/background setting, still re-renders.
    """
    files = {}
    for p in paths:
        for f in sorted(p.iterdir()) if p.is_dir() else [p]:
            if f.is_file():
                st = f.stat()
                files[str(f)] = [st.st_size, st.st_mtime_ns]
    settings = {k: v for k, v in vars(C).items() if k.isupper() and not isinstance(v, Path)}
    return {"files": files, "settings": json.loads(json.dumps(settings, default=str))}


def render(ep: Path, clip: dict, layout: dict, src: Path, out: Path, ass: Path, gain: float) -> None:
    plan = span_plan(clip)
    total = sum(d for _, d in plan)
    graph = [audio_graph(plan, gain, final_fades=True)]
    inputs = span_inputs(src, plan)
    if layout["layout"] == "stacked":
        graph += [video_concat(plan), stacked_filter("[vc]", "[vl]", layout["tiles"]), f"[vl]{subtitles_filter(ass)}[v]"]
        tune: list[str] = []
    else:
        inputs += ["-loop", "1", "-framerate", str(C.VIDEO_FPS), "-i", str(C.BACKGROUND_IMAGE)]
        graph.append(f"[{len(plan)}:v]{background_filter()},setsar=1,format=yuv420p,{subtitles_filter(ass)}[v]")
        tune = ["-tune", "stillimage"]

    tmp = out.with_name("digest.tmp.mp4")
    run(["ffmpeg", "-y", "-hide_banner", "-v", "error", *inputs,
         "-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]",
         "-t", f"{total:.3f}", "-r", str(C.VIDEO_FPS),
         "-c:v", "libx264", "-preset", C.X264_PRESET, *tune, "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", C.OUTPUT_AUDIO_BITRATE,
         "-movflags", "+faststart", str(tmp)], cwd=ep)
    os.replace(tmp, out)


def has_video_stream(path: Path) -> bool:
    proc = run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_type",
                "-of", "csv=p=0", str(path)])
    return "video" in proc.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode_dir")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--audio-from-video", action="store_true",
                    help="background layout: cut audio from input.mp4 instead of audio.mp3")
    args = ap.parse_args()
    ep = episode_dir(args.episode_dir)
    require_tool("ffmpeg")

    clip_p, tr_p, out, ass = ep / C.CLIP, ep / C.TRANSCRIPT, ep / C.DIGEST_VIDEO, ep / C.DIGEST_ASS
    try:
        layout = load_layout(ep)
    except StepError as e:
        die(str(e))
    stacked = layout["layout"] == "stacked"
    # The stacked layout cuts picture and sound from the same file, so they stay in sync.
    src = ep / (C.INPUT_VIDEO if stacked or args.audio_from_video else C.AUDIO)
    if not src.exists():
        die(f"missing {src}")
    if stacked and not has_video_stream(src):
        die(f"{C.EPISODE_CONFIG} says layout 'stacked' but {src.name} has no video stream")
    if not stacked and not C.BACKGROUND_IMAGE.exists():
        die(f"missing background image {C.BACKGROUND_IMAGE}")

    errors, _ = validate(ep)
    if errors:
        die("clip.json is invalid (run validate_clip.py):\n  " + "\n  ".join(errors))

    stamp_p = ep / C.DIGEST_STAMP
    code = [Path(__file__), Path(__file__).with_name("common.py"), Path(__file__).with_name("config.py")]
    assets = [] if stacked else [C.BACKGROUND_IMAGE]
    stamp = input_stamp(clip_p, tr_p, src, ep / C.EPISODE_CONFIG, *assets, C.FONTS_DIR, *code)
    if (not args.force and out.exists() and ass.exists() and stamp_p.exists()
            and load_json(stamp_p) == stamp):
        print(f"skip  {C.DIGEST_VIDEO} (up to date)")
        return
    if not font_available():
        print(f"warn  font '{C.FONT_NAME}' not found; libass will fall back to another Japanese font "
              f"(see README: Setup)")

    clip, transcript = load_json(clip_p), load_json(tr_p)
    plan = span_plan(clip)
    try:
        gain, silences = normalize(src, plan)
        cues = build_cues(clip, transcript, silences)
        write_ass(cues, ass, clip.get("title", ""), layout)
        print(f"make  {C.DIGEST_ASS}  ({len(cues)} cues)")
        render(ep, clip, layout, src, out, ass, gain)
    except StepError as e:
        die(str(e))
    save_json(stamp_p, stamp)
    print(f"make  {C.DIGEST_VIDEO}  ({sum(d for _, d in plan):.1f}s, {len(plan)} span(s), {layout['layout']} layout)")


if __name__ == "__main__":
    main()
