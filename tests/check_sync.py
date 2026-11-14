#!/usr/bin/env python3
"""Check caption sync of a rendered digest against ground truth (±0.2s).

Usage: uv run tests/check_sync.py episodes/<slug> tests/fixtures/sample/ground_truth.json [--flashes]

For every cue that begins at the start of a ground-truth phrase, compare:
  - cue start vs. the phrase start (mapped onto the digest timeline), and
  - cue start vs. the actual speech onset measured in digest.mp4 (silencedetect),
which also proves the audio cuts are aligned. Multi-span digests are checked
span by span (each span's phrases against the cues inside its output window).
--flashes (video fixture): the source video flashes a box at every phrase start;
check that each flash appears in digest.mp4 at the mapped phrase time, i.e. the
picture is in sync with the sound after re-framing and hard cuts.
Exit 1 if any |delta| > tolerance.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

TOL = 0.2
FLASH_TOL = 0.07  # two frames at 30 fps
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import clip_spans  # noqa: E402
from render import span_plan  # noqa: E402


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    return "".join(c for c in s if c.isalnum())


def ass_seconds(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def read_cues(ass: Path) -> list[tuple[float, float, str]]:
    cues = []
    for line in ass.read_text(encoding="utf-8").splitlines():
        if line.startswith("Dialogue:"):
            parts = line.split(",", 9)
            cues.append((ass_seconds(parts[1]), ass_seconds(parts[2]), parts[9].replace(r"\N", "")))
    return cues


def speech_onsets(video: Path) -> list[float]:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(video), "-vn", "-af",
                           "silencedetect=noise=-40dB:d=0.08", "-f", "null", "-"],
                          capture_output=True, text=True)
    onsets = [0.0] + [float(x) for x in re.findall(r"silence_end: ([\d.]+)", proc.stderr)]
    return onsets


def match_span(span: dict, offset: float, dur: float, truth: list[dict],
               cues: list[tuple[float, float, str]], onsets: list[float]) -> list[tuple]:
    """Rows (cue_t, truth_t, onset, d_truth, d_onset, text) for cues that start a phrase in this span."""
    phrases = [p for p in truth if p["end"] > span["start"] and p["start"] < span["end"]]
    mine = [c for c in cues if offset - 1e-3 <= c[0] < offset + dur]
    t_text, t_start_at = "", {}
    for p in phrases:
        t_start_at[len(t_text)] = p
        t_text += norm(p["text"])
    c_text, c_start_at = "", {}
    for i, (_, _, txt) in enumerate(mine):
        c_start_at[len(c_text)] = i
        c_text += norm(txt)

    c2t = {}
    for a, b, size in difflib.SequenceMatcher(None, c_text, t_text, autojunk=False).get_matching_blocks():
        for k in range(size):
            c2t[a + k] = b + k

    rows = []
    for c_off, i in c_start_at.items():
        t_off = c2t.get(c_off)
        if t_off is None or t_off not in t_start_at:
            continue  # cue starts mid-phrase: no exact ground truth for it
        p = t_start_at[t_off]
        if p["start"] < span["start"]:
            continue
        truth_t = p["start"] - span["start"] + offset
        cue_t = mine[i][0]
        onset = min(onsets, key=lambda o: abs(o - truth_t))
        rows.append((cue_t, truth_t, onset, cue_t - truth_t, cue_t - onset, mine[i][2][:18]))
    return rows


def flash_times(video: Path) -> list[float]:
    """Rising edges of the bright flash box in the top panel of digest.mp4."""
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(video), "-an", "-vf",
                           "crop=200:200:440:330,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-",
                           "-f", "null", "-"], capture_output=True, text=True)
    times, vals = [], []
    for line in proc.stdout.splitlines():
        if line.startswith("frame:"):
            times.append(float(re.search(r"pts_time:([\d.]+)", line).group(1)))
        elif "YAVG=" in line:
            vals.append(float(line.split("=")[1]))
    return [t for k, (t, v) in enumerate(zip(times, vals)) if v > 128 and (k == 0 or vals[k - 1] <= 128)]


def check_flashes(clip: dict, truth: list[dict], video: Path) -> float:
    flashes = flash_times(video)
    offset, worst, n = 0.0, 0.0, 0
    for span, (_, dur) in zip(clip_spans(clip), span_plan(clip)):
        for p in truth:
            # skip flashes the cut itself starts/ends inside (box only partly visible)
            if span["start"] + 0.1 <= p["start"] <= span["start"] + dur - 0.3:
                want = p["start"] - span["start"] + offset
                got = min(flashes, key=lambda f: abs(f - want)) if flashes else float("inf")
                worst, n = max(worst, abs(got - want)), n + 1
                if abs(got - want) > FLASH_TOL:
                    print(f"flash at {want:7.2f}s: nearest {got:7.2f}s  <-- picture out of sync")
        offset += dur
    print(f"{n} flashes checked; worst |Δ| = {worst:.3f}s (tol {FLASH_TOL}s)")
    return worst if n else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("ground_truth")
    ap.add_argument("--flashes", action="store_true", help="also check picture sync (video fixture)")
    args = ap.parse_args()
    ep = Path(args.episode_dir)
    clip = json.loads((ep / "clip.json").read_text(encoding="utf-8"))
    truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))["phrases"]
    cues = read_cues(ep / "digest.ass")
    onsets = speech_onsets(ep / "digest.mp4")

    rows, offset = [], 0.0
    for span, (_, dur) in zip(clip_spans(clip), span_plan(clip)):
        rows += match_span(span, offset, dur, truth, cues, onsets)
        offset += dur
    worst = max((max(abs(r[3]), abs(r[4])) for r in rows), default=0.0)

    print(f"{'cue':>7} {'truth':>7} {'onset':>7} {'Δtruth':>7} {'Δonset':>7}  text")
    for r in rows:
        flag = "" if max(abs(r[3]), abs(r[4])) <= TOL else "  <-- out of sync"
        print(f"{r[0]:7.2f} {r[1]:7.2f} {r[2]:7.2f} {r[3]:+7.2f} {r[4]:+7.2f}  {r[5]}{flag}")
    print(f"\n{len(rows)}/{len(cues)} cues checked against phrase starts; worst |Δ| = {worst:.3f}s (tol {TOL}s)")
    if not rows or worst > TOL:
        sys.exit(1)
    if args.flashes and check_flashes(clip, truth, ep / "digest.mp4") > FLASH_TOL:
        sys.exit(1)
    print("SYNC OK")


if __name__ == "__main__":
    main()
