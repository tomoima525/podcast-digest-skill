#!/usr/bin/env python3
"""Print a transcript compactly for reading / searching (steps 1b and 2).

Usage:
  uv run scripts/show_transcript.py episodes/<slug>                  # everything
  uv run scripts/show_transcript.py episodes/<slug> --grep 料金 --grep pricing -C 3
  uv run scripts/show_transcript.py episodes/<slug> --from-id 120 --to-id 160
  uv run scripts/show_transcript.py episodes/<slug> --words 133      # word timings
  uv run scripts/show_transcript.py episodes/<slug> --raw            # raw whisper text

Line format:  [id] start–end (len) text      (times are mm:ss.ss of audio.mp3)
Uses transcript.json when present, else transcript.raw.json. --grep is
case-insensitive and also matches the raw text of corrected segments.
"""

from __future__ import annotations

import argparse

import config as C
from common import die, episode_dir, fmt_ts, load_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode_dir")
    ap.add_argument("--raw", action="store_true", help="read transcript.raw.json")
    ap.add_argument("--grep", action="append", default=[], help="keyword (repeatable, OR)")
    ap.add_argument("-C", "--context", type=int, default=0, help="segments of context around matches")
    ap.add_argument("--from-id", type=int)
    ap.add_argument("--to-id", type=int)
    ap.add_argument("--words", type=int, metavar="ID", help="show word timestamps of one segment")
    ap.add_argument("--diff", action="store_true", help="also show raw text of corrected segments")
    args = ap.parse_args()
    ep = episode_dir(args.episode_dir)

    path = ep / (C.TRANSCRIPT_RAW if args.raw or not (ep / C.TRANSCRIPT).exists() else C.TRANSCRIPT)
    if not path.exists():
        die(f"missing {path}")
    t = load_json(path)
    segs = t["segments"]
    print(f"# {path.name}: {len(segs)} segments, {fmt_ts(t['duration'])}")

    if args.words is not None:
        if not 0 <= args.words < len(segs):
            die(f"no segment {args.words}")
        s = segs[args.words]
        print(f"[{s['id']}] {s['text']}")
        for i, w in enumerate(s["words"]):
            print(f"  {i:3d}  {w['start']:9.3f}–{w['end']:9.3f}  {w['word']}")
        return

    keep = set(range(len(segs)))
    if args.from_id is not None or args.to_id is not None:
        lo = args.from_id or 0
        hi = args.to_id if args.to_id is not None else len(segs) - 1
        keep = {i for i in keep if lo <= i <= hi}
    if args.grep:
        terms = [g.casefold() for g in args.grep]
        hits = {i for i in keep if any(term in (segs[i]["text"] + " " + segs[i]["original_text"]).casefold()
                                       for term in terms)}
        keep = {j for i in hits for j in range(i - args.context, i + args.context + 1) if 0 <= j < len(segs)}
        print(f"# {len(hits)} matching segments")

    prev = None
    for i in sorted(keep):
        if prev is not None and i != prev + 1:
            print("  ...")
        s = segs[i]
        print(f"[{s['id']}] {fmt_ts(s['start'])}–{fmt_ts(s['end'])} ({s['end'] - s['start']:.1f}s) {s['text']}")
        if args.diff and s["text"] != s["original_text"]:
            print(f"      raw: {s['original_text']}")
        prev = i


if __name__ == "__main__":
    main()
