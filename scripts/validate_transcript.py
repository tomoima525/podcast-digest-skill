#!/usr/bin/env python3
"""Validate transcript.json against transcript.raw.json.

Usage: uv run scripts/validate_transcript.py episodes/<slug>

Checks: schema, language, duration, segment count, ids/timestamps/words unchanged,
original_text preserved, and every changed segment is listed in corrections.md.
Exit code 0 = valid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config as C
from common import episode_dir, load_json

NUM = (int, float)


def schema_errors(t: object, name: str) -> list[str]:
    errs: list[str] = []
    if not isinstance(t, dict):
        return [f"{name}: top level must be an object"]
    if t.get("language") != "ja":
        errs.append(f"{name}: language must be 'ja'")
    if not isinstance(t.get("duration"), NUM) or t["duration"] <= 0:
        errs.append(f"{name}: duration must be a positive number")
    segs = t.get("segments")
    if not isinstance(segs, list) or not segs:
        return errs + [f"{name}: segments must be a non-empty list"]
    prev_start = -1.0
    for i, s in enumerate(segs):
        p = f"{name}: segments[{i}]"
        if not isinstance(s, dict):
            errs.append(f"{p} must be an object")
            continue
        if s.get("id") != i:
            errs.append(f"{p}.id must be {i}")
        for k in ("start", "end"):
            if not isinstance(s.get(k), NUM):
                errs.append(f"{p}.{k} must be a number")
        for k in ("text", "original_text"):
            if not isinstance(s.get(k), str):
                errs.append(f"{p}.{k} must be a string")
        if not isinstance(s.get("words"), list):
            errs.append(f"{p}.words must be a list")
        else:
            for j, w in enumerate(s["words"]):
                if not (isinstance(w, dict) and isinstance(w.get("word"), str)
                        and isinstance(w.get("start"), NUM) and isinstance(w.get("end"), NUM)):
                    errs.append(f"{p}.words[{j}] must be {{word, start, end}}")
                    break
        if isinstance(s.get("start"), NUM) and isinstance(s.get("end"), NUM):
            if s["end"] < s["start"]:
                errs.append(f"{p}: end < start")
            if s["start"] < prev_start:
                errs.append(f"{p}: segments not in time order")
            prev_start = s["start"]
    return errs


def corrected_ids(md: str) -> set[int]:
    from apply_corrections import parse_corrections  # local import: avoids a cycle
    ids: set[int] = set()
    for row_ids, *_ in parse_corrections(md):
        ids.update(row_ids)
    return ids


def validate(ep: Path) -> list[str]:
    raw_p, fix_p, md_p = ep / C.TRANSCRIPT_RAW, ep / C.TRANSCRIPT, ep / C.CORRECTIONS
    for p in (raw_p, fix_p):
        if not p.exists():
            return [f"missing {p.name}"]
    raw, fix = load_json(raw_p), load_json(fix_p)
    errs = schema_errors(raw, "raw") + schema_errors(fix, "transcript")
    if errs:
        return errs
    if fix["duration"] != raw["duration"]:
        errs.append("duration changed")
    if len(fix["segments"]) != len(raw["segments"]):
        return errs + [f"segment count changed: {len(raw['segments'])} -> {len(fix['segments'])}"]
    changed = []
    for r, f in zip(raw["segments"], fix["segments"]):
        sid = r["id"]
        if (f["start"], f["end"]) != (r["start"], r["end"]):
            errs.append(f"segment {sid}: timestamps changed")
        if f["words"] != r["words"]:
            errs.append(f"segment {sid}: words changed (only `text` may change)")
        if f["original_text"] != r["text"]:
            errs.append(f"segment {sid}: original_text must equal the raw text")
        extra = set(f) - {"id", "start", "end", "text", "original_text", "words"}
        if extra:
            errs.append(f"segment {sid}: unexpected keys {sorted(extra)}")
        if f["text"] != r["text"]:
            changed.append(sid)
    if not md_p.exists():
        errs.append(f"missing {C.CORRECTIONS}")
    else:
        try:
            listed = corrected_ids(md_p.read_text(encoding="utf-8"))
        except ValueError as e:
            errs.append(f"{C.CORRECTIONS}: {e}")
        else:
            missing = [i for i in changed if i not in listed]
            if missing:
                errs.append(f"segments changed but not listed in {C.CORRECTIONS}: {missing}")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode_dir")
    ep = episode_dir(ap.parse_args().episode_dir)
    errs = validate(ep)
    if errs:
        print("INVALID transcript.json:\n  " + "\n  ".join(errs), file=sys.stderr)
        sys.exit(1)
    fix = load_json(ep / C.TRANSCRIPT)
    n = sum(s["text"] != s["original_text"] for s in fix["segments"])
    print(f"ok    transcript.json: {len(fix['segments'])} segments, {n} corrected")


if __name__ == "__main__":
    main()
