#!/usr/bin/env python3
"""Validate clip.json against transcript.json (and audio.mp3 duration).

Usage: uv run scripts/validate_clip.py episodes/<slug>

Errors: schema; total of all spans <= 180s; spans chronological and not
overlapping; times inside the audio; each span's segments are a contiguous
run of transcript segments with matching times/text (the first may be trimmed
to a word start + suffix of its text, the last to a word end + prefix);
padding <= 0.3s and never past a neighbouring word; alternatives.
Warnings: a span opens with a back-reference ("さっき", "前回", ...).
Accepts the original single-span clip.json (start/end/segments at top level).
Exit code 0 = valid (warnings allowed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config as C
from common import StepError, clip_duration, clip_spans, episode_dir, flatten_words, load_json, media_duration

EPS = 1e-3
NUM = (int, float)
BACK_REFERENCES = ("さっき", "先ほど", "先程", "前回", "前に言った", "さきほど", "それで", "で、", "なので",
                   "ちなみに", "あと、", "その話", "その件")


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= EPS


def _check_times(span: dict, label: str, max_t: float, *, need_text: tuple[str, ...]) -> list[str]:
    errs = []
    s, e = span.get("start"), span.get("end")
    if not isinstance(s, NUM) or not isinstance(e, NUM):
        return [f"{label}: start/end must be numbers"]
    if not 0 <= s < e:
        errs.append(f"{label}: need 0 <= start < end (got {s}, {e})")
    if e > max_t + 0.05:
        errs.append(f"{label}: end {e:.2f}s is past the end of the audio ({max_t:.2f}s)")
    if e - s > C.MAX_CLIP_SECONDS + EPS:
        errs.append(f"{label}: duration {e - s:.2f}s exceeds {C.MAX_CLIP_SECONDS:.0f}s")
    for k in need_text:
        if not isinstance(span.get(k), str) or not span[k].strip():
            errs.append(f"{label}: {k} must be a non-empty string")
    return errs


def _check_segments(span: dict, label: str, tr: dict, flat: list[dict]) -> tuple[list[str], list[int]]:
    """Segments of one span vs transcript.json, plus padding vs neighbouring words."""
    errs: list[str] = []
    segs = span.get("segments")
    if not isinstance(segs, list) or not segs:
        return [f"{label}: segments must be a non-empty list"], []
    by_id = {s["id"]: s for s in tr["segments"]}
    ids = []
    for i, cs in enumerate(segs):
        if not (isinstance(cs, dict) and isinstance(cs.get("start"), NUM) and isinstance(cs.get("end"), NUM)
                and isinstance(cs.get("text"), str)):
            return [f"{label}.segments[{i}] must be {{start, end, text}}"], []
        sid = cs.get("id")
        if sid is None:  # tolerate id-less segments: match by overlap
            match = [t for t in tr["segments"] if t["start"] - EPS <= cs["start"] <= t["end"] + EPS
                     and t["start"] - EPS <= cs["end"] <= t["end"] + EPS]
            sid = match[0]["id"] if match else None
        if sid not in by_id:
            return [f"{label}.segments[{i}] does not correspond to a transcript segment"], []
        ids.append(sid)
    if ids != list(range(ids[0], ids[0] + len(ids))):
        return [f"{label}: segments must be contiguous transcript ids, got {ids}"], ids

    n = len(segs)
    for i, (cs, sid) in enumerate(zip(segs, ids)):
        t = by_id[sid]
        word_starts = [w["start"] for w in t["words"]]
        word_ends = [w["end"] for w in t["words"]]
        p = f"{label}.segments[{i}] (id {sid})"
        trimmed_start = i == 0 and not _close(cs["start"], t["start"])
        trimmed_end = i == n - 1 and not _close(cs["end"], t["end"])
        if trimmed_start and not any(_close(cs["start"], x) for x in word_starts):
            errs.append(f"{p}: trimmed start {cs['start']} is not a word start")
        if not trimmed_start and not _close(cs["start"], t["start"]):
            errs.append(f"{p}: start {cs['start']} != transcript {t['start']}")
        if trimmed_end and not any(_close(cs["end"], x) for x in word_ends):
            errs.append(f"{p}: trimmed end {cs['end']} is not a word end")
        if not trimmed_end and not _close(cs["end"], t["end"]):
            errs.append(f"{p}: end {cs['end']} != transcript {t['end']}")
        full = t["text"]
        if n == 1:
            ok = cs["text"] in full
        elif i == 0:
            ok = full.endswith(cs["text"])
        elif i == n - 1:
            ok = full.startswith(cs["text"])
        else:
            ok = cs["text"] == full
        if not ok or not cs["text"].strip() and full.strip():
            errs.append(f"{p}: text does not match transcript.json: {cs['text']!r} vs {full!r}")

    first, last = segs[0], segs[-1]
    # Words outside a *trimmed* first/last segment are not part of the span. Untrimmed
    # segments keep all their words: whisper word times can stick out past the segment's
    # own start/end by a few hundredths, and build_clip pads from the words.
    cut_start = not _close(first["start"], by_id[ids[0]]["start"])
    cut_end = not _close(last["end"], by_id[ids[-1]]["end"])
    inside = [k for k, w in enumerate(flat)
              if ids[0] <= w["segment_id"] <= ids[-1]
              and not (cut_start and w["segment_id"] == ids[0] and w["start"] < first["start"] - EPS)
              and not (cut_end and w["segment_id"] == ids[-1] and w["end"] > last["end"] + EPS)]
    s, e = span["start"], span["end"]
    if inside:
        fw, lw = flat[inside[0]], flat[inside[-1]]
        if s > fw["start"] + EPS:
            errs.append(f"{label}: start {s} cuts into the first word ({fw['start']})")
        if s < fw["start"] - C.CLIP_PADDING_SECONDS - 0.01:
            errs.append(f"{label}: more than {C.CLIP_PADDING_SECONDS}s padding before the first word")
        if inside[0] > 0 and s < flat[inside[0] - 1]["end"] - EPS and s < fw["start"] - EPS:
            errs.append(f"{label}: start padding reaches into the previous word")
        if e < lw["end"] - EPS:
            errs.append(f"{label}: end {e} cuts into the last word ({lw['end']})")
        if e > lw["end"] + C.CLIP_PADDING_SECONDS + 0.01:
            errs.append(f"{label}: more than {C.CLIP_PADDING_SECONDS}s padding after the last word")
        if inside[-1] + 1 < len(flat) and e > flat[inside[-1] + 1]["start"] + EPS and e > lw["end"] + EPS:
            errs.append(f"{label}: end padding reaches into the next word")
    elif s > first["start"] + EPS or e < last["end"] - EPS:
        errs.append(f"{label}: start/end must cover its segments")
    return errs, ids


def validate(ep: Path) -> tuple[list[str], list[str]]:
    tr_p, clip_p = ep / C.TRANSCRIPT, ep / C.CLIP
    for p in (tr_p, clip_p):
        if not p.exists():
            return [f"missing {p.name}"], []
    tr, clip = load_json(tr_p), load_json(clip_p)
    errs: list[str] = []
    warns: list[str] = []
    if not isinstance(clip, dict):
        return ["clip.json must be an object"], []

    max_t = float(tr["duration"])
    audio = ep / C.AUDIO
    if audio.exists():
        try:
            max_t = min(max_t, media_duration(audio))
        except StepError:
            warns.append("could not read audio.mp3 duration; using transcript duration")

    if not isinstance(clip.get("instruction"), str) or not clip["instruction"].strip():
        errs.append("instruction must be the human's instruction (non-empty string)")
    for k in ("title", "reason"):
        if not isinstance(clip.get(k), str) or not clip[k].strip():
            errs.append(f"{k} must be a non-empty string")

    spans = clip_spans(clip)
    if not isinstance(spans, list) or not spans or not all(isinstance(s, dict) for s in spans):
        return errs + ["spans must be a non-empty list of objects"], warns
    multi = len(spans) > 1
    for i, sp in enumerate(spans):
        errs += _check_times(sp, f"spans[{i}]", max_t, need_text=("reason",) if multi else ())

    alts = clip.get("alternatives", [])
    if not isinstance(alts, list) or len(alts) > C.MAX_ALTERNATIVES:
        errs.append(f"alternatives must be a list of at most {C.MAX_ALTERNATIVES}")
        alts = []
    for i, alt in enumerate(alts):
        errs += (_check_times(alt, f"alternatives[{i}]", max_t, need_text=("title", "reason"))
                 if isinstance(alt, dict) else [f"alternatives[{i}] must be an object"])
    if errs:
        return errs, warns

    total = clip_duration(clip)
    if total > C.MAX_CLIP_SECONDS + EPS:
        errs.append(f"spans total {total:.2f}s exceeds {C.MAX_CLIP_SECONDS:.0f}s")
    if "duration" in clip and abs(clip["duration"] - total) > 0.01:
        errs.append(f"duration {clip['duration']} != sum of spans {total:.3f}")

    flat = flatten_words(tr)
    prev_ids: list[int] = []
    for i, sp in enumerate(spans):
        label = f"spans[{i}]" if multi or "spans" in clip else "clip"
        span_errs, ids = _check_segments(sp, label, tr, flat)
        errs += span_errs
        if i > 0:
            if sp["start"] < spans[i - 1]["end"] - EPS:
                errs.append(f"{label}: starts before spans[{i - 1}] ends (spans must be chronological "
                            f"and not overlap)")
            elif ids and prev_ids and ids[0] <= prev_ids[-1]:
                errs.append(f"{label}: shares segments with spans[{i - 1}]")
        prev_ids = ids or prev_ids
        if ids:
            opener = sp["segments"][0]["text"].lstrip()
            for ref in BACK_REFERENCES:
                if opener.startswith(ref):
                    warns.append(f"{label} opens with {ref!r} — make sure it is understandable on its own "
                                 f"(trim with start_at if needed)")
                    break
    return errs, warns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode_dir")
    ep = episode_dir(ap.parse_args().episode_dir)
    errs, warns = validate(ep)
    for w in warns:
        print(f"warn  {w}")
    if errs:
        print("INVALID clip.json:\n  " + "\n  ".join(errs), file=sys.stderr)
        sys.exit(1)
    clip = load_json(ep / C.CLIP)
    print(f"ok    clip.json: {len(clip_spans(clip))} span(s), {clip_duration(clip):.1f}s total, "
          f"{len(clip.get('alternatives', []))} alternatives")


if __name__ == "__main__":
    main()
