#!/usr/bin/env python3
"""Step 2 helper: clip.draft.json (segment ids chosen by the agent) -> clip.json.

Usage: uv run scripts/build_clip.py episodes/<slug> [--force] [--promote N]

The agent decides *which* segments to feature; this script does the mechanical
part: snapping to word timestamps, 0.3s padding (never past a neighbouring word),
copying segment text, ordering, and the 180s total check. Draft format:

{
  "instruction": "the human's instruction, verbatim",
  "title": "title of the whole short", "reason": "why these parts",
  "spans": [                                  # 1+ highlights, stitched with hard cuts
    { "start_id": 120, "end_id": 151,
      "start_at":  "optional: text inside segment start_id where the span begins",
      "end_after": "optional: text inside segment end_id after which it ends",
      "title": "optional span title", "reason": "why this part" },
    { "start_id": 410, "end_id": 433, "reason": "..." }
  ],
  "alternatives": [ { "start_id": 300, "end_id": 322, "title": "...", "reason": "..." } ]
}

The original single-span draft (start_id/end_id/... at the top level) is still
accepted. start_at / end_after trim the first/last segment at a word boundary,
e.g. to drop a leading "で、さっきの話ですけど、". Spans are output in
chronological order; overlapping spans are an error. The total of all spans
must be <= 180s.

--promote N: with one span, swap alternatives[N] with it (checkpoint 2);
with several spans, move alternatives[N] into the spans.
Regenerates clip.json whenever the draft or transcript is newer (or --force),
then runs validate_clip.
"""

from __future__ import annotations

import argparse

import config as C
from common import (align_text, clip_duration, die, episode_dir, flatten_words, fmt_ts, is_fresh, load_json,
                    save_json)
from validate_clip import validate


class DraftError(ValueError):
    pass


def _word_index_at(seg: dict, t: float, *, by: str) -> int:
    words = seg["words"]
    if by == "start":
        cands = [i for i, w in enumerate(words) if w["start"] <= t + 0.05] or [0]
        return max(cands, key=lambda i: words[i]["start"])
    return min(range(len(words)), key=lambda i: abs(words[i]["end"] - t))


def build_span(transcript: dict, span: dict, label: str) -> dict:
    segs = transcript["segments"]
    try:
        a, b = int(span["start_id"]), int(span["end_id"])
    except (KeyError, TypeError, ValueError):
        raise DraftError(f"{label}: start_id and end_id are required integers") from None
    if not (0 <= a <= b < len(segs)):
        raise DraftError(f"{label}: need 0 <= start_id <= end_id < {len(segs)}, got {a}..{b}")

    first, last = segs[a], segs[b]
    out_segs = [{"id": s["id"], "start": s["start"], "end": s["end"], "text": s["text"]} for s in segs[a:b + 1]]

    # --- start --------------------------------------------------------------------
    start_off = 0
    first_word = (a, 0) if first["words"] else None
    start_at = span.get("start_at")
    if start_at:
        start_off = first["text"].find(start_at)
        if start_off < 0:
            raise DraftError(f"{label}: start_at {start_at!r} not found in segment {a}: {first['text']!r}")
        if start_off > 0 and first["words"]:
            times, _ = align_text(first["text"], first["words"], first["start"], first["end"])
            wi = _word_index_at(first, times[start_off][0], by="start")
            first_word = (a, wi)
            out_segs[0]["start"] = first["words"][wi]["start"]
        out_segs[0]["text"] = first["text"][start_off:]

    # --- end ----------------------------------------------------------------------
    last_word = (b, len(last["words"]) - 1) if last["words"] else None
    end_after = span.get("end_after")
    if end_after:
        base = start_off if a == b else 0
        idx = last["text"].find(end_after, base)
        if idx < 0:
            raise DraftError(f"{label}: end_after {end_after!r} not found in segment {b}: {last['text']!r}")
        end_off = idx + len(end_after)
        if end_off < len(last["text"]) and last["words"]:
            times, _ = align_text(last["text"], last["words"], last["start"], last["end"])
            wi = _word_index_at(last, times[end_off - 1][1], by="end")
            last_word = (b, wi)
            out_segs[-1]["end"] = last["words"][wi]["end"]
        out_segs[-1]["text"] = out_segs[-1]["text"][: end_off - (start_off if a == b else 0)]

    # --- padding, bounded by neighbouring words --------------------------------------
    flat = flatten_words(transcript)
    pos = {(w["segment_id"], w["index"]): i for i, w in enumerate(flat)}

    if first_word is not None:
        fi = pos[first_word]
        speech_start = flat[fi]["start"]
        prev_end = flat[fi - 1]["end"] if fi > 0 else 0.0
    else:
        speech_start = out_segs[0]["start"]
        before = [w for w in flat if w["segment_id"] < a]
        prev_end = before[-1]["end"] if before else 0.0
    start = max(speech_start - C.CLIP_PADDING_SECONDS, prev_end, 0.0)
    start = min(start, speech_start)

    if last_word is not None:
        li = pos[last_word]
        speech_end = flat[li]["end"]
        next_start = flat[li + 1]["start"] if li + 1 < len(flat) else transcript["duration"]
    else:
        speech_end = out_segs[-1]["end"]
        after = [w for w in flat if w["segment_id"] > b]
        next_start = after[0]["start"] if after else transcript["duration"]
    end = min(speech_end + C.CLIP_PADDING_SECONDS, next_start, transcript["duration"])
    end = max(end, speech_end)

    if end - start > C.MAX_CLIP_SECONDS:
        raise DraftError(f"{label}: {fmt_ts(start)}–{fmt_ts(end)} is {end - start:.1f}s "
                         f"(> {C.MAX_CLIP_SECONDS:.0f}s). Pick a shorter span or trim with start_at/end_after.")
    return {"start": round(start, 3), "end": round(end, 3), "segments": out_segs,
            "start_id": a, "end_id": b}


SPAN_KEYS = ("start_id", "end_id", "start_at", "end_after", "title", "reason")
V1_SPAN_KEYS = ("start_id", "end_id", "start_at", "end_after")  # title/reason stay: they name the whole clip


def draft_spans(draft: dict) -> list[dict]:
    """Spans of a draft; a v1 draft (ids at the top level) is one span."""
    if "spans" in draft:
        return list(draft["spans"])
    return [{k: draft[k] for k in SPAN_KEYS if k in draft}]


def build(transcript: dict, draft: dict) -> dict:
    for key in ("instruction", "title", "reason"):
        if not str(draft.get(key) or "").strip():
            raise DraftError(f"draft.{key} is required")
    raw_spans = draft_spans(draft)
    if not raw_spans:
        raise DraftError("draft.spans must list at least one span")

    spans = []
    for i, sp in enumerate(raw_spans):
        built = build_span(transcript, sp, f"spans[{i}]")
        spans.append({"start": built["start"], "end": built["end"],
                      "title": sp.get("title") or (draft["title"] if len(raw_spans) == 1 else ""),
                      "reason": sp.get("reason") or (draft["reason"] if len(raw_spans) == 1 else ""),
                      "start_id": built["start_id"], "end_id": built["end_id"],
                      "segments": built["segments"]})
    spans.sort(key=lambda s: s["start"])  # chronological (hard cuts between them)
    for prev, cur in zip(spans, spans[1:]):
        if cur["start_id"] <= prev["end_id"] or cur["start"] < prev["end"]:
            raise DraftError(f"spans overlap: segments {prev['start_id']}–{prev['end_id']} and "
                             f"{cur['start_id']}–{cur['end_id']}. Merge them into one span or pick another part.")
    for sp in spans:
        if not sp["reason"].strip():
            raise DraftError(f"span {sp['start_id']}–{sp['end_id']}: reason is required when there are several spans")

    total = sum(s["end"] - s["start"] for s in spans)
    if total > C.MAX_CLIP_SECONDS:
        lens = ", ".join(f"{s['start_id']}–{s['end_id']}: {s['end'] - s['start']:.1f}s" for s in spans)
        raise DraftError(f"spans total {total:.1f}s (> {C.MAX_CLIP_SECONDS:.0f}s): {lens}. "
                         "Drop a span, pick shorter ones, or trim with start_at/end_after.")

    alts = []
    for i, alt in enumerate(draft.get("alternatives") or []):
        span = build_span(transcript, alt, f"alternatives[{i}]")
        alts.append({"start": span["start"], "end": span["end"], "title": alt.get("title", ""),
                     "reason": alt.get("reason", ""), "start_id": span["start_id"], "end_id": span["end_id"]})
    if len(alts) > C.MAX_ALTERNATIVES:
        raise DraftError(f"at most {C.MAX_ALTERNATIVES} alternatives")
    return {"version": 2, "instruction": draft["instruction"], "title": draft["title"],
            "reason": draft["reason"], "duration": round(total, 3), "spans": spans, "alternatives": alts}


def promote(draft: dict, n: int) -> str:
    """Checkpoint 2 helper. Returns a description of what changed."""
    alts = draft.get("alternatives") or []
    if not 0 <= n < len(alts):
        raise DraftError(f"no alternatives[{n}] in the draft")
    spans = draft_spans(draft)
    alt = alts.pop(n)
    if len(spans) == 1:  # swap: the old pick becomes an alternative
        old = spans[0]
        if "spans" not in draft:
            old = {**old, "title": draft["title"], "reason": draft["reason"]}
            draft["title"], draft["reason"] = alt.get("title") or draft["title"], alt.get("reason") or draft["reason"]
        alts.insert(n, old)
        spans = [{k: alt[k] for k in SPAN_KEYS if k in alt}]
        msg = f"swapped alternative {n} ({alt.get('title', '')!r}) in as the clip"
    else:
        spans.append({k: alt[k] for k in SPAN_KEYS if k in alt})
        msg = f"added alternative {n} ({alt.get('title', '')!r}) as a span"
    for k in V1_SPAN_KEYS:
        draft.pop(k, None)
    draft["spans"] = spans
    draft["alternatives"] = alts
    return msg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode_dir")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--promote", type=int, metavar="N",
                    help="one span: swap alternatives[N] in; several spans: add it as a span. Then rebuild")
    args = ap.parse_args()
    ep = episode_dir(args.episode_dir)
    tr_p, draft_p, out_p = ep / C.TRANSCRIPT, ep / C.CLIP_DRAFT, ep / C.CLIP
    for p in (tr_p, draft_p):
        if not p.exists():
            die(f"missing {p}")

    if args.promote is not None:
        draft = load_json(draft_p)
        try:
            msg = promote(draft, args.promote)
        except DraftError as e:
            die(str(e))
        save_json(draft_p, draft)
        args.force = True
        print(f"draft: {msg}")

    if not args.force and is_fresh(out_p, tr_p, draft_p):
        print(f"skip  {C.CLIP} (up to date)")
    else:
        try:
            clip = build(load_json(tr_p), load_json(draft_p))
        except DraftError as e:
            die(str(e))
        save_json(out_p, clip)
        print(f"make  {C.CLIP}  {len(clip['spans'])} span(s), {clip_duration(clip):.1f}s total")
        for i, sp in enumerate(clip["spans"]):
            print(f"      [{i}] {fmt_ts(sp['start'])}–{fmt_ts(sp['end'])} ({sp['end'] - sp['start']:.1f}s) "
                  f"segments {sp['start_id']}–{sp['end_id']}  {sp['title']}")

    errors, warnings = validate(ep)
    for w in warnings:
        print(f"warn  {w}")
    if errors:
        die("validate_clip failed:\n  " + "\n  ".join(errors))
    print("ok    clip.json validated")


if __name__ == "__main__":
    main()
