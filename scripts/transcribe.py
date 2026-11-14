#!/usr/bin/env python3
"""Step 1a: input.mp4 -> audio.mp3 -> transcript.raw.json (OpenAI whisper-1).

Usage: uv run scripts/transcribe.py episodes/<slug> [--force] [--print-prompt]

- Extracts mono 16 kHz low-bitrate mp3.
- If audio.mp3 exceeds the upload limit, splits it into overlapping chunks,
  transcribes each, offsets timestamps and de-duplicates the overlaps.
- Builds a vocabulary prompt from shownote.txt (override with prompt.txt).
- Per-chunk API responses are cached in chunks/ so a failed run resumes cheaply.
- whisper-1 occasionally skips a stretch of speech. Long pauses between recognised
  words that still contain sound are re-sent on their own and the recovered segments
  inserted ("gap" lines in the output).

--fill-gaps: run only that last step on an existing transcript.raw.json (cheap: only
the gaps are uploaded). Segment ids after a recovered gap shift; the script prints
how, so corrections.md / clip.draft.json can be renumbered.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path

import config as C
from common import (StepError, align_text, detect_silences, die, episode_dir, fmt_ts, is_fresh, load_env,
                    load_json, media_duration, require_tool, run, save_json)

# --- Audio -------------------------------------------------------------------------

def extract_audio(ep: Path, force: bool) -> Path:
    src, dst = ep / C.INPUT_VIDEO, ep / C.AUDIO
    if not src.exists() and not dst.exists():
        die(f"missing {src}")
    if not force and (dst.exists() and (not src.exists() or is_fresh(dst, src))):
        print(f"skip  {C.AUDIO} (exists)")
        return dst
    print(f"make  {C.AUDIO}")
    tmp = dst.with_name("audio.tmp.mp3")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1",
         "-ar", str(C.AUDIO_SAMPLE_RATE), "-c:a", "libmp3lame", "-b:a", C.AUDIO_BITRATE, str(tmp)])
    os.replace(tmp, dst)
    return dst


# --- Prompt ------------------------------------------------------------------------

_URL = re.compile(r"https?://\S+|www\.\S+")
_ASCII_TERM = re.compile(r"[A-Za-z][\w.+#&'\-]*(?: [A-Z0-9][\w.+#&'\-]*)*", re.ASCII)
_KATAKANA = re.compile(r"[ァ-ヴヵヶー・]{3,}")
_BRACKETED = re.compile(r"[「『【]([^」』】]{2,12})[」』】]")
# "出演: 今井、佐藤" / "ゲスト：山田太郎さん" -> people's names (kanji names are the usual misses)
_CAST_LINE = re.compile(r"^\s*[-・*]?\s*(?:出演|ゲスト|パーソナリティ|ホスト|MC|話し手)\s*[:：]\s*(.+)$", re.M)
_STOP = {"the", "and", "or", "of", "to", "in", "on", "for", "a", "an", "is", "with",
         "http", "https", "www", "com", "jp", "ep", "episode", "podcast", "url",
         # common shownote headings, not vocabulary
         "トピック", "リンク", "ゲスト", "パーソナリティ", "エピソード", "ポッドキャスト", "ショーノート",
         "チャプター", "メンバー", "スポンサー", "ハッシュタグ", "フィードバック", "プレゼント"}


def extract_terms(shownote: str) -> list[str]:
    text = _URL.sub(" ", shownote)
    found: list[tuple[int, str]] = []
    for m in _CAST_LINE.finditer(text):  # names first: they are the most valuable terms
        for name in re.split(r"[、,，/／&＆と\s]+", m.group(1)):
            name = re.sub(r"(さん|氏|様)$", "", name.strip(" ()（）"))
            if name:
                found.append((-1, name))
    for rx, grp in ((_BRACKETED, 1), (_ASCII_TERM, 0), (_KATAKANA, 0)):
        for m in rx.finditer(text):
            term = m.group(grp).strip(" .-'・")
            if len(term) >= 2 and term.lower() not in _STOP and not term.isdigit():
                found.append((m.start(grp), term))
    found.sort()
    seen, out = set(), []
    for _, term in found:
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            out.append(term)
    return out


def build_prompt(ep: Path) -> str:
    override = ep / "prompt.txt"
    if override.exists():
        return override.read_text(encoding="utf-8").strip()[: C.PROMPT_MAX_CHARS]
    note = ep / C.SHOWNOTE
    terms = extract_terms(note.read_text(encoding="utf-8")) if note.exists() else []
    prompt = "日本語のポッドキャストです。"
    if terms:
        prompt += "用語: "
        kept: list[str] = []
        for t in terms:
            if len(prompt) + len("、".join(kept + [t])) + 1 > C.PROMPT_MAX_CHARS:
                break
            kept.append(t)
        prompt += "、".join(kept) + "。"
    return prompt


# --- Chunking ----------------------------------------------------------------------

def plan_chunks(size_bytes: int, duration: float, max_bytes: int, overlap: float) -> list[tuple[float, float]]:
    """[(start, end)] covering [0, duration]; each chunk stays under max_bytes."""
    if size_bytes <= max_bytes:
        return [(0.0, duration)]
    bytes_per_sec = size_bytes / duration
    max_len = (max_bytes * 0.95) / bytes_per_sec
    step = max_len - overlap
    if step <= 0:
        raise StepError("chunk overlap is larger than the chunk length")
    n = math.ceil((duration - overlap) / step)
    step = (duration - overlap) / n  # spread evenly
    return [(i * step, min(duration, i * step + step + overlap)) for i in range(n)]


def assign_words(segments: list[dict], words: list[dict]) -> None:
    """Attach top-level `words` to the segment they fall in (by word midpoint)."""
    for seg in segments:
        seg["words"] = []
    if not segments:
        return
    j = 0
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        while j + 1 < len(segments) and mid >= segments[j + 1]["start"]:
            j += 1
        segments[j]["words"].append(w)


def normalize_response(resp: dict, offset: float) -> list[dict]:
    """API verbose_json (chunk-relative) -> segments with absolute times and words."""
    r3 = lambda x: round(float(x) + offset, 3)  # noqa: E731
    segs = [{"start": r3(s["start"]), "end": r3(s["end"]), "text": (s.get("text") or "").strip()}
            for s in resp.get("segments") or []]
    words = [{"word": w["word"], "start": r3(w["start"]), "end": r3(w["end"])}
             for w in resp.get("words") or []]
    assign_words(segs, words)
    return segs


def _pick_cut(segs: list[dict], lo: float, hi: float) -> float:
    """A cut time inside [lo, hi]: the middle of the longest pause between words."""
    words = [w for s in segs for w in s["words"]]
    best, best_gap = (lo + hi) / 2, -1.0
    for a, b in zip(words, words[1:]):
        if a["end"] >= lo and b["start"] <= hi and b["start"] - a["end"] > best_gap:
            best, best_gap = (a["end"] + b["start"]) / 2, b["start"] - a["end"]
    return best


def _split_segment(seg: dict, cut: float, keep: str) -> dict | None:
    """Part of `seg` before (keep='head') or after (keep='tail') the cut, at a word boundary."""
    words = seg["words"]
    if not words:
        mid = (seg["start"] + seg["end"]) / 2
        return seg if (mid < cut) == (keep == "head") else None
    k = next((i for i, w in enumerate(words) if w["start"] >= cut), len(words))
    if (keep == "head" and k == len(words)) or (keep == "tail" and k == 0):
        return {**seg, "_span": (seg["start"], seg["end"])}  # entirely on the kept side
    _, offsets = align_text(seg["text"], words, seg["start"], seg["end"])
    off = offsets[k] if k < len(words) else len(seg["text"])
    if keep == "head":
        if k == 0:
            return None
        return {**seg, "end": words[k - 1]["end"], "text": seg["text"][:off], "words": words[:k],
                "_cut": True, "_span": (seg["start"], seg["end"])}
    if k == len(words):
        return None
    return {**seg, "start": words[k]["start"], "text": seg["text"][off:], "words": words[k:],
            "_cut": True, "_span": (seg["start"], seg["end"])}


def merge_chunks(chunks: list[tuple[float, float, list[dict]]]) -> list[dict]:
    """De-duplicate overlapping chunk transcripts at word level.

    For each overlap, cut at the longest pause between words (well inside the
    overlap, away from both chunk edges where recognition is unreliable). Keep
    the earlier chunk before the cut and the later chunk after it. A sentence
    that straddles the cut is rebuilt from both halves, so even a segment
    longer than the overlap is not lost.
    """
    merged = [dict(s) for s in chunks[0][2]]
    for (_, prev_end, _), (cur_start, _, segs) in zip(chunks, chunks[1:]):
        cut = _pick_cut(merged, cur_start + 1.0, prev_end - 1.0)
        head = [p for s in merged if s["start"] < cut and (p := _split_segment(s, cut, "head")) is not None]
        tail = [p for s in segs if s["end"] > cut and (p := _split_segment(s, cut, "tail")) is not None]
        # Both chunks heard the same utterance (original spans overlap) and at least
        # one copy was cut: stitch the halves back into one segment.
        if (head and tail and (head[-1].get("_cut") or tail[0].get("_cut"))
                and tail[0].get("_span", (math.inf,))[0] < head[-1].get("_span", (0, -math.inf))[1]):
            a, b = head.pop(), tail.pop(0)
            head.append({**a, "end": b["end"], "text": a["text"] + b["text"], "words": a["words"] + b["words"]})
        merged = head + tail
    for s in merged:
        s.pop("_cut", None)
        s.pop("_span", None)
    return merged


def transcribe_file(client, path: Path, prompt: str) -> dict:
    for attempt in range(4):
        try:
            with open(path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model=C.WHISPER_MODEL, file=f, language=C.LANGUAGE, prompt=prompt,
                    response_format="verbose_json", timestamp_granularities=["segment", "word"])
            return resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        except Exception as e:  # network / 5xx: back off and retry
            if attempt == 3:
                raise
            wait = 5 * 2 ** attempt
            print(f"      API error ({e.__class__.__name__}: {e}); retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable")


# --- Gaps whisper skipped -----------------------------------------------------------

def missed_gaps(segments: list[dict], duration: float, silences: list[tuple[float, float]],
                min_gap: float = C.GAP_MIN_SECONDS, min_speech: float = C.GAP_SPEECH_SECONDS
                ) -> list[tuple[float, float]]:
    """Pauses between recognised words (or before the first / after the last word) of
    at least `min_gap` seconds that still hold `min_speech` seconds of non-silence."""
    times = sorted((w["start"], w["end"]) for s in segments for w in s["words"])
    bounds, prev_end = [], 0.0
    for a, b in times + [(duration, duration)]:
        if a - prev_end >= min_gap:
            bounds.append((prev_end, a))
        prev_end = max(prev_end, b)
    out = []
    for a, b in bounds:
        quiet = sum(max(0.0, min(b, y) - max(a, x)) for x, y in silences)
        if (b - a) - quiet >= min_speech:
            out.append((a, b))
    return out


def recover_gap(resp: dict, a: float, b: float) -> list[dict]:
    """Segments of a gap's own transcription, clipped to the gap."""
    out = []
    for seg in normalize_response(resp, a):
        words = [w for w in seg["words"] if a - 0.05 <= (w["start"] + w["end"]) / 2 <= b + 0.05]
        if not seg["text"] or (seg["words"] and not words):
            continue
        if any(h in seg["text"] for h in C.GAP_HALLUCINATIONS):
            continue
        for w in words:
            w["start"], w["end"] = max(w["start"], a), min(w["end"], b)
        start = words[0]["start"] if words else max(seg["start"], a)
        end = words[-1]["end"] if words else min(seg["end"], b)
        if end > start:
            out.append({"start": start, "end": end, "text": seg["text"], "words": words})
    return out


def fill_gaps(client, ep: Path, audio: Path, segments: list[dict], duration: float, prompt: str,
              force: bool) -> tuple[list[dict], list[tuple[float, float, int]]]:
    """Re-transcribe missed stretches. Returns (segments, [(gap_start, gap_end, n_recovered)])."""
    silences = detect_silences(audio, C.GAP_SILENCE_DB, C.GAP_SILENCE_MIN)
    gaps = missed_gaps(segments, duration, silences)
    cache = ep / C.CHUNKS_DIR
    added: list[dict] = []
    report = []
    for a, b in gaps:
        cached = cache / f"gap_{round(a * 1000):09d}.json"
        meta = {"start": a, "end": b, "prompt": prompt}
        if cached.exists() and not force and load_json(cached).get("meta") == meta:
            resp = load_json(cached)["response"]
        else:
            cache.mkdir(exist_ok=True)
            upload = cache / "gap.mp3"
            run(["ffmpeg", "-y", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(audio),
                 "-ac", "1", "-ar", str(C.AUDIO_SAMPLE_RATE), "-c:a", "libmp3lame", "-b:a", C.AUDIO_BITRATE,
                 str(upload)])
            resp = transcribe_file(client, upload, prompt)
            upload.unlink(missing_ok=True)
            save_json(cached, {"meta": meta, "response": resp})
        segs = recover_gap(resp, a, b)
        added += segs
        report.append((a, b, len(segs)))
        if segs:
            print(f"      gap  {fmt_ts(a)}–{fmt_ts(b)}: recovered {len(segs)} segment(s) whisper had skipped")
        else:
            print(f"warn  gap  {fmt_ts(a)}–{fmt_ts(b)}: sound but no speech recognised (music/noise?)")
    return sorted(segments + added, key=lambda s: s["start"]), report


def build_raw(segments: list[dict], duration: float, prompt: str) -> dict:
    out = []
    for seg in segments:
        if not seg["text"] and not seg["words"]:
            continue
        out.append({"id": len(out), "start": seg["start"], "end": seg["end"], "text": seg["text"],
                    "original_text": seg["text"], "words": seg["words"]})
    return {"language": C.LANGUAGE, "duration": round(duration, 3), "model": C.WHISPER_MODEL,
            "prompt": prompt, "segments": out}


def transcribe(ep: Path, audio: Path, prompt: str, force: bool) -> dict:
    client = openai_client()
    duration = media_duration(audio)
    plan = plan_chunks(audio.stat().st_size, duration, C.MAX_UPLOAD_BYTES, C.CHUNK_OVERLAP_SECONDS)
    cache = ep / C.CHUNKS_DIR
    results = []
    for i, (start, end) in enumerate(plan):
        cached = cache / f"chunk_{i:03d}.json"
        if len(plan) > 1:
            print(f"      chunk {i + 1}/{len(plan)}  {start:.1f}s–{end:.1f}s")
        meta = {"start": start, "end": end, "prompt": prompt}
        if cached.exists() and not force and load_json(cached).get("meta") == meta:
            resp = load_json(cached)["response"]
        else:
            if len(plan) == 1:
                upload = audio
            else:
                cache.mkdir(exist_ok=True)
                upload = cache / f"chunk_{i:03d}.mp3"
                run(["ffmpeg", "-y", "-v", "error", "-i", str(audio), "-ss", f"{start:.3f}",
                     "-t", f"{end - start:.3f}", "-ac", "1", "-ar", str(C.AUDIO_SAMPLE_RATE),
                     "-c:a", "libmp3lame", "-b:a", C.AUDIO_BITRATE, str(upload)])
            resp = transcribe_file(client, upload, prompt)
            cache.mkdir(exist_ok=True)
            save_json(cached, {"meta": meta, "response": resp})
        results.append((start, end, normalize_response(resp, start)))
    segments, _ = fill_gaps(client, ep, audio, merge_chunks(results), duration, prompt, force)
    return build_raw(segments, duration, prompt)


def openai_client():
    if not load_env("OPENAI_API_KEY"):
        die(f"OPENAI_API_KEY is not set: add it to {C.ENV_FILE} (see .env.example) or export it")
    from openai import OpenAI  # imported lazily so other scripts need no deps
    return OpenAI(timeout=900, max_retries=2)


def fill_existing(ep: Path, audio: Path, force: bool) -> None:
    """--fill-gaps on an existing transcript.raw.json; prints how segment ids shift."""
    out = ep / C.TRANSCRIPT_RAW
    if not out.exists():
        die(f"missing {out}: run transcribe.py first")
    raw = load_json(out)
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"], "words": s["words"], "_id": s["id"]}
            for s in raw["segments"]]
    filled, report = fill_gaps(openai_client(), ep, audio, segs, raw["duration"], raw["prompt"], force)
    if not any(n for _, _, n in report):
        print(f"ok    {C.TRANSCRIPT_RAW}: no missed speech found")
        return
    new = build_raw(filled, raw["duration"], raw["prompt"])
    shifts, k = [], 0
    for seg, src in zip(new["segments"], [s for s in filled if s["text"] or s["words"]]):
        if "_id" not in src:
            k += 1
        elif k and (not shifts or shifts[-1][1] != k):
            shifts.append((src["_id"], k))
    save_json(out, new)
    print(f"make  {C.TRANSCRIPT_RAW}  {len(raw['segments'])} -> {len(new['segments'])} segments")
    for old_id, by in shifts:
        print(f"      ids >= {old_id} (old) are now +{by}")
    if (ep / C.CORRECTIONS).exists() or (ep / C.CLIP_DRAFT).exists():
        print(f"note  renumber the ids above in {C.CORRECTIONS} / {C.CLIP_DRAFT}, then re-run apply_corrections.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode_dir")
    ap.add_argument("--force", action="store_true", help="redo audio extraction and transcription")
    ap.add_argument("--print-prompt", action="store_true", help="print the whisper prompt and exit")
    ap.add_argument("--fill-gaps", action="store_true",
                    help="only re-transcribe speech whisper skipped in an existing transcript.raw.json")
    args = ap.parse_args()
    ep = episode_dir(args.episode_dir)

    prompt = build_prompt(ep)
    if args.print_prompt:
        print(prompt)
        return
    require_tool("ffmpeg")
    require_tool("ffprobe")
    try:
        audio = extract_audio(ep, args.force and not args.fill_gaps)
        if args.fill_gaps:
            fill_existing(ep, audio, args.force)
            return
        out = ep / C.TRANSCRIPT_RAW
        if not args.force and is_fresh(out, audio):
            print(f"skip  {C.TRANSCRIPT_RAW} (exists)")
            return
        if not load_env("OPENAI_API_KEY"):  # fail before announcing any work
            die(f"OPENAI_API_KEY is not set: add it to {C.ENV_FILE} (see .env.example) or export it")
        print(f"make  {C.TRANSCRIPT_RAW}  (prompt: {prompt})", flush=True)
        raw = transcribe(ep, audio, prompt, args.force)
        save_json(out, raw)
        print(f"      {len(raw['segments'])} segments, {raw['duration']:.1f}s")
    except StepError as e:
        die(str(e))


if __name__ == "__main__":
    main()
