#!/usr/bin/env python3
"""TEST ONLY: offline stand-in for step 1a using a local openai-whisper model.

Produces transcript.raw.json exactly like scripts/transcribe.py (same audio
extraction, prompt, chunk planning, overlap merge and normalisation), but the
recognition runs locally so tests work without OPENAI_API_KEY. Production
always uses whisper-1 via scripts/transcribe.py.

Usage (openai-whisper + PyTorch go into a throwaway env, not .venv):
  uv run --with openai-whisper tests/local_transcribe.py episodes/<slug> [--model small] [--chunk-bytes N]

--chunk-bytes forces the chunked path (e.g. 300000) to exercise overlap merging.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import config as C  # noqa: E402
import transcribe as T  # noqa: E402
from common import media_duration, run, save_json  # noqa: E402


def to_verbose_json(result: dict) -> dict:
    """openai-whisper result -> the OpenAI API verbose_json shape."""
    segs, words = [], []
    for s in result["segments"]:
        segs.append({"start": s["start"], "end": s["end"], "text": s["text"]})
        for w in s.get("words", []):
            words.append({"word": w["word"].strip(), "start": w["start"], "end": w["end"]})
    return {"segments": segs, "words": words}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--model", default="small")
    ap.add_argument("--chunk-bytes", type=int, default=C.MAX_UPLOAD_BYTES)
    ap.add_argument("--out", default=C.TRANSCRIPT_RAW)
    args = ap.parse_args()
    import whisper

    ep = Path(args.episode_dir).resolve()
    audio = T.extract_audio(ep, force=False)
    prompt = T.build_prompt(ep)
    model = whisper.load_model(args.model)
    duration = media_duration(audio)
    plan = T.plan_chunks(audio.stat().st_size, duration, args.chunk_bytes, C.CHUNK_OVERLAP_SECONDS)
    print(f"prompt: {prompt}\nchunks: {[(round(a, 1), round(b, 1)) for a, b in plan]}")
    results = []
    with tempfile.TemporaryDirectory() as td:
        for i, (start, end) in enumerate(plan):
            part = Path(td) / f"c{i}.mp3"
            run(["ffmpeg", "-y", "-v", "error", "-i", str(audio), "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                 "-c:a", "libmp3lame", "-b:a", C.AUDIO_BITRATE, str(part)])
            res = model.transcribe(str(part), language=C.LANGUAGE, initial_prompt=prompt,
                                   word_timestamps=True, temperature=0.0, fp16=False)
            results.append((start, end, T.normalize_response(to_verbose_json(res), start)))
    raw = T.build_raw(T.merge_chunks(results), duration, prompt)
    raw["model"] = f"local-whisper-{args.model} (test stand-in for {C.WHISPER_MODEL})"
    save_json(ep / args.out, raw)
    print(f"wrote {ep / args.out}: {len(raw['segments'])} segments")


if __name__ == "__main__":
    main()
