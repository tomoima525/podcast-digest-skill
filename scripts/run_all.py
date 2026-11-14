#!/usr/bin/env python3
"""Orchestrator: run every deterministic step that is ready, stop at agent/human steps.

Usage: uv run scripts/run_all.py episodes/<slug> [--render] [--force]

  1a transcribe.py            (deterministic)          -> audio.mp3, transcript.raw.json
  1b corrections              (AGENT, then CHECKPOINT 1) -> corrections.md -> transcript.json
  2  clip selection           (AGENT, needs the human's instruction, then CHECKPOINT 2)
                                                        -> clip.draft.json -> clip.json
  3  render.py                (deterministic, only with --render after approval)

input.mp4 is required (an audio-only file is fine for the background layout);
shownote.txt is recommended. episode.json picks the layout (see render.py).

Safe to re-run at any time: finished steps are skipped, derived files are
rebuilt when their inputs changed. Exit codes: 0 = digest rendered,
10 = paused waiting for the agent/human (message says what is needed), 1 = error.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import config as C
from common import StepError, episode_dir
from render import load_layout

SCRIPTS = Path(__file__).resolve().parent
PAUSED = 10


def step(script: str, ep: Path, *extra: str) -> None:
    print(f"\n== {script}", flush=True)
    proc = subprocess.run([sys.executable, str(SCRIPTS / script), str(ep), *extra])
    if proc.returncode != 0:
        print(f"\nFAILED: {script} (exit {proc.returncode})", file=sys.stderr)
        sys.exit(1)


def pause(msg: str) -> None:
    print(f"\n== PAUSED\n{msg}", flush=True)
    sys.exit(PAUSED)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode_dir")
    ap.add_argument("--render", action="store_true", help="render digest.mp4 (only after checkpoint 2 approval)")
    ap.add_argument("--force", action="store_true", help="pass --force to the deterministic steps")
    args = ap.parse_args()
    ep = episode_dir(args.episode_dir)
    force = ("--force",) if args.force else ()
    rel = lambda p: p.name  # noqa: E731

    if not (ep / C.INPUT_VIDEO).exists() and not (ep / C.AUDIO).exists():
        print(f"error: {ep / C.INPUT_VIDEO} is missing", file=sys.stderr)
        sys.exit(1)
    if not (ep / C.SHOWNOTE).exists() and not (ep / C.TRANSCRIPT_RAW).exists():
        print(f"warn  no {C.SHOWNOTE}: whisper gets no hint for names and terms, and step 1b has nothing "
              f"to check spellings against. Add one (cast, product names, topics) for better results.", flush=True)
    try:
        layout = load_layout(ep)["layout"]
    except StepError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    step("transcribe.py", ep, *force)

    # Step 1b ---------------------------------------------------------------------------
    if not (ep / C.CORRECTIONS).exists():
        pause(f"Step 1b (agent): correct {rel(ep / C.TRANSCRIPT_RAW)} using {C.SHOWNOTE}.\n"
              f"  Write {C.CORRECTIONS}, then run scripts/apply_corrections.py {ep}\n"
              f"  Then CHECKPOINT 1: ask the human to review {C.CORRECTIONS}.")
    step("apply_corrections.py", ep)

    # Step 2 ------------------------------------------------------------------------------
    draft, clip = ep / C.CLIP_DRAFT, ep / C.CLIP
    if not draft.exists() and not clip.exists():
        pause("Step 2 (agent): needs the human's instruction (which part to feature).\n"
              f"  Write {C.CLIP_DRAFT}, then run scripts/build_clip.py {ep}\n"
              "  Then CHECKPOINT 2: show the clip + alternatives and ask the human to approve or pick one.")
    if draft.exists():
        step("build_clip.py", ep)
    else:
        step("validate_clip.py", ep)

    # Step 3 ---------------------------------------------------------------------------------
    if not args.render:
        state = "rendered (re-run with --render to refresh)" if (ep / C.DIGEST_VIDEO).exists() else "not rendered yet"
        hint = ""
        if layout == "stacked" and not (ep / C.LAYOUT_PREVIEW).exists():
            hint = (f"\n  Layout is 'stacked': run scripts/preview_layout.py {ep} and check "
                    f"{C.LAYOUT_PREVIEW} before rendering.")
        pause(f"CHECKPOINT 2: clip.json is valid; digest.mp4 is {state}. Layout: {layout}.{hint}\n"
              f"  After the human approves, run: scripts/run_all.py {ep} --render")
    step("render.py", ep, *force)
    print(f"\n== DONE  {ep / C.DIGEST_VIDEO}")


if __name__ == "__main__":
    main()
