#!/usr/bin/env python3
"""End-to-end tests on tests/fixtures (2 min synthetic Japanese episode).

Usage:
  uv run tests/e2e.py                 # offline: podcast, then stacked video
  uv run tests/e2e.py --only video    # just one of them (podcast | video)
  uv run tests/e2e.py --live          # step 1a via the real whisper-1 API (needs OPENAI_API_KEY)

Offline mode replays the committed raw transcript and the agent's recorded
outputs for steps 1b/2 and checks that every pause, validator, the render and
caption sync (±0.2s vs ground truth) behave.
- podcast (tests/.work/sample): background layout, single-span v1 draft.
- video (tests/.work/video): a generated StreamYard-like 2-up recording
  (tests/make_video_fixture.py), episode.json layout "stacked", a three-span
  draft (fixtures/video/expected/clip.draft.json) joined with hard cuts; also
  checks the panels, the duration and picture sync (flash boxes).
--live runs real transcription and stops at the step-1b pause: the raw text and
segment ids differ from the recording, so steps 1b/2 must then be done by the
agent (follow SKILL.md), followed by `uv run tests/check_sync.py`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures" / "sample"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "tests"))


def sh(script: str, *args: str, expect: int = 0) -> str:
    proc = subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    print(f"$ {script} {' '.join(Path(a).name if '/' in a else a for a in args)}  -> exit {proc.returncode}")
    if proc.returncode != expect:
        print(out)
        sys.exit(f"FAIL: {script} exited {proc.returncode}, expected {expect}")
    return out


def fresh_dir(name: str) -> Path:
    ep = ROOT / "tests" / ".work" / name
    if ep.exists():
        shutil.rmtree(ep)
    ep.mkdir(parents=True)
    return ep


def check_output(ep: Path) -> None:
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,width,height,r_frame_rate",
                            "-of", "json", str(ep / "digest.mp4")], capture_output=True, text=True, check=True)
    streams = json.loads(probe.stdout)["streams"]
    v = next(s for s in streams if s["codec_name"] == "h264")
    assert (v["width"], v["height"], v["r_frame_rate"]) == (1080, 1920, "30/1"), v
    assert any(s["codec_name"] == "aac" for s in streams), streams

    loud = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(ep / "digest.mp4"), "-af",
                           "ebur128=framelog=quiet", "-f", "null", "-"], capture_output=True, text=True).stderr
    lufs = float(loud[loud.rfind("Summary:"):].split("I:")[1].split("LUFS")[0])
    print(f"loudness: {lufs} LUFS")
    assert abs(lufs + 14) <= 0.5, lufs


def check_sync(ep: Path, *extra: str) -> None:
    sync = subprocess.run([sys.executable, str(ROOT / "tests" / "check_sync.py"), str(ep),
                           str(FIX / "ground_truth.json"), *extra], capture_output=True, text=True)
    for line in sync.stdout.strip().splitlines():
        if "checked" in line:
            print(line)
    if sync.returncode != 0:
        print(sync.stdout)
        sys.exit("FAIL: out of sync")


def frame_rgb(video: Path, at: float, x: int, y: int) -> tuple[int, int, int]:
    """Average colour of a 20x20 patch of the 1080x1920 frame at `at` seconds."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1",
                          "-vf", f"crop=20:20:{x - 10}:{y - 10},scale=1:1:flags=area", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return raw[0], raw[1], raw[2]


def video_e2e() -> None:
    print("\n### video: stacked layout, three spans")
    import make_video_fixture
    import transcribe
    from render import span_plan
    vfix = ROOT / "tests" / "fixtures" / "video"
    ep = fresh_dir("video")
    e = str(ep)
    make_video_fixture.build(ep / "input.mp4")
    (ep / "episode.json").write_text('{"layout": "stacked"}\n', encoding="utf-8")  # default StreamYard tiles
    transcribe.extract_audio(ep, force=False)
    shutil.copy(FIX / "transcript.raw.json", ep / "transcript.raw.json")
    shutil.copy(FIX / "expected" / "corrections.md", ep / "corrections.md")

    out = sh("run_all.py", e, expect=10)
    assert "Step 2" in out, out
    draft = json.loads((vfix / "expected" / "clip.draft.json").read_text(encoding="utf-8"))
    assert draft["instruction"] == (vfix / "instruction.txt").read_text(encoding="utf-8").strip()
    shutil.copy(vfix / "expected" / "clip.draft.json", ep / "clip.draft.json")
    out = sh("run_all.py", e, expect=10)
    assert "CHECKPOINT 2" in out and "3 span(s)" in out and "preview_layout.py" in out, out
    sh("preview_layout.py", e)
    assert (ep / "layout.preview.png").exists() and (ep / "layout.source.png").exists()
    out = sh("run_all.py", e, expect=10)
    assert "preview_layout.py" not in out, out  # hint gone once previewed

    sh("run_all.py", e, "--render")
    out = sh("run_all.py", e, "--render")
    assert "skip  digest.mp4" in out, out
    check_output(ep)

    clip = json.loads((ep / "clip.json").read_text(encoding="utf-8"))
    want = sum(d for _, d in span_plan(clip))
    got = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(ep / "digest.mp4")], capture_output=True, text=True, check=True).stdout)
    print(f"duration: {got:.3f}s (spans {want:.3f}s)")
    assert abs(got - want) <= 0.05, (got, want)

    # top panel = left tile (teal), band black, bottom panel = right tile (orange)
    top, band, bottom = (frame_rgb(ep / "digest.mp4", 2.0, 540, y) for y in (100, 960, 1820))
    print(f"panels: top {top}, band {band}, bottom {bottom}")
    assert top[1] > 100 and top[2] > 100 and top[0] < 40, top
    assert max(band) < 30, band
    assert bottom[0] > 180 and bottom[2] < 80, bottom
    check_sync(ep, "--flashes")
    print(f"video E2E OK  ->  {ep / 'digest.mp4'}")


def podcast_e2e(live: bool) -> None:
    print("\n### podcast: background layout, one span")
    ep = fresh_dir("sample")
    for f in ("input.mp4", "shownote.txt"):
        shutil.copy(FIX / f, ep / f)
    e = str(ep)

    if live:
        out = sh("run_all.py", e, expect=10)
        assert "Step 1b" in out, out
        raw = json.loads((ep / "transcript.raw.json").read_text(encoding="utf-8"))
        print(f"whisper-1: {len(raw['segments'])} segments. Now do steps 1b and 2 with the agent (SKILL.md) in {ep},")
        print(f"then: uv run scripts/run_all.py {ep} --render && uv run tests/check_sync.py {ep} {FIX / 'ground_truth.json'}")
        return

    import transcribe
    transcribe.extract_audio(ep, force=False)
    shutil.copy(FIX / "transcript.raw.json", ep / "transcript.raw.json")  # newer than audio.mp3 -> step 1a skips

    out = sh("run_all.py", e, expect=10)
    assert "skip  transcript.raw.json" in out and "Step 1b" in out, out

    shutil.copy(FIX / "expected" / "corrections.md", ep / "corrections.md")  # agent: step 1b
    out = sh("run_all.py", e, expect=10)
    assert "Step 2" in out and "needs the human's instruction" in out, out
    sh("validate_transcript.py", e)

    draft = json.loads((FIX / "expected" / "clip.draft.json").read_text(encoding="utf-8"))
    assert draft["instruction"] == (FIX / "instruction.txt").read_text(encoding="utf-8").strip()
    shutil.copy(FIX / "expected" / "clip.draft.json", ep / "clip.draft.json")  # agent: step 2
    out = sh("run_all.py", e, expect=10)
    assert "CHECKPOINT 2" in out, out
    sh("validate_clip.py", e)

    sh("run_all.py", e, "--render")  # human approved
    out = sh("run_all.py", e, "--render")  # idempotent re-run
    assert "skip  digest.mp4" in out, out

    check_output(ep)
    check_sync(ep)
    print(f"podcast E2E OK  ->  {ep / 'digest.mp4'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="podcast with real whisper-1; stops at step 1b")
    ap.add_argument("--only", choices=("podcast", "video"))
    args = ap.parse_args()
    if args.only != "video":
        podcast_e2e(args.live)
        if args.live:
            return
    if args.only != "podcast":
        video_e2e()
    print("\nE2E OK")


if __name__ == "__main__":
    main()
