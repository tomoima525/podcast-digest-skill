#!/usr/bin/env python3
"""Preview the digest layout on one real frame, before rendering.

Usage: uv run scripts/preview_layout.py episodes/<slug> [--at SECONDS]

Writes into the episode directory:
  layout.preview.png  one 1080x1920 frame exactly as render.py draws it, with a sample caption
  layout.source.png   (stacked only) the source frame with the tiles outlined:
                      red = tiles[0] (top panel), blue = tiles[1] (bottom panel)

For a stacked episode, adjust "tiles" in episode.json until each panel shows one
person, roughly centred, with no frame edge or neighbouring tile showing, then
re-run. Tiles are fractions of the source frame ({x, y, w, h}); each is scaled and
centre-cropped to fill a 1080x860 panel, so a tile somewhat wider than 1.26:1
loses its left and right edges, never the faces in the middle.
--at defaults to the middle of the first span in clip.json, else 60s.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import config as C
import render as R
from common import StepError, clip_spans, die, episode_dir, load_json, media_duration, require_tool, run

SAMPLE_LINES = ["字幕はこの帯の中に入ります", "最大2行、1行16文字まで"]


def pick_time(ep: Path, src: Path) -> float:
    clip_p = ep / C.CLIP
    if clip_p.exists():
        spans = clip_spans(load_json(clip_p))
        if spans:
            return (spans[0]["start"] + spans[0]["end"]) / 2
    return min(60.0, media_duration(src) / 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode_dir")
    ap.add_argument("--at", type=float, metavar="SECONDS", help="source time of the frame to preview")
    args = ap.parse_args()
    ep = episode_dir(args.episode_dir)
    require_tool("ffmpeg")
    try:
        layout = R.load_layout(ep)
        src = ep / C.INPUT_VIDEO
        stacked = layout["layout"] == "stacked"
        if stacked and not (src.exists() and R.has_video_stream(src)):
            die(f"layout 'stacked' needs a video {src}")
        at = args.at if args.at is not None else (pick_time(ep, src) if src.exists() else 0.0)

        ass = ep / ".layout.preview.ass"
        R.write_ass([{"start": 0.0, "end": 60.0, "lines": SAMPLE_LINES}], ass, "layout preview", layout)
        out = ep / C.LAYOUT_PREVIEW
        if stacked:
            graph = (R.stacked_filter("[0:v]", "[vl]", layout["tiles"]) + f";[vl]{R.subtitles_filter(ass)}[v]")
            run(["ffmpeg", "-y", "-hide_banner", "-v", "error", "-ss", f"{at:.3f}", "-i", str(src),
                 "-filter_complex", graph, "-map", "[v]", "-frames:v", "1", str(out)], cwd=ep)
            boxes = ",".join(
                f"drawbox=x=iw*{t['x']}:y=ih*{t['y']}:w=iw*{t['w']}:h=ih*{t['h']}:color={color}@0.9:t=8"
                for t, color in zip(layout["tiles"], ("red", "blue")))
            run(["ffmpeg", "-y", "-hide_banner", "-v", "error", "-ss", f"{at:.3f}", "-i", str(src),
                 "-vf", boxes, "-frames:v", "1", str(ep / "layout.source.png")], cwd=ep)
            size = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0:s=x", str(src)]).stdout.strip()
            w, h = (int(v) for v in size.split("x"))
            print(f"source {w}x{h}, frame at {at:.2f}s")
            for name, t in zip(("top", "bottom"), layout["tiles"]):
                print(f"  {name:6} tile {t}  = {round(t['w'] * w)}x{round(t['h'] * h)} px "
                      f"at ({round(t['x'] * w)}, {round(t['y'] * h)})")
            print(f"wrote  {out.name}, layout.source.png")
        else:
            if not C.BACKGROUND_IMAGE.exists():
                die(f"missing background image {C.BACKGROUND_IMAGE}")
            graph = f"[0:v]{R.background_filter()},setsar=1,format=yuv420p,{R.subtitles_filter(ass)}[v]"
            run(["ffmpeg", "-y", "-hide_banner", "-v", "error", "-i", str(C.BACKGROUND_IMAGE),
                 "-filter_complex", graph, "-map", "[v]", "-frames:v", "1", str(out)], cwd=ep)
            print(f"wrote  {out.name} (background layout)")
        ass.unlink(missing_ok=True)
    except StepError as e:
        die(str(e))


if __name__ == "__main__":
    main()
