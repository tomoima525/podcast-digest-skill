#!/usr/bin/env python3
"""Step 1b helper: transcript.raw.json + corrections.md -> transcript.json.

Usage: uv run scripts/apply_corrections.py episodes/<slug> [--force]

corrections.md holds one markdown table (any extra prose is ignored):

    | id | before | after | reason |
    |----|--------|-------|--------|
    | 3, 17 | クロードコード | Claude Code | product name in shownote |
    | 42 | 生成AIA | 生成AI | trailing noise |

- `id`: one or more segment ids (comma separated; ranges like 10-12 allowed).
- `before`: exact substring of that segment's text; every occurrence in the
  segment is replaced by `after` (which may be empty). Escape a literal | as \\|.
- Rows apply in order. Only `text` changes; timings and words are untouched.

Regenerates transcript.json whenever corrections.md is newer (or with --force),
then runs validate_transcript.
"""

from __future__ import annotations

import argparse
import copy

import config as C
from common import die, episode_dir, is_fresh, iter_md_table_rows, load_json, save_json
from validate_transcript import validate


def parse_ids(cell: str) -> list[int]:
    ids: list[int] = []
    for part in cell.replace("、", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    return ids


def parse_corrections(md: str) -> list[tuple[list[int], str, str, str]]:
    rows = []
    for cells in iter_md_table_rows(md):
        if not cells or cells[0].strip().lower() in ("id", "segment id", "segment", "seg"):
            continue
        if len(cells) < 3:
            raise ValueError(f"row needs at least id | before | after: {cells}")
        try:
            ids = parse_ids(cells[0])
        except ValueError:
            raise ValueError(f"bad segment id cell: {cells[0]!r}") from None
        reason = cells[3] if len(cells) > 3 else ""
        rows.append((ids, cells[1], cells[2], reason))
    return rows


def apply(raw: dict, rows) -> tuple[dict, list[str]]:
    out = copy.deepcopy(raw)
    by_id = {s["id"]: s for s in out["segments"]}
    errors = []
    for ids, before, after, _ in rows:
        if not before:
            errors.append(f"empty `before` for ids {ids}")
            continue
        for sid in ids:
            seg = by_id.get(sid)
            if seg is None:
                errors.append(f"segment {sid} does not exist")
            elif before not in seg["text"]:
                errors.append(f"segment {sid}: {before!r} not found in {seg['text']!r}")
            else:
                seg["text"] = seg["text"].replace(before, after)
    return out, errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode_dir")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ep = episode_dir(args.episode_dir)
    raw_p, md_p, out_p = ep / C.TRANSCRIPT_RAW, ep / C.CORRECTIONS, ep / C.TRANSCRIPT
    if not raw_p.exists():
        die(f"missing {raw_p} (run transcribe.py first)")
    if not md_p.exists():
        die(f"missing {md_p}: write the corrections table first (an empty table is fine)")

    if not args.force and is_fresh(out_p, raw_p, md_p):
        print(f"skip  {C.TRANSCRIPT} (up to date)")
    else:
        try:
            rows = parse_corrections(md_p.read_text(encoding="utf-8"))
        except ValueError as e:
            die(f"{C.CORRECTIONS}: {e}")
        fixed, errors = apply(load_json(raw_p), rows)
        if errors:
            die("corrections not applied:\n  " + "\n  ".join(errors))
        save_json(out_p, fixed)
        changed = sum(a["text"] != b["text"] for a, b in zip(fixed["segments"], load_json(raw_p)["segments"]))
        print(f"make  {C.TRANSCRIPT}  ({len(rows)} rows, {changed} segments changed)")

    errors = validate(ep)
    if errors:
        die("validate_transcript failed:\n  " + "\n  ".join(errors))
    print("ok    transcript.json validated")


if __name__ == "__main__":
    main()
