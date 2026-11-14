---
name: podcast-digest
description: Turn a Japanese podcast episode — an audio podcast (input.mp4 + shownote.txt) or a recorded two-person video call such as a StreamYard recording — into a vertical 9:16 short of up to 180 seconds with burned-in Japanese captions. It features the part(s) a human describes ("the part where we talk about X", "the important points"); several highlights are joined into one short with hard cuts. Use when asked to make a digest, highlight, short, reel, clip or 切り抜き動画 from a podcast episode or recording, or to transcribe/correct a Japanese episode for that purpose.
---

# Podcast / recording → highlight video

Deterministic work (audio, API calls, timing math, rendering) is done by scripts.
You do the two judgment steps: **correcting the transcript** (1b) and **choosing the
highlights** (2). There are two human checkpoints. Never skip them.

`SKILL_DIR` below means the directory that contains this file. Run scripts with uv,
which uses this skill's own locked environment from any working directory:
`uv run --project "$SKILL_DIR" "$SKILL_DIR/scripts/<name>.py" <episode_dir>`.
In the steps below, `<name>.py <ep>` is short for that command. Every script prints what it made
or skipped. Re-running is always safe: finished steps are skipped and derived files
are rebuilt when their inputs change. `--force` redoes a step.

## Episode layout (one directory per episode)

```
episodes/<slug>/
  input.mp4                        provided by the user (audio podcast or video recording)
  shownote.txt                     provided by the user (recommended: names, products, topics)
  episode.json                     you write it for video recordings: {"layout": "stacked"}
  audio.mp3, transcript.raw.json   step 1a  (transcribe.py)
  corrections.md                   step 1b  (you write it)
  transcript.json                  step 1b  (apply_corrections.py)
  clip.draft.json                  step 2   (you write it)
  clip.json                        step 2   (build_clip.py)
  layout.preview.png               step 2   (preview_layout.py, video recordings)
  digest.ass, digest.mp4           step 3   (render.py)
```

## Two kinds of episode, one pipeline

| | Audio podcast | Video recording (StreamYard 2-up) |
|---|---|---|
| `episode.json` | none (or `{"layout": "background"}`) | `{"layout": "stacked"}` |
| Picture | `assets/background.png` | left speaker tile on top, right tile below, captions in the band between |
| Sound | `audio.mp3` | cut from `input.mp4`, in sync with the picture |

Everything else (transcription, corrections, highlight selection, captions, loudness) is
the same. A 50-minute, ~1.5 GB recording is fine: only its audio is uploaded, and
rendering seeks straight to each highlight.

## Step 0: Set up

1. Run `check_env.py` (no episode argument). It checks ffmpeg/libass, the background,
   the font, and that `OPENAI_API_KEY` is set in the shell or in `$SKILL_DIR/.env`. If it
   exits 1, stop and point the user to README → Setup.
   - Never read, print, `cat` or `grep` the `.env` file, and never ask the user to paste the
     key into the chat. The user puts the key in `.env` themselves:
     `cp .env.example .env`, then edit the file.
2. Create `episodes/<slug>/` (short ASCII slug, e.g. `ep042`). Copy the recording there as
   `input.mp4` and the shownote as `shownote.txt`. If there is no shownote, ask the user
   for the cast and key names/terms and write them into `shownote.txt` yourself.
   Do not move or copy a recording you were not given a path to.
3. If it is a video recording of two people side by side, write `episode.json`:
   `{"layout": "stacked"}`.
4. Run `run_all.py episodes/<slug>`. It runs step 1a and stops
   at step 1b with exit code 10. That pause is expected. Exit code 1 means an error: read it.

## Step 1a: Transcribe (script)

`run_all.py` already ran `transcribe.py`. It extracts mono 16 kHz audio and calls whisper-1
with word timestamps, `language="ja"`, and a vocabulary prompt built from the shownote.
Long files are split into overlapping chunks automatically. whisper-1 sometimes skips a
stretch of speech: long pauses between recognised words that still contain sound are
re-sent on their own and the recovered segments inserted (`gap ... recovered` lines). To see the prompt, run
`transcribe.py <ep> --print-prompt`. To override it, put text in `<ep>/prompt.txt` and
re-run with `--force`. The re-run costs API money, so ask the user first.

## Step 1b: Correct the transcript (you)

Goal: fix misrecognized words in each segment's `text`, using `shownote.txt` as the
source of truth for names, products and terms.

1. Read `shownote.txt` in full. List the names (hosts, guests), products, companies and
   technical terms in it.
2. Read the transcript: `show_transcript.py <ep> --raw`. For long episodes, read it in
   ranges (`--from-id/--to-id`) and grep for likely mishearings of each term
   (`--grep クロード --grep claude`). Whisper usually writes English names in katakana,
   misspells them (`Wisper`), drops letters (`NTHROPIC`), or picks the wrong kanji
   homophone (`制度`→`精度`).
3. Write `corrections.md`: one markdown table. Each row replaces an exact substring in the
   listed segment(s):

   ```markdown
   | id | before | after | reason |
   |----|--------|-------|--------|
   | 6 | クロットコード | Claude Code | product name in shownote |
   | 7, 8, 12 | エーデント | エージェント | misrecognition of エージェント |
   ```

   - `id`: segment ids, comma-separated, ranges allowed (`10-12`). `before` must occur in
     that segment's current text. Rows apply in order.
   - Fix recognition errors only. Do not rephrase, summarize, remove fillers, or "improve"
     the Japanese. Fix an obvious homophone from context, and say "(context)" in `reason`
     when the shownote does not settle it.
   - If a segment is a whisper hallucination (for example 「ご視聴ありがとうございました」
     over silence), set it to empty: `| 41 | ご視聴ありがとうございました |  | hallucination |`.
   - **Never** merge, split, reorder or re-time segments. The scripts make that
     impossible as long as you go through `corrections.md`. Do not edit `transcript.json`
     by hand.
   - If nothing needs fixing, write the header row only.
4. Run `apply_corrections.py <ep>`. It writes `transcript.json` and runs
   `validate_transcript.py`. If a `before` is not found, fix that row and run it again.
5. **CHECKPOINT 1.** Show the human the corrections table (or a summary of it, plus the
   rows marked "(context)") and ask them to review `corrections.md`. Apply their changes
   by editing `corrections.md` and re-running `apply_corrections.py`. Continue only after
   they approve.

## Step 2: Choose the highlights (you)

This step **requires the human's instruction**: either a topic (「料金の話をしているところ」)
or a request for the best parts (「大事なポイントをまとめて」). If they have not given one,
ask for it. Do not guess a topic.

A short is one or more **spans** (contiguous parts of the episode) played in
chronological order and joined with hard cuts. **Total: at most 180 s.**

1. Search: `show_transcript.py <ep> --grep <keyword> ... -C 5` with the instruction's
   keywords plus synonyms and likely spellings. Then read generously around each hit
   (`--from-id/--to-id`) to see where each discussion really starts and ends. For "the
   important points", read the whole transcript in ranges first and list the candidate
   moments with their ids before choosing.
2. Choose the spans:
   - **One span** when the instruction names one topic and its discussion fits in 180 s.
   - **Several spans** (typically 2–4) when the instruction asks for highlights, or when one
     topic is spread out and the in-between talk is off-topic. Each span must make sense
     on its own *and* right after the previous one: a hard cut gives no context.
   - For every span: **start** where the point is introduced, not mid-thought or on a
     back-reference (「さっきの」「前回」「それで」). Trim a leading filler or transition
     (「では本題です。」「で、」) with `start_at`. **End** on a complete sentence; use
     `end_after` to stop inside a segment.
   - Prefer fewer, complete spans to many fragments; a span shorter than ~8 s rarely
     stands alone.
   - If the material is longer than 180 s, keep the strongest parts and say so in `reason`.
   - Add up to 2 `alternatives` (other strong spans) with reasons.
   - **If nothing matches, stop and tell the human what you searched for and what the
     episode does cover. Do not write a draft.**
3. Write `clip.draft.json`. You choose segment ids. The script does the timing:

   ```json
   {
     "instruction": "<the human's instruction, verbatim>",
     "title": "スキル機能と文字起こしの要点",
     "reason": "Why these parts, as a whole.",
     "spans": [
       { "start_id": 6, "end_id": 8, "start_at": "今日は、Claude Codeのスキル機能について",
         "title": "スキル機能とは", "reason": "Introduces the topic." },
       { "start_id": 12, "end_id": 12, "title": "分担が大事", "reason": "The core point." }
     ],
     "alternatives": [
       { "start_id": 120, "end_id": 131, "title": "...", "reason": "..." }
     ]
   }
   ```

   `start_at` and `end_after` are optional exact substrings of a span's first and last
   segment's text. Alternatives accept them too. Every span needs a `reason` when there
   are several. (A single-span draft may also put `start_id`/`end_id`/... at the top level.)
4. Run `build_clip.py <ep>`. It snaps every boundary to word timestamps, adds about 0.3 s
   of padding (never past a neighbouring word), sorts the spans, checks that they do not
   overlap and total ≤ 180 s, writes `clip.json`, and validates it. It lists each span
   with its time range and length. Read any `warn` lines it prints. Adjacent spans
   (touching segment ids) count as overlapping: merge them into one span.
5. Video recordings only: run `preview_layout.py <ep>` and look at `layout.preview.png`
   (and `layout.source.png`, where the tiles are outlined red = top, blue = bottom). Each
   panel must show one person, centred, with no frame edge or neighbouring tile. If not,
   set `"tiles"` in `episode.json` (fractions of the source frame) and re-run:

   ```json
   {"layout": "stacked",
    "tiles": [{"x": 0.0, "y": 0.14, "w": 0.497, "h": 0.70},
              {"x": 0.504, "y": 0.14, "w": 0.496, "h": 0.70}]}
   ```

   The first tile goes on top. Swap the two entries to swap the speakers. Recordings from
   the same studio setup share the same tiles, so reuse a working `episode.json`.
6. **CHECKPOINT 2.** Show the human the plan: the overall title and total length, then
   each span in order with title, `mm:ss–mm:ss`, duration, reason, and its first and last
   sentences (`show_transcript.py <ep> --from-id A --to-id B`), then the alternatives. For
   video, show `layout.preview.png` too. Ask them to approve, swap, add or drop spans.
   - `build_clip.py <ep> --promote N`: with one span, swaps alternative N in; with several,
     adds alternative N as another span (watch the 180 s total).
   - Any other change: edit `clip.draft.json` and re-run `build_clip.py`.
   - Continue only after they approve.

## Step 3: Render (script)

Run `run_all.py <ep> --render` (or `render.py <ep>`). Output: `digest.mp4`, 1080x1920,
30 fps, H.264 + AAC, normalized to -14 LUFS, with captions burned in from `digest.ass`.
The spans are joined with hard cuts. Each cut gets a 50 ms audio fade so it does not click.

Then check the result:

- `ffprobe -v error -show_entries format=duration -of csv=p=0 <ep>/digest.mp4` must match
  the total that `build_clip.py` printed.
- Extract a frame from each span and look at them, so you know the captions render, are
  not tofu boxes, and (video) the panels are framed right:
  `ffmpeg -y -v error -ss 5 -i <ep>/digest.mp4 -frames:v 1 /tmp/frame.png`.
- If render warned that the font `Noto Sans JP` was not found, tell the user (README → Setup).

Report the path to `digest.mp4`, the title, and each span's source time range.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OPENAI_API_KEY is not set` | The user adds it to `$SKILL_DIR/.env` (template `.env.example`) or exports it. Never handle the key yourself |
| `before ... not found` | `before` must be an exact substring of that segment's *current* text (earlier rows may have changed it) |
| `segments changed but not listed` | Every changed segment id must appear in a `corrections.md` row |
| `spans total ... > 180s` / `exceeds 180s` | Drop a span, pick shorter ones, or trim with `start_at` / `end_after` |
| Speech missing from the transcript (a long jump in times where people talk) in a transcript made before gap recovery existed | `transcribe.py <ep> --fill-gaps` re-sends only the gaps. It prints how ids shift: renumber `corrections.md` / `clip.draft.json` to match, then `apply_corrections.py <ep>`. Blank any recovered 「ご視聴ありがとうございました」 as a hallucination |
| A second short from the same episode | Rename the finished `digest.mp4`, `digest.ass`, `clip.json`, `clip.draft.json` (e.g. `digest-keypoints.mp4`), then write a new `clip.draft.json` |
| `spans overlap` | Two spans share or touch segments: merge them into one span |
| `opens with 'さっき'` (warn) | Start later or trim with `start_at` so the span stands alone |
| Panels show the wrong area / a border | Fix `tiles` in `episode.json`, check with `preview_layout.py`, then `render.py <ep>` |
| `layout 'stacked' but input.mp4 has no video stream` | It is an audio podcast: remove `episode.json` |
| Captions in a fallback or thin font | `assets/fonts/NotoSansJP-Bold.ttf` must exist (a static Bold file, not the variable font); see README → Setup |
| Changed `background.png` / font settings | `render.py <ep>` notices and re-renders; `--force` to be sure |
