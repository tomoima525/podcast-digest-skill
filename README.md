# podcast digests skill for agents

Turn a podcast(video/audio) episode into a vertical (9:16) short of up to 180 s with
burned-in captions. You say what to feature in plain words ("the section where
we talk about pricing", "the important points"). The short is one part of the episode,
or several parts joined in order with hard cuts. English and Japanese supported.

<img width="270" height="480" alt="image" src="https://github.com/user-attachments/assets/1b5cc4fc-ce6e-4f3a-b14a-03646ff4bce7" />


Two kinds of episode go through the same pipeline:

- **Audio podcast**: a fixed background image plus the clip's audio.
- **Video recording** of two people side by side (e.g. StreamYard): the left speaker's
  tile on top, the right speaker's tile below, captions in the band between them. Chosen
  per episode with `episode.json` → `{"layout": "stacked"}`.

It is packaged as a **Claude skill**: [SKILL.md](SKILL.md) tells an agent when and how to
run each step. The deterministic steps are plain Python CLI scripts in `scripts/`.

```
input.mp4 + shownote.txt (+ episode.json for video)
   │ 1a  transcribe.py            whisper-1, word timestamps       → transcript.raw.json
   │ 1b  agent fixes names/terms  (corrections.md)                 → transcript.json
   │     ── CHECKPOINT 1: human reviews corrections.md
   │ 2   agent picks the span(s)  (clip.draft.json, + instruction) → clip.json
   │     preview_layout.py        (video: check the framing)       → layout.preview.png
   │     ── CHECKPOINT 2: human approves, swaps or adds spans
   │ 3   render.py                cut + join spans, 1080x1920,     → digest.mp4
   │                              captions, -14 LUFS
```

## Setup

**1. uv and ffmpeg (with libass)**

```bash
brew install uv ffmpeg         # macOS
```

```bash
sudo apt install ffmpeg        # Debian/Ubuntu; for uv see https://docs.astral.sh/uv/
```

Check that the subtitle filter is there: `ffmpeg -hide_banner -filters | grep -w ass`.

**2. The Python environment.** uv reads `pyproject.toml`, `uv.lock` and `.python-version`
(3.12) and creates `.venv/`, downloading that Python if needed. The only runtime
dependency is the OpenAI SDK (used by `transcribe.py`); `pyflakes` is in the `dev` group.

```bash
uv sync
```

`uv run` also syncs automatically, so this step is optional. Run every script as
`uv run scripts/<name>.py ...`; there is nothing to activate.

**3. The Noto Sans JP font** (captions use its bold weight)

The repo ships a static Bold font at `assets/fonts/NotoSansJP-Bold.ttf` (SIL Open Font
License 1.1), and `render.py` passes that folder to libass. No install is needed.

Why a static file: Homebrew and Google Fonts now ship Noto Sans JP only as a
*variable* font (`NotoSansJP[wght].ttf`). libass renders a variable font in its default
Regular weight, so Japanese captions come out thin while Latin words fall back to
another bold font. To regenerate the Bold instance from a variable font:

```bash
uvx --from fonttools fonttools varLib.instancer "NotoSansJP[wght].ttf" wght=700 --update-name-table -o assets/fonts/NotoSansJP-Bold.ttf
```

If the font is missing, `render.py` warns and libass falls back to another Japanese
font (Hiragino on macOS).

**4. The API key.** Put it in `.env` at the repo root. That file is gitignored, and
`.env.example` is the template:

```bash
cp .env.example .env
```

Then edit `.env` so it has `OPENAI_API_KEY=sk-...`. The scripts load it from the repo
root no matter which directory you run them from. A variable already exported in your
shell takes precedence over `.env`. Check the setup, which reports whether the key is
set but never prints it:

```bash
uv run scripts/check_env.py
```

**5. Use it as a Claude skill.** Symlink the repo into your skills directory:

```bash
ln -s "$(pwd)" ~/.claude/skills/podcast-digest
```

Then ask Claude something like: "make a digest of episodes/ep042, the part where we talk about pricing".

## Commands

All commands take an episode directory, `episodes/<slug>/`, that contains `input.mp4`
(audio or video) and, ideally, `shownote.txt` (names and terms for whisper and the
corrections step; a missing one is a warning, not an error). A step skips when its output already exists and is newer than its inputs.
`render.py` instead compares a fingerprint of its inputs, settings and code (in
`.digest.stamp.json`), so swapping in an asset with an older file date still re-renders.
`--force` redoes any step.

| Command | Step | What it does |
|---|---|---|
| `uv run scripts/check_env.py` | 0 | Checks ffmpeg with libass, the background image, the font, and `OPENAI_API_KEY` (from your shell or `.env`; the value is never printed). Exit `1` if something required is missing. |
| `uv run scripts/run_all.py <ep>` | all | Runs every step that is ready and stops at the next agent or human step. Exit `10` means paused and prints what is needed; exit `0` means done. |
| `uv run scripts/run_all.py <ep> --render` | all | Same, and also renders once `clip.json` is valid. Use it after checkpoint 2. |
| `uv run scripts/transcribe.py <ep> [--print-prompt] [--fill-gaps]` | 1a | Writes `audio.mp3` (mono, 16 kHz, 32 kbps) and `transcript.raw.json`. Files over 24 MB are split into 8 s-overlap chunks, then merged and de-duplicated at word level. Pauses of 6 s or more between recognised words that still hold 2 s or more of sound (whisper-1 sometimes skips speech) are re-sent on their own and the result is inserted; stock hallucinations such as 「ご視聴ありがとうございました」 are dropped from those. `--fill-gaps` runs only that step on an existing transcript and prints how segment ids shift. API responses are cached in `chunks/`. |
| `uv run scripts/show_transcript.py <ep> [--grep KW] [-C N] [--from-id A --to-id B] [--words ID] [--raw] [--diff]` | 1b, 2 | Prints the transcript compactly for reading and searching. |
| `uv run scripts/apply_corrections.py <ep>` | 1b | Applies the `corrections.md` table to `transcript.raw.json` to produce `transcript.json`, then validates it. |
| `uv run scripts/validate_transcript.py <ep>` | 1b | Checks the schema and that segment count, ids, timestamps and words are unchanged from the raw file. Every changed segment must be listed in `corrections.md`. |
| `uv run scripts/build_clip.py <ep> [--promote N]` | 2 | Turns `clip.draft.json` into `clip.json`. It snaps each span to word timestamps and adds 0.3 s padding without crossing a neighbouring word. Then it sorts the spans, rejects overlaps and enforces the 180 s total. `--promote N`: with one span, swaps in alternative N; with several, adds it as a span. |
| `uv run scripts/validate_clip.py <ep>` | 2 | Checks the total is ≤ 180 s, the spans are in order and don't overlap, the times fall inside the audio, each span's segments match `transcript.json`, and the padding rules hold. |
| `uv run scripts/preview_layout.py <ep> [--at S]` | 2 | Renders one frame as the digest will look (`layout.preview.png`). For video, it also writes `layout.source.png` with the tiles outlined. Use it to check or tune `tiles`. |
| `uv run scripts/render.py <ep> [--audio-from-video]` | 3 | Writes `digest.ass` and `digest.mp4`. It cuts each span, joins them with hard cuts (50 ms audio fades at each cut), and normalises the joined audio. Video (stacked) takes picture and sound from `input.mp4`. Background takes audio from `audio.mp3`; `--audio-from-video` cuts it from `input.mp4` instead, which sounds better. |

Settings such as the background path, font, caption size and position, loudness and the
180 s limit live in [scripts/config.py](scripts/config.py).

The background is `assets/background.png` (any size). `BACKGROUND_FIT` in `config.py`
controls how it fills the 1080x1920 frame:
- `"cover"` (default): scale and crop to fill the frame.
- `"contain"`: fit the whole image and fill the rest with `BACKGROUND_PAD_COLOR`. This
  suits small or square artwork on a plain background, because it needs less upscaling
  and crops nothing.

### Layouts (`episode.json`)

Optional, per episode. Without it the episode uses `DEFAULT_LAYOUT` (`"background"`).

```json
{"layout": "stacked",
 "tiles": [{"x": 0.0,   "y": 0.14, "w": 0.497, "h": 0.70},
           {"x": 0.504, "y": 0.14, "w": 0.496, "h": 0.70}]}
```

- `tiles` (optional, default `STACK_TILES` in `config.py`) gives where each speaker
  sits in the source frame. Values are fractions of the source width and height, so they
  work at any resolution. The first tile is drawn on top, the second below.
- Each tile is scaled and centre-cropped to fill a 1080x860 panel. The middle 200 px
  band (`STACK_BAND_HEIGHT`) holds the captions, at 72 px (`STACK_FONT_SIZE`), centred.
- The default tiles were measured on a StreamYard two-person layout. Check a new studio
  setup once with `preview_layout.py`, then reuse that `episode.json`.

### File formats

- `transcript.json`: `{language, duration, segments: [{id, start, end, text, original_text, words: [{word, start, end}]}]}`.
  Only `text` differs from the raw file.
- `corrections.md`: a table `| id | before | after | reason |`. Each row replaces an
  exact substring in the listed segment ids. Use commas and ranges for several ids.
- `clip.draft.json`: written by the agent: `instruction`, `title`, `reason`, and
  `spans: [{start_id, end_id, start_at?, end_after?, title, reason}]`, plus
  `alternatives`. A single-span draft can put `start_id`/`end_id`/... at the top level
  instead. See [SKILL.md](SKILL.md).
- `clip.json`: `{version: 2, instruction, title, reason, duration, spans: [{start, end, title, reason, start_id, end_id, segments: [{id, start, end, text}]}], alternatives: [{start, end, title, reason, start_id, end_id}]}`.
  Spans are in output order (chronological). Times are absolute seconds in the
  recording, and `duration` is the sum of the spans. The validator and renderer also
  accept the original single-span file (`start`/`end`/`segments` at the top level).

### How captions are made

Corrected text is aligned to whisper's word timestamps with a character diff, so a
fixed name (`クロットコード` → `Claude Code`) keeps the timing of the audio it replaced.
Each segment is split into cues by dynamic programming over break points:

- At most 2 lines × 16 full-width characters per cue. Half-width characters count ½.
- Breaks are preferred after 。、, after a particle that is followed by a new word, and
  at spaces.
- No break inside a katakana word, a kanji compound, an English word, or a multi-word
  name like "Claude Code". Line-start rules (kinsoku) are respected.

Cues are kept on screen for at least 1 s and never overlap. They are built per span and
never run across a cut. Each span's length is snapped to whole video frames, so the
picture, the sound and the captions stay in sync across the joins. Whisper word timestamps
drift by a few hundred ms around pauses: the local model tends to be early and whisper-1
tends to be late. So cue starts are aligned to speech onsets detected in the clip audio
(`silencedetect`), moving at most 0.6 s:
- An early start that falls inside a pause moves forward to where speech resumes.
- A late start moves back to the onset just before it, but only if the previous cue's
  speech had already ended when that pause began.

## Tests

```bash
uv run python -m unittest discover -s tests -v   # unit tests (fake OpenAI client, no network)
```

```bash
uv run tests/e2e.py                       # full pipeline, offline: podcast, then stacked video
```

```bash
uv run tests/e2e.py --live                # real whisper-1 for step 1a (needs OPENAI_API_KEY in .env)
```

The video E2E builds a StreamYard-like recording at test time
(`tests/make_video_fixture.py`, a few seconds). It has two coloured speaker tiles where
the default `tiles` expect them, the fixture's audio, and a white box that flashes at
every phrase start. It renders a three-span draft (`tests/fixtures/video/expected/`) with
`{"layout": "stacked"}`, then checks:
- the panel and band colours,
- that the duration equals the sum of the spans,
- caption sync,
- `check_sync.py --flashes`: each flash must appear in `digest.mp4` within 2 frames of its
  mapped time, which proves picture/sound sync after reframing and cuts.

`tests/fixtures/sample/` holds a synthetic 2-minute episode (two voices from macOS `say`)
with three topics, a shownote, and the instruction 「Claude Codeのスキル機能について話しているところ」.
`ground_truth.json` has the exact start time of every phrase.

The offline E2E replays the committed raw transcript and the recorded agent outputs in
`expected/`. It checks every pause and validator, the output format (1080x1920, 30 fps,
H.264/AAC), loudness (−14 ± 0.5 LUFS) and caption sync. `tests/check_sync.py` checks
sync by comparing each cue start against both the ground truth and the speech onsets
measured in `digest.mp4`; the tolerance is ±0.2 s. The committed raw transcript was
produced offline by `tests/local_transcribe.py`, a local openai-whisper `small` model
used only as a test stand-in. Production always uses whisper-1. To regenerate it
(downloads PyTorch into a throwaway environment, not into `.venv`):

```bash
uv run --with openai-whisper tests/local_transcribe.py <ep>
```

To rebuild the fixture (macOS only): `uv run tests/make_fixture.py`.

Lint: `uv run pyflakes scripts tests`.


Created by Tomoaki Imai/MIT license
