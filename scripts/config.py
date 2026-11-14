"""Single source of truth for paths and tunables used by every script."""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
BACKGROUND_IMAGE = ASSETS_DIR / "background.png"
# "cover":   scale and crop to fill the 9:16 frame (nothing letterboxed, edges cropped).
# "contain": fit the whole image inside the frame, fill the rest with BACKGROUND_PAD_COLOR
#            (best for small or square artwork on a plain background: less upscaling, no crop).
BACKGROUND_FIT = "cover"
BACKGROUND_PAD_COLOR = "black"  # any ffmpeg color: "black", "#101820", "0x1e1e1e"
# Optional: drop NotoSansJP-Bold.ttf (or the variable font) here to avoid
# depending on a system-wide install. Passed to libass as `fontsdir` if present.
FONTS_DIR = ASSETS_DIR / "fonts"
# Secrets (OPENAI_API_KEY) are read from here; a variable already set in the
# environment takes precedence. Gitignored; see .env.example.
ENV_FILE = ROOT / ".env"

# --- Episode file contract ---------------------------------------------------
INPUT_VIDEO = "input.mp4"
SHOWNOTE = "shownote.txt"
AUDIO = "audio.mp3"
TRANSCRIPT_RAW = "transcript.raw.json"
TRANSCRIPT = "transcript.json"
CORRECTIONS = "corrections.md"
CLIP_DRAFT = "clip.draft.json"
CLIP = "clip.json"
DIGEST_VIDEO = "digest.mp4"
DIGEST_ASS = "digest.ass"
DIGEST_STAMP = ".digest.stamp.json"  # fingerprint of render inputs + settings
CHUNKS_DIR = "chunks"  # cached per-chunk API responses (resume after failure)
EPISODE_CONFIG = "episode.json"  # optional per-episode settings: {"layout": ..., "tiles": [...]}
LAYOUT_PREVIEW = "layout.preview.png"  # written by preview_layout.py

# --- Transcription (step 1a) -------------------------------------------------
LANGUAGE = "ja"
WHISPER_MODEL = "whisper-1"  # only whisper-1 returns timestamps; do not change
AUDIO_SAMPLE_RATE = 16000
AUDIO_BITRATE = "32k"  # mono 16 kHz @ 32 kbps ≈ 14 MB/hour
MAX_UPLOAD_BYTES = 24 * 1024 * 1024  # API hard limit is 25 MB; keep margin
CHUNK_OVERLAP_SECONDS = 8.0
PROMPT_MAX_CHARS = 180  # whisper only looks at the last ~224 tokens of the prompt
# whisper-1 sometimes skips a stretch of speech in a long file (no segment at all).
# A pause between recognised words of at least GAP_MIN_SECONDS that holds at least
# GAP_SPEECH_SECONDS of non-silent audio is re-sent on its own and the result inserted.
GAP_MIN_SECONDS = 6.0
GAP_SPEECH_SECONDS = 2.0
GAP_SILENCE_DB = -40  # silencedetect threshold used to tell speech from room noise
GAP_SILENCE_MIN = 0.4
# Stock phrases whisper invents over near-silence. Recovered gap segments containing
# one are dropped (elsewhere they are blanked by hand in corrections.md).
GAP_HALLUCINATIONS = ("ご視聴ありがとうございました", "ご視聴ありがとうございます", "チャンネル登録",
                      "最後までご覧いただき")

# --- Clip selection (step 2) -------------------------------------------------
MAX_CLIP_SECONDS = 180.0
CLIP_PADDING_SECONDS = 0.3
MAX_ALTERNATIVES = 2

# --- Render (step 3) -----------------------------------------------------------
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30
X264_PRESET = "veryfast"
AUDIO_FADE_SECONDS = 0.3  # at the very start and end of the digest
JOIN_FADE_SECONDS = 0.05  # tiny fade out/in at each hard cut between spans (avoids clicks)

# --- Layout ----------------------------------------------------------------------
# "background": a still image (BACKGROUND_IMAGE) + the clip's audio. For audio podcasts.
# "stacked":    a two-person recording (e.g. StreamYard side-by-side) re-framed for 9:16:
#               the first tile on top, the second below, captions in the band between.
# An episode picks one in episode.json ({"layout": "stacked"}); this is the default.
DEFAULT_LAYOUT = "background"
STACK_BAND_HEIGHT = 200  # caption band in the middle; each panel is (1920 - band) / 2 = 860 px
STACK_BAND_COLOR = "black"
# Where each speaker's tile sits in the source frame, as fractions of its width and height
# (so they hold for any resolution). Each tile is scaled and centre-cropped to fill its
# panel. Defaults measured on a real 1280x720 StreamYard two-person recording (tiles
# fill the frame height, 6 px divider at x=637-642): each crop is panel-shaped (1.26:1),
# frames the face, and leaves out the name badges and the StreamYard logo. Override per
# episode in episode.json "tiles" and check with preview_layout.py.
STACK_TILES = [
    {"x": 0.0, "y": 0.14, "w": 0.497, "h": 0.70},
    {"x": 0.504, "y": 0.14, "w": 0.496, "h": 0.70},
]
STACK_FONT_SIZE = 72  # two lines must fit in the band
TARGET_LUFS = -14.0
TRUE_PEAK_DB = -1.0
SILENCE_NOISE_DB = -35  # pause detection on the normalised clip (cue starts snap to speech onsets)
OUTPUT_AUDIO_RATE = 48000
OUTPUT_AUDIO_BITRATE = "192k"

# --- Captions --------------------------------------------------------------------
FONT_NAME = "Noto Sans JP"
FONT_SIZE = 80  # ASS size = line height; CJK glyphs are ~0.7x that, so 16 chars ≈ 900px of the 960px safe width
FONT_BOLD = True
CAPTION_MARGIN_LR = 60
CAPTION_MARGIN_V = 420  # px from the bottom edge: lower third, clear of app UI
CAPTION_OUTLINE = 5
CAPTION_SHADOW = 2
MAX_CHARS_PER_LINE = 16  # full-width = 1, half-width (ASCII) = 0.5
MAX_LINES = 2
MIN_CUE_SECONDS = 1.0
MERGE_GAP_SECONDS = 0.3  # gaps shorter than this are closed to avoid flicker
