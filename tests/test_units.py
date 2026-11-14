"""Unit tests (stdlib unittest, no API, no ffmpeg needed).

Run: uv run python -m unittest discover -s tests -v
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import apply_corrections as AC  # noqa: E402
import build_clip as BC  # noqa: E402
import config as C  # noqa: E402
import render as R  # noqa: E402
import transcribe as T  # noqa: E402
import validate_clip as VC  # noqa: E402
import validate_transcript as VT  # noqa: E402
from common import StepError, align_text  # noqa: E402


def words_for(text: str, start: float, per_char: float = 0.1) -> list[dict]:
    """One word per 2 chars, evenly timed."""
    out, t = [], start
    for i in range(0, len(text), 2):
        w = text[i:i + 2]
        out.append({"word": w, "start": round(t, 3), "end": round(t + per_char * len(w), 3)})
        t += per_char * len(w)
    return out


def make_transcript(texts: list[str], gap: float = 0.5) -> dict:
    segs, t = [], 0.5
    for i, text in enumerate(texts):
        ws = words_for(text, t)
        segs.append({"id": i, "start": ws[0]["start"], "end": ws[-1]["end"], "text": text,
                     "original_text": text, "words": ws})
        t = ws[-1]["end"] + gap
    return {"language": "ja", "duration": round(t + 1.0, 3), "segments": segs}


class TestTranscribe(unittest.TestCase):
    def test_terms_from_shownote(self):
        note = (ROOT / "tests/fixtures/sample/shownote.txt").read_text(encoding="utf-8")
        terms = T.extract_terms(note)
        for t in ("今井", "佐藤", "Claude Code", "SKILL.md", "ffmpeg", "Anthropic"):
            self.assertIn(t, terms)
        self.assertNotIn("トピック", terms)
        self.assertFalse(any("https" in t for t in terms))

    def test_plan_chunks(self):
        self.assertEqual(T.plan_chunks(10, 100.0, 20, 8.0), [(0.0, 100.0)])
        plan = T.plan_chunks(60_000_000, 3600.0, C.MAX_UPLOAD_BYTES, 8.0)
        self.assertGreaterEqual(len(plan), 3)
        self.assertEqual(plan[0][0], 0.0)
        self.assertAlmostEqual(plan[-1][1], 3600.0)
        for (a0, a1), (b0, b1) in zip(plan, plan[1:]):
            self.assertAlmostEqual(a1 - b0, 8.0, places=6)  # overlap
        bps = 60_000_000 / 3600
        self.assertTrue(all((e - s) * bps <= C.MAX_UPLOAD_BYTES for s, e in plan))

    def test_merge_stitches_long_straddling_segment(self):
        def seg(text, start):
            ws = words_for(text, start)
            return {"start": ws[0]["start"], "end": ws[-1]["end"], "text": text, "words": ws}
        long_text = "あいうえおかきくけこさしすせそたちつてとなにぬねの"  # 2.5s
        # chunk A covers 0-11 and hears long_text truncated; chunk B covers 9-20 and hears its tail.
        a = [seg("はじめの文です", 1.0), seg(long_text[:20], 8.0)]
        b = [seg(long_text[10:], 9.0), seg("つぎの文です", 12.0)]
        merged = T.merge_chunks([(0.0, 11.0, a), (9.0, 20.0, b)])
        texts = [s["text"] for s in merged]
        self.assertEqual(texts[0], "はじめの文です")
        self.assertEqual(texts[-1], "つぎの文です")
        self.assertEqual(len(merged), 3)
        starts = [w["start"] for s in merged for w in s["words"]]
        self.assertEqual(starts, sorted(starts))  # no duplicated / reordered words

    def test_merge_keeps_whole_segments_apart(self):
        def seg(text, start):
            ws = words_for(text, start)
            return {"start": ws[0]["start"], "end": ws[-1]["end"], "text": text, "words": ws}
        a = [seg("いちばん", 1.0), seg("にばんめ", 9.5)]
        b = [seg("にばんめ", 9.5), seg("さんばん", 15.0)]
        merged = T.merge_chunks([(0.0, 12.0, a), (8.0, 20.0, b)])
        self.assertEqual([s["text"] for s in merged], ["いちばん", "にばんめ", "さんばん"])

    def test_missed_gaps_need_sound(self):
        def seg(text, start):
            ws = words_for(text, start)
            return {"start": ws[0]["start"], "end": ws[-1]["end"], "text": text, "words": ws}
        segs = [seg("あいうえ", 1.0), seg("かきくけ", 20.0), seg("さしすせ", 40.0)]  # each 0.4s long
        silences = [(1.4, 20.0)]  # first pause is silent; second (20.4–40) holds sound
        self.assertEqual(T.missed_gaps(segs, 41.0, silences), [(20.4, 40.0)])
        # a quiet room with a few seconds of sound is not enough; trailing gap counts too
        self.assertEqual(T.missed_gaps(segs, 60.0, [(1.4, 20.0), (20.4, 39.0), (40.4, 59.0)]), [])
        self.assertEqual(T.missed_gaps(segs, 60.0, [(1.4, 20.0), (20.4, 39.0)]), [(40.4, 60.0)])

    def test_recover_gap_clips_and_drops_hallucinations(self):
        resp = {"segments": [{"start": 0.2, "end": 3.0, "text": "あとはSkillsとか"},
                             {"start": 3.5, "end": 5.0, "text": "ご視聴ありがとうございました。"}],
                "words": [{"word": "あとは", "start": 0.2, "end": 0.9}, {"word": "Skillsとか", "start": 1.0, "end": 3.2},
                          {"word": "ご視聴", "start": 3.5, "end": 4.0}, {"word": "ありがとうございました", "start": 4.0, "end": 5.0}]}
        segs = T.recover_gap(resp, 100.0, 103.0)  # gap 100–103s
        self.assertEqual([s["text"] for s in segs], ["あとはSkillsとか"])
        self.assertEqual((segs[0]["start"], segs[0]["end"]), (100.2, 103.0))  # word end clipped to the gap


@unittest.skipUnless(__import__("shutil").which("ffmpeg"), "needs ffmpeg")
class TestTranscribeApiPath(unittest.TestCase):
    """Runs transcribe.transcribe() end to end with a fake OpenAI client (no network)."""

    def test_chunked_api_path_with_cache(self):
        import os
        import shutil
        import types
        from unittest import mock

        fixture = json.loads((ROOT / "tests/fixtures/sample/transcript.raw.json").read_text(encoding="utf-8"))
        calls = []

        class FakeTranscriptions:
            def create(self, *, model, file, language, prompt, response_format, timestamp_granularities):
                self_args = (model, language, response_format, tuple(timestamp_granularities))
                assert self_args == ("whisper-1", "ja", "verbose_json", ("segment", "word")), self_args
                calls.append(file.name)
                dur = float(__import__("common").media_duration(Path(file.name)))
                start = plans[len(calls) - 1][0] if len(plans) > 1 else 0.0
                segs = [s for s in fixture["segments"] if s["end"] > start and s["start"] < start + dur]
                resp = {"segments": [{"start": s["start"] - start, "end": s["end"] - start, "text": s["text"]}
                                     for s in segs],
                        "words": [{"word": w["word"], "start": w["start"] - start, "end": w["end"] - start}
                                  for s in segs for w in s["words"]]}
                return types.SimpleNamespace(model_dump=lambda: resp)

        class FakeClient:
            def __init__(self, **_):
                self.audio = types.SimpleNamespace(transcriptions=FakeTranscriptions())

        with tempfile.TemporaryDirectory() as d:
            ep = Path(d)
            shutil.copy(ROOT / "tests/fixtures/sample/input.mp4", ep / "input.mp4")
            shutil.copy(ROOT / "tests/fixtures/sample/shownote.txt", ep / "shownote.txt")
            audio = T.extract_audio(ep, force=False)
            plans = T.plan_chunks(audio.stat().st_size, __import__("common").media_duration(audio), 200_000, 8.0)
            fake_openai = types.ModuleType("openai")
            fake_openai.OpenAI = FakeClient
            with mock.patch.dict(sys.modules, {"openai": fake_openai}), \
                    mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}), \
                    mock.patch.object(C, "MAX_UPLOAD_BYTES", 200_000):
                raw = T.transcribe(ep, audio, "prompt", force=False)
                self.assertEqual(len(calls), len(plans))
                self.assertGreater(len(plans), 1)
                self.assertEqual([s["text"] for s in raw["segments"]], [s["text"] for s in fixture["segments"]])
                T.transcribe(ep, audio, "prompt", force=False)  # second run: served from chunks/ cache
                self.assertEqual(len(calls), len(plans))


class TestEnvFile(unittest.TestCase):
    """OPENAI_API_KEY from <repo>/.env; a real environment variable wins."""

    def run_load(self, env_text, environ):
        import os
        from unittest import mock

        import common
        with tempfile.TemporaryDirectory() as d:
            env_file = Path(d) / ".env"
            if env_text is not None:
                env_file.write_text(env_text, encoding="utf-8")
            with mock.patch.object(C, "ENV_FILE", env_file), mock.patch.dict(os.environ, environ, clear=True):
                return common.load_env("OPENAI_API_KEY"), os.environ.get("OPENAI_API_KEY")

    def test_reads_dotenv(self):
        self.assertEqual(self.run_load('# comment\nOPENAI_API_KEY="sk-from-file"\n', {}), (".env", "sk-from-file"))

    def test_environment_takes_precedence(self):
        self.assertEqual(self.run_load("OPENAI_API_KEY=sk-from-file\n", {"OPENAI_API_KEY": "sk-env"}),
                         ("environment", "sk-env"))

    def test_empty_value_or_missing_file(self):
        self.assertEqual(self.run_load("OPENAI_API_KEY=\n", {}), (None, ""))
        self.assertEqual(self.run_load(None, {}), (None, None))


class TestCorrections(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ep = Path(self.tmp.name)
        self.raw = make_transcript(["クロードコードの話です", "エーデントとクロードコード"])
        (self.ep / C.TRANSCRIPT_RAW).write_text(json.dumps(self.raw, ensure_ascii=False), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, md: str, fixed: dict | None = None):
        (self.ep / C.CORRECTIONS).write_text(md, encoding="utf-8")
        if fixed is None:
            fixed, errs = AC.apply(self.raw, AC.parse_corrections(md))
            self.assertEqual(errs, [])
        (self.ep / C.TRANSCRIPT).write_text(json.dumps(fixed, ensure_ascii=False), encoding="utf-8")
        return fixed

    def test_apply_and_validate(self):
        md = "| id | before | after | reason |\n|---|---|---|---|\n| 0, 1 | クロードコード | Claude Code | name |\n| 1 | エーデント | エージェント | x |\n"
        fixed = self.write(md)
        self.assertEqual(fixed["segments"][1]["text"], "エージェントとClaude Code")
        self.assertEqual(fixed["segments"][1]["original_text"], "エーデントとクロードコード")
        self.assertEqual(VT.validate(self.ep), [])

    def test_missing_before_is_error(self):
        _, errs = AC.apply(self.raw, AC.parse_corrections("| 0 | ないもの | x | y |\n"))
        self.assertTrue(errs)

    def test_validator_rejects_retiming_and_unlisted_change(self):
        fixed = copy.deepcopy(self.raw)
        fixed["segments"][0]["start"] += 0.1
        fixed["segments"][1]["text"] = "変更"
        self.write("| id | before | after | reason |\n|---|---|---|---|\n", fixed)
        errs = " ".join(VT.validate(self.ep))
        self.assertIn("timestamps changed", errs)
        self.assertIn("not listed", errs)

    def test_validator_rejects_merge(self):
        fixed = copy.deepcopy(self.raw)
        fixed["segments"] = fixed["segments"][:1]
        self.write("| id | before | after | reason |\n", fixed)
        self.assertIn("segment count changed", " ".join(VT.validate(self.ep)))


class TestClip(unittest.TestCase):
    def setUp(self):
        self.tr = make_transcript(["さっきの話ですが", "今日はスキルの話をします", "スキルは便利です", "次の話題です"])

    def test_padding_snaps_and_never_crosses_neighbours(self):
        clip = BC.build(self.tr, {"instruction": "スキル", "start_id": 1, "end_id": 2, "title": "t", "reason": "r"})
        sp = clip["spans"][0]
        s1, s2 = self.tr["segments"][1], self.tr["segments"][2]
        self.assertAlmostEqual(sp["start"], s1["words"][0]["start"] - C.CLIP_PADDING_SECONDS, places=3)
        self.assertGreaterEqual(sp["start"], self.tr["segments"][0]["words"][-1]["end"])
        self.assertAlmostEqual(sp["end"], s2["words"][-1]["end"] + C.CLIP_PADDING_SECONDS, places=3)
        self.assertAlmostEqual(clip["duration"], sp["end"] - sp["start"], places=3)

        tight = make_transcript(["前の文", "本文です", "後の文"], gap=0.1)
        sp = BC.build(tight, {"instruction": "x", "start_id": 1, "end_id": 1, "title": "t", "reason": "r"})["spans"][0]
        self.assertAlmostEqual(sp["start"], tight["segments"][0]["words"][-1]["end"], places=3)
        self.assertAlmostEqual(sp["end"], tight["segments"][2]["words"][0]["start"], places=3)

    def test_start_at_trims_to_word(self):
        clip = BC.build(self.tr, {"instruction": "x", "start_id": 1, "end_id": 2, "start_at": "スキル",
                                  "title": "t", "reason": "r"})
        first = clip["spans"][0]["segments"][0]
        self.assertEqual(first["text"], "スキルの話をします")
        self.assertIn(first["start"], [w["start"] for w in self.tr["segments"][1]["words"]])

    def test_too_long_is_rejected(self):
        long_tr = make_transcript(["話" * 400, "話" * 1500])
        with self.assertRaises(BC.DraftError):
            BC.build(long_tr, {"instruction": "x", "start_id": 0, "end_id": 1, "title": "t", "reason": "r"})

    def test_multi_span_sorted_overlap_and_total(self):
        draft = {"instruction": "x", "title": "t", "reason": "r",
                 "spans": [{"start_id": 3, "end_id": 3, "reason": "b"}, {"start_id": 1, "end_id": 1, "reason": "a"}]}
        clip = BC.build(self.tr, draft)
        self.assertEqual([s["start_id"] for s in clip["spans"]], [1, 3])  # chronological
        self.assertAlmostEqual(clip["duration"], sum(s["end"] - s["start"] for s in clip["spans"]), places=3)
        with self.assertRaisesRegex(BC.DraftError, "overlap"):
            BC.build(self.tr, {**draft, "spans": [{"start_id": 1, "end_id": 2, "reason": "a"},
                                                  {"start_id": 2, "end_id": 3, "reason": "b"}]})
        with self.assertRaisesRegex(BC.DraftError, "reason"):
            BC.build(self.tr, {**draft, "spans": [{"start_id": 1, "end_id": 1}, {"start_id": 3, "end_id": 3}]})
        # each span fits, but together they exceed the cap
        long_tr = make_transcript(["話" * 1000, "間", "話" * 1000])
        with self.assertRaisesRegex(BC.DraftError, "total"):
            BC.build(long_tr, {**draft, "spans": [{"start_id": 0, "end_id": 0, "reason": "a"},
                                                  {"start_id": 2, "end_id": 2, "reason": "b"}]})

    def test_validate_clip_roundtrip_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d)
            (ep / C.TRANSCRIPT).write_text(json.dumps(self.tr, ensure_ascii=False), encoding="utf-8")
            clip = BC.build(self.tr, {"instruction": "x", "start_id": 0, "end_id": 2, "title": "t", "reason": "r"})
            (ep / C.CLIP).write_text(json.dumps(clip, ensure_ascii=False), encoding="utf-8")
            errs, warns = VC.validate(ep)
            self.assertEqual(errs, [])
            self.assertTrue(any("さっき" in w for w in warns))  # opens with a back-reference
            sp = clip["spans"][0]
            sp["segments"][1]["text"] = "改変"
            sp["end"] = sp["start"] + 200
            (ep / C.CLIP).write_text(json.dumps(clip, ensure_ascii=False), encoding="utf-8")
            errs, _ = VC.validate(ep)
            self.assertTrue(any("exceeds" in e for e in errs))

    def test_validate_multi_span_and_v1(self):
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d)
            (ep / C.TRANSCRIPT).write_text(json.dumps(self.tr, ensure_ascii=False), encoding="utf-8")
            clip = BC.build(self.tr, {"instruction": "x", "title": "t", "reason": "r", "spans": [
                {"start_id": 1, "end_id": 1, "reason": "a"}, {"start_id": 3, "end_id": 3, "reason": "b"}]})
            write = lambda c: (ep / C.CLIP).write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")  # noqa: E731
            write(clip)
            self.assertEqual(VC.validate(ep)[0], [])
            swapped = copy.deepcopy(clip)
            swapped["spans"].reverse()
            write(swapped)
            self.assertTrue(any("chronological" in e for e in VC.validate(ep)[0]))
            # original single-span format (start/end/segments at the top level) still validates
            sp = clip["spans"][0]
            write({"instruction": "x", "title": "t", "reason": "r", "start": sp["start"], "end": sp["end"],
                   "segments": sp["segments"], "alternatives": []})
            self.assertEqual(VC.validate(ep)[0], [])

    def test_word_past_segment_end_with_neighbour_right_after(self):
        # whisper: the last word ends after its segment's end, and the next word starts
        # exactly there (no silence) — build pads to that word end, validate must agree.
        tr = copy.deepcopy(self.tr)
        seg, nxt = tr["segments"][1], tr["segments"][2]
        seg["words"][-1]["end"] = round(seg["end"] + 0.1, 3)
        nxt["start"] = nxt["words"][0]["start"] = seg["words"][-1]["end"]
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d)
            (ep / C.TRANSCRIPT).write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
            clip = BC.build(tr, {"instruction": "x", "title": "t", "reason": "r", "start_id": 1, "end_id": 1})
            self.assertAlmostEqual(clip["spans"][0]["end"], seg["words"][-1]["end"])
            (ep / C.CLIP).write_text(json.dumps(clip, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(VC.validate(ep)[0], [])

    def run_main(self, ep: Path, *args: str) -> None:
        sys_argv = sys.argv
        try:
            sys.argv = ["build_clip.py", str(ep), *args]
            BC.main()
        finally:
            sys.argv = sys_argv

    def test_promote_alternative(self):
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d)
            (ep / C.TRANSCRIPT).write_text(json.dumps(self.tr, ensure_ascii=False), encoding="utf-8")
            draft = {"instruction": "x", "start_id": 1, "end_id": 2, "title": "main", "reason": "r",
                     "alternatives": [{"start_id": 3, "end_id": 3, "title": "alt", "reason": "r2"}]}
            (ep / C.CLIP_DRAFT).write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")
            self.run_main(ep, "--promote", "0")
            new = json.loads((ep / C.CLIP_DRAFT).read_text(encoding="utf-8"))
            self.assertEqual((new["title"], new["spans"][0]["start_id"]), ("alt", 3))
            self.assertEqual(new["alternatives"][0]["title"], "main")
            clip = json.loads((ep / C.CLIP).read_text(encoding="utf-8"))
            self.assertEqual((clip["title"], clip["spans"][0]["start_id"]), ("alt", 3))

    def test_promote_adds_span_when_several(self):
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d)
            tr = make_transcript(["一つ目の話", "間の話", "二つ目の話", "間の話", "三つ目の話"])
            (ep / C.TRANSCRIPT).write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
            draft = {"instruction": "x", "title": "main", "reason": "r",
                     "spans": [{"start_id": 0, "end_id": 0, "reason": "a"}, {"start_id": 4, "end_id": 4, "reason": "c"}],
                     "alternatives": [{"start_id": 2, "end_id": 2, "title": "alt", "reason": "b"}]}
            (ep / C.CLIP_DRAFT).write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")
            self.run_main(ep, "--promote", "0")
            clip = json.loads((ep / C.CLIP).read_text(encoding="utf-8"))
            self.assertEqual([s["start_id"] for s in clip["spans"]], [0, 2, 4])
            self.assertEqual(clip["alternatives"], [])


class TestCaptions(unittest.TestCase):
    def test_align_corrected_text(self):
        words = [{"word": "クロット", "start": 1.0, "end": 1.4}, {"word": "コード", "start": 1.4, "end": 1.8},
                 {"word": "の", "start": 1.8, "end": 1.9}, {"word": "話", "start": 2.0, "end": 2.3}]
        times, offsets = align_text("Claude Codeの話。", words, 1.0, 2.3)
        self.assertEqual(len(times), len("Claude Codeの話。"))
        self.assertAlmostEqual(times[0][0], 1.0)
        self.assertAlmostEqual(times["Claude Codeの話。".index("話")][0], 2.0)
        self.assertEqual([s for s, _ in times], sorted(s for s, _ in times))
        self.assertEqual(offsets[-1], "Claude Codeの話。".index("話"))

    def test_layout_limits(self):
        text = ("まさにそれがポイントです。決まった処理は、スクリプトに任せて、判断が必要なところだけ"
                "エージェントにやってもらう、この分担が大事だと思います。Claude Codeのスキル機能について話しましょう。")
        for a, b in R.split_cues(text, set()):
            lines = R.split_lines(text[a:b], set())
            self.assertLessEqual(len(lines), C.MAX_LINES)
            for line in lines:
                self.assertLessEqual(R.text_width(line), C.MAX_CHARS_PER_LINE, line)
            self.assertNotIn("Claude\nCode", "\n".join(lines))
            joined = "|".join(lines)
            self.assertNotIn("Claude|Code", joined)

    def test_min_duration_and_no_overlap(self):
        tr = make_transcript(["はい", "そうですね", "では次の話題に行きましょう"], gap=0.2)
        clip = BC.build(tr, {"instruction": "x", "start_id": 0, "end_id": 2, "title": "t", "reason": "r"})
        cues = R.build_cues(clip, tr)
        for c, n in zip(cues, cues[1:]):
            self.assertLessEqual(c["end"], n["start"] + 1e-6)
        for c in cues[:-1]:
            self.assertGreaterEqual(c["end"] - c["start"], min(C.MIN_CUE_SECONDS, 1.0) - 1e-6)

    def test_snap_early_start_inside_silence(self):
        sil = [(1.0, 1.5)]
        self.assertEqual(R.snap_start(1.2, sil), 1.5)
        self.assertEqual(R.snap_start(0.5, [(0.0, 2.0)]), 0.5)  # >0.6s away: leave it
        self.assertEqual(R.snap_start(3.0, sil), 3.0)  # no pause nearby: leave it

    def test_snap_late_start_only_across_a_cue_boundary(self):
        sil = [(4.5, 5.4)]
        # previous cue's speech ended before the pause -> late start moves back to the onset
        self.assertEqual(R.snap_start(5.75, sil, prev_end=4.6), 5.4)
        # previous cue was still speaking after the pause -> the onset is not ours
        self.assertEqual(R.snap_start(5.75, sil, prev_end=5.6), 5.75)
        self.assertEqual(R.snap_start(6.5, sil, prev_end=4.6), 6.5)  # too far from the onset


class TestMultiSpanRender(unittest.TestCase):
    def setUp(self):
        self.tr = make_transcript(["一つ目の話です", "間の雑談です", "二つ目の話です"], gap=0.5)
        self.clip = BC.build(self.tr, {"instruction": "x", "title": "t", "reason": "r", "spans": [
            {"start_id": 0, "end_id": 0, "reason": "a"}, {"start_id": 2, "end_id": 2, "reason": "b"}]})

    def test_span_plan_is_frame_exact(self):
        plan = R.span_plan(self.clip)
        self.assertEqual([s for s, _ in plan], [sp["start"] for sp in self.clip["spans"]])
        for (_, d), sp in zip(plan, self.clip["spans"]):
            self.assertAlmostEqual(d * C.VIDEO_FPS, round(d * C.VIDEO_FPS), places=6)
            self.assertLessEqual(abs(d - (sp["end"] - sp["start"])), 0.5 / C.VIDEO_FPS + 1e-9)

    def test_cues_are_offset_per_span_and_stay_inside_it(self):
        (_, d0), (_, d1) = R.span_plan(self.clip)
        pad = C.CLIP_PADDING_SECONDS
        cues = R.build_cues(self.clip, self.tr, [(0.0, pad), (d0, d0 + pad)])  # the padding is silent
        first = [c for c in cues if c["start"] < d0]
        second = [c for c in cues if c["start"] >= d0]
        self.assertTrue(first and second)
        self.assertTrue(all(c["end"] <= d0 + 1e-6 for c in first))  # never runs across the cut
        self.assertNotIn("雑談", "".join("".join(c["lines"]) for c in cues))  # the gap is cut out
        self.assertIn("二つ目", "".join(second[0]["lines"]))
        # second span's words start 0.3 s (padding) into it
        sp1 = self.clip["spans"][1]
        self.assertAlmostEqual(second[0]["start"], d0 + sp1["segments"][0]["start"] - sp1["start"], places=2)
        self.assertLessEqual(cues[-1]["end"], d0 + d1 + 1e-6)

    def test_audio_graph_fades_only_at_cuts(self):
        g = R.audio_graph(R.span_plan(self.clip), 0.0, final_fades=False)
        self.assertIn("concat=n=2:v=0:a=1", g)
        self.assertEqual(g.count("afade=t=out"), 1)  # end of span 0
        self.assertEqual(g.count("afade=t=in"), 1)  # start of span 1
        single = R.audio_graph([(10.0, 5.0)], 0.0, final_fades=True)
        self.assertNotIn("concat", single)
        self.assertEqual(single.count("afade"), 2)  # just the 0.3 s start/end fades

    def test_load_layout(self):
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d)
            self.assertEqual(R.load_layout(ep), {"layout": C.DEFAULT_LAYOUT})
            (ep / C.EPISODE_CONFIG).write_text('{"layout": "stacked"}', encoding="utf-8")
            self.assertEqual(R.load_layout(ep)["tiles"], C.STACK_TILES)
            for bad in ('{"layout": "grid"}', '{"layout": "stacked", "tiles": [{"x": 0, "y": 0, "w": 1, "h": 1}]}',
                        '{"layout": "stacked", "tiles": [{"x": 0.6, "y": 0, "w": 0.5, "h": 1}, '
                        '{"x": 0, "y": 0, "w": 0.5, "h": 1}]}'):
                (ep / C.EPISODE_CONFIG).write_text(bad, encoding="utf-8")
                with self.assertRaises(StepError):
                    R.load_layout(ep)

    def test_stacked_panels_fill_the_frame(self):
        self.assertEqual(2 * R.panel_height() + C.STACK_BAND_HEIGHT, C.VIDEO_HEIGHT)
        f = R.stacked_filter("[in]", "[out]", C.STACK_TILES)
        self.assertIn(f"overlay=0:{C.VIDEO_HEIGHT - R.panel_height()}", f)
        self.assertEqual(R.caption_style({"layout": "stacked"})[1], 5)  # centred in the band


if __name__ == "__main__":
    unittest.main()
