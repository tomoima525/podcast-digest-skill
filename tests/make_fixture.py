#!/usr/bin/env python3
"""Generate tests/fixtures/sample/input.mp4 (+ ground_truth.json) with macOS `say`.

Usage: uv run tests/make_fixture.py [--out tests/fixtures/sample]

Every phrase is synthesised separately, trimmed of silence and concatenated
with known gaps, so ground_truth.json holds exact phrase start/end times. The
sync check (tests/check_sync.py) compares caption cues against these.
macOS only (needs `say` with Japanese voices); the generated files are committed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import wave
from pathlib import Path

RATE = 22050
VOICES = {"今井": "Kyoko", "佐藤": "Eddy (Japanese (Japan))"}
GAP_CLAUSE, GAP_SENTENCE, GAP_TURN = 0.12, 0.30, 0.55

SCRIPT = [
    ("今井", "こんにちは、テックラジオ第四十二回です。パーソナリティの今井です。"),
    ("佐藤", "佐藤です。よろしくお願いします。"),
    ("今井", "今日は三つの話題を用意しています。まずは雑談からいきましょう。"),
    ("佐藤", "先週、京都に旅行してきました。紅葉にはまだ少し早かったんですが、嵐山の竹林がとても綺麗でした。"),
    ("今井", "いいですね。平日に行くと人も少なくて歩きやすいですよね。"),
    ("佐藤", "そうなんです。朝早く行ったので、ほとんど貸し切り状態でした。"),
    ("今井", "では本題です。今日はClaude Codeのスキル機能について話したいと思います。"),
    ("佐藤", "スキル機能というのは、エージェントに手順書を渡す仕組みのことですよね。"),
    ("今井", "そうです。SKILL.mdというファイルに、いつ何をするかを書いておくと、エージェントが必要なときに読み込んでくれます。"),
    ("佐藤", "スクリプトも一緒に置けるんですか。"),
    ("今井", "置けます。たとえば、ffmpegで動画を切り出すスクリプトを用意しておけば、毎回同じ品質で処理できます。"),
    ("佐藤", "なるほど。プロンプトに毎回書くよりも、再現性が高くなるわけですね。"),
    ("今井", "まさにそれがポイントです。決まった処理はスクリプトに任せて、判断が必要なところだけエージェントにやってもらう。この分担が大事だと思います。"),
    ("佐藤", "Anthropicのドキュメントにも、そういう設計が推奨されていますね。"),
    ("今井", "次は文字起こしの話です。OpenAIのWhisper APIを使っています。"),
    ("佐藤", "日本語の精度はどうですか。"),
    ("今井", "かなり良いです。ただ、固有名詞はよく間違えるので、ショーノートの用語をプロンプトに入れています。"),
    ("佐藤", "料金も一時間あたり数十円なので、気軽に試せますね。"),
    ("今井", "というわけで、今日はここまでです。お便りもお待ちしています。"),
    ("佐藤", "ありがとうございました。"),
]

SHOWNOTE = """テックラジオ 第42回「Claude Codeのスキル機能と文字起こし」

出演: 今井、佐藤

トピック
- 京都旅行の雑談（嵐山の竹林）
- Claude Code のスキル機能（SKILL.md、スクリプト、ffmpeg）
- OpenAI Whisper API での文字起こしと料金

リンク
- Anthropic: https://www.anthropic.com
- Claude Code: https://docs.anthropic.com/claude-code
- OpenAI Whisper: https://platform.openai.com/docs/guides/speech-to-text
"""

INSTRUCTION = "Claude Codeのスキル機能について話しているところ"


def phrases(text: str) -> list[str]:
    return [p for p in re.findall(r"[^、。！？]+[、。！？]?", text) if p.strip()]


def synth(text: str, voice: str, tmp: Path) -> bytes:
    aiff, wav = tmp / "p.aiff", tmp / "p.wav"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    # Trim leading and trailing silence so phrase boundaries are exact.
    trim = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.02"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(aiff), "-af",
                    f"{trim},areverse,{trim},areverse", "-ac", "1", "-ar", str(RATE),
                    "-sample_fmt", "s16", str(wav)], check=True)
    with wave.open(str(wav)) as w:
        return w.readframes(w.getnframes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "fixtures" / "sample"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    pcm = bytearray()
    silence = lambda s: b"\x00\x00" * int(round(s * RATE))  # noqa: E731
    truth = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pcm += silence(1.0)
        for turn, (speaker, line) in enumerate(SCRIPT):
            if turn:
                pcm += silence(GAP_TURN)
            for k, ph in enumerate(phrases(line)):
                start = len(pcm) / 2 / RATE
                pcm += synth(ph, VOICES[speaker], tmp)
                end = len(pcm) / 2 / RATE
                truth.append({"turn": turn, "speaker": speaker, "text": ph,
                              "start": round(start, 3), "end": round(end, 3)})
                pcm += silence(GAP_SENTENCE if ph[-1] in "。！？" else GAP_CLAUSE)
        pcm += silence(1.0)
        wav = tmp / "all.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(bytes(pcm))
        dur = len(pcm) / 2 / RATE
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=0x303030:s=640x360:r=10:d={dur:.3f}",
                        "-i", str(wav), "-c:v", "libx264", "-preset", "veryfast", "-crf", "35",
                        "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart",
                        str(out / "input.mp4")], check=True)

    (out / "ground_truth.json").write_text(json.dumps({"duration": round(dur, 3), "phrases": truth},
                                                      ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "shownote.txt").write_text(SHOWNOTE, encoding="utf-8")
    (out / "instruction.txt").write_text(INSTRUCTION + "\n", encoding="utf-8")
    print(f"wrote {out}/input.mp4 ({dur:.1f}s, {len(truth)} phrases)")


if __name__ == "__main__":
    main()
