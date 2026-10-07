#!/usr/bin/env python3
"""Seed-TTS English texts through OmniVoice OV GPU, scored with Whisper turbo.

Official OmniVoice Seed-TTS English uses whisper-large-v3 + clone + 32 steps.
This run: same 1088 texts, AMC hop (auto-voice, 16 steps), Whisper large-v3 turbo,
and the official Seed-TTS English post_process (lowercase, strip punctuation).
"""

from __future__ import annotations

import json
import os
import string
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from athena.amc import AMC, AMCConfig  # noqa: E402
from athena.amc.cli import _write_wav  # noqa: E402
from athena.amc.tts import TTSOVBackend  # noqa: E402
from athena.noise import word_error_rate  # noqa: E402

DOC_MODELS = Path("/home/light/Documents/projectxi1/models")
JSONL = DOC_MODELS / "tts_eval" / "seedtts_test_en.jsonl"
WAV_DIR = Path(os.environ.get("SEEDTTS_WAV_DIR", "/tmp/seedtts_en_ov"))
HYP_PATH = Path(os.environ.get("SEEDTTS_HYP", "/tmp/seedtts_en_turbo_hyp.jsonl"))
SUMMARY = Path(os.environ.get("SEEDTTS_SUMMARY", "/tmp/seedtts_en_turbo_summary.txt"))


def _models() -> Path:
    local = ROOT / "models"
    if (local / "omnivoice-fp16-ov" / "forward.xml").is_file():
        return local
    return DOC_MODELS


def _cfg() -> AMCConfig:
    models = _models()
    return AMCConfig(
        device="GPU",
        backend="openvino",
        llama_model_path=str(models / "llama-3.1-8b-instruct-int4-ov"),
        llama_int8_path=str(models / "llama-3.1-8b-instruct-int8-ov"),
        whisper_model_path=str(models / "whisper-large-v3-turbo-int8-ov"),
        translate_model_path=str(models / "translategemma-4b-it-int8-ov"),
        tts_model_path=str(models / "omnivoice-fp16-ov"),
        cache_dir=str(models / "ov_cache"),
        job_timeout_s=1200.0,
        allow_cpu_fallback=False,
        tts_num_step=16,
        whisper_language="<|en|>",
    )


def seedtts_post_process(text: str) -> str:
    """OmniVoice eval/wer/seedtts.py post_process for lang=en."""
    punct = string.punctuation
    for x in punct:
        if x == "'":
            continue
        text = text.replace(x, "")
    text = text.replace("  ", " ")
    return text.lower().strip()


def _edit_counts(ref: str, hyp: str) -> tuple[int, int, int, int]:
    a, b = ref.split(), hyp.split()
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "d"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "i"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "k"
            else:
                subs = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                best = min(subs, dele, ins)
                dp[i][j] = best
                op[i][j] = "s" if best == subs else ("d" if best == dele else "i")
    i, j = n, m
    s = d = ins = 0
    while i > 0 or j > 0:
        o = op[i][j]
        if o == "k":
            i -= 1
            j -= 1
        elif o == "s":
            s += 1
            i -= 1
            j -= 1
        elif o == "d":
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins, n


def resample_24k_to_16k(pcm: list[float]) -> list[float]:
    import numpy as np

    x = np.asarray(pcm, dtype=np.float32)
    if x.size == 0:
        return []
    n_out = max(1, int(round(x.size * 16000 / 24000)))
    t_in = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32).tolist()


def _read_wav(path: Path) -> list[float]:
    import struct
    import wave

    with wave.open(str(path), "rb") as wf:
        n = wf.getnframes()
        raw = wf.readframes(n)
        sr = wf.getframerate()
    if sr != 24000:
        raise SystemExit(f"expected 24 kHz wav, got {sr}: {path}")
    samples = struct.unpack("<" + "h" * n, raw)
    return [x / 32768.0 for x in samples]


def load_items() -> list[dict]:
    items = []
    with JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            items.append({"id": o["id"], "text": o["text"]})
    return items


def main() -> int:
    if not JSONL.is_file():
        print(f"missing {JSONL}", file=sys.stderr)
        return 2
    items = load_items()
    print(f"seedtts_test_en  n={len(items)}", flush=True)
    print("generator: OmniVoice FP16 OpenVINO GPU auto-voice num_step=16", flush=True)
    print(f"wav_dir {WAV_DIR}", flush=True)
    print("ASR: Whisper large-v3 turbo (AMC)  scorer: seedtts post_process", flush=True)
    print("official paper uses clone + whisper-large-v3 + 32 steps", flush=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _cfg()
    t_all = time.perf_counter()
    backend = TTSOVBackend(cfg)
    backend.load()
    try:
        for i, item in enumerate(items, 1):
            wav = WAV_DIR / f"{i:04d}.wav"
            if wav.is_file() and wav.stat().st_size > 44:
                if i % 100 == 0:
                    print(f"  tts skip {i}/{len(items)}", flush=True)
                continue
            speech = backend.speak(item["text"], language="en")
            _write_wav(wav, speech.pcm, speech.sample_rate)
            if i == 1 or i % 25 == 0 or i == len(items):
                print(
                    f"  tts {i}/{len(items)}  {len(speech.pcm)/speech.sample_rate:.2f}s audio",
                    flush=True,
                )
    finally:
        backend.unload()
    print(f"tts done  {time.perf_counter() - t_all:.1f}s", flush=True)

    hyps: list[dict] = []
    with AMC(cfg) as amc:
        for i, item in enumerate(items, 1):
            wav = WAV_DIR / f"{i:04d}.wav"
            pcm = _read_wav(wav)
            hyp = amc.transcribe(resample_24k_to_16k(pcm))
            gold = item["text"]
            gold_n = seedtts_post_process(gold)
            hyp_n = seedtts_post_process(hyp)
            s, d, ins, w = _edit_counts(gold_n, hyp_n)
            row = {
                "id": item["id"],
                "gold": gold,
                "hyp": hyp,
                "gold_norm": gold_n,
                "hyp_norm": hyp_n,
                "S": s,
                "D": d,
                "I": ins,
                "W": w,
                "raw_wer": word_error_rate(gold, hyp),
                "norm_wer": word_error_rate(gold_n, hyp_n),
            }
            hyps.append(row)
            if i == 1 or i % 50 == 0 or i == len(items):
                print(f"  asr {i}/{len(items)}  {hyp!r}", flush=True)

    S = sum(r["S"] for r in hyps)
    D = sum(r["D"] for r in hyps)
    I = sum(r["I"] for r in hyps)
    W = sum(r["W"] for r in hyps)
    corpus = (S + D + I) / W if W else 0.0
    mean_raw = sum(r["raw_wer"] for r in hyps) / len(hyps)
    mean_norm = sum(r["norm_wer"] for r in hyps) / len(hyps)
    lines = [
        f"n={len(hyps)}",
        f"corpus WER (seedtts post_process) {corpus:.2%}  S={S} D={D} I={I} W={W}",
        f"mean sentence raw WER {mean_raw:.2%}",
        f"mean sentence norm WER {mean_norm:.2%}",
        f"wall {time.perf_counter() - t_all:.1f}s",
        "SEEDTTS_TURBO_OK",
    ]
    text = "\n".join(lines) + "\n"
    SUMMARY.write_text(text)
    with HYP_PATH.open("w") as f:
        for row in hyps:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(text, end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
