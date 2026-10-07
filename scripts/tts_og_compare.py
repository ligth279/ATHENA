#!/usr/bin/env python3
"""Same-line WER: original OmniVoice (CPU PyTorch) vs AMC OpenVINO GPU hop.

Both use num_step=16 and the same Whisper large-v3 turbo on GPU.
Does not pip-install omnivoice (that pulls CUDA torch). Uses the local clone.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import importlib.util  # noqa: E402

from athena.amc import AMC, AMCConfig  # noqa: E402
from athena.amc.cli import _write_wav  # noqa: E402
from athena.noise import word_error_rate  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "tts_accuracy", ROOT / "scripts" / "tts_accuracy.py"
)
_acc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_acc)
EN = _acc.EN
OTHER = _acc.OTHER
_cfg = _acc._cfg
_models = _acc._models
_norm_wer = _acc._norm_wer
resample_24k_to_16k = _acc.resample_24k_to_16k

OV_DIR = Path("/tmp/omnivoice_acc")
OG_DIR = Path("/tmp/omnivoice_og")
CLONE = Path("/tmp/OmniVoice-0.2.1")
SRC = Path("/home/light/Documents/projectxi1/models/omnivoice")


def _og_generate(items: list[tuple[str, str, str]]) -> list[Path]:
    sys.path.insert(0, str(CLONE))
    import torch
    from transformers import AutoTokenizer, HiggsAudioV2TokenizerModel
    from omnivoice import OmniVoice
    from omnivoice.utils.duration import RuleDurationEstimator

    print("loading original OmniVoice on CPU torch (export-style, no CUDA)", flush=True)
    m = OmniVoice.from_pretrained(
        str(SRC),
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="sdpa",
        train=True,
    )
    m.to("cpu")
    m.text_tokenizer = AutoTokenizer.from_pretrained(str(SRC), local_files_only=True)
    tok = HiggsAudioV2TokenizerModel.from_pretrained(
        str(SRC / "audio_tokenizer"), local_files_only=True
    )
    tok.to("cpu")
    tok.eval()
    m.audio_tokenizer = tok
    m.duration_estimator = RuleDurationEstimator()
    m.sampling_rate = 24000
    m.eval()
    OG_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with torch.inference_mode():
        for i, (lang, _tag, text) in enumerate(items, 1):
            t0 = time.perf_counter()
            wavs = m.generate(text, language=lang, num_step=16)
            w = wavs[0]
            if hasattr(w, "detach"):
                w = w.detach().cpu().numpy()
            pcm = np.asarray(w, dtype=np.float32).reshape(-1)
            path = OG_DIR / f"{i:02d}_{lang}.wav"
            _write_wav(path, pcm.tolist(), 24000)
            print(
                f"  og {i:02d} {lang}  {time.perf_counter() - t0:.1f}s  "
                f"audio {len(pcm)/24000:.2f}s  {path.name}",
                flush=True,
            )
            paths.append(path)
    return paths


def _read_wav(path: Path) -> tuple[list[float], int]:
    import wave
    import struct

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        sw = wf.getsampwidth()
    if sw != 2:
        raise SystemExit(f"expected 16-bit wav: {path}")
    samples = struct.unpack("<" + "h" * n, raw)
    pcm = [x / 32768.0 for x in samples]
    return pcm, sr


def main() -> int:
    if not CLONE.is_dir() or not (SRC / "model.safetensors").is_file():
        print("missing original OmniVoice clone or weights", file=sys.stderr)
        return 2
    items: list[tuple[str, str, str]] = [("en", "<|en|>", t) for t in EN]
    items.extend(OTHER)
    ov_wavs = [OV_DIR / f"{i:02d}_{lang}.wav" for i, (lang, _, _) in enumerate(items, 1)]
    if not all(p.is_file() for p in ov_wavs):
        print(f"missing OV wavs in {OV_DIR}; run scripts/tts_accuracy.py first", file=sys.stderr)
        return 2

    t_all = time.perf_counter()
    og_wavs = _og_generate(items)

    cfg = _cfg(_models())
    print("\nWhisper GPU on both wav sets", flush=True)
    rows: list[tuple[str, float, float, float, float]] = []
    with AMC(cfg) as amc:
        for i, ((lang, tag, gold), ov_p, og_p) in enumerate(zip(items, ov_wavs, og_wavs), 1):
            amc.config.whisper_language = tag
            ov_pcm, ov_sr = _read_wav(ov_p)
            og_pcm, og_sr = _read_wav(og_p)
            ov_h = amc.transcribe(resample_24k_to_16k(ov_pcm) if ov_sr != 16000 else ov_pcm)
            og_h = amc.transcribe(resample_24k_to_16k(og_pcm) if og_sr != 16000 else og_pcm)
            ov_raw, ov_n = word_error_rate(gold, ov_h), _norm_wer(gold, ov_h)
            og_raw, og_n = word_error_rate(gold, og_h), _norm_wer(gold, og_h)
            print(f"{i:2d} {lang}")
            print(f"    gold {gold}")
            print(f"    OV   raw {ov_raw:6.1%}  {ov_h!r}")
            print(f"    OG   raw {og_raw:6.1%}  {og_h!r}")
            rows.append((lang, ov_raw, ov_n, og_raw, og_n))

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    en = [r for r in rows if r[0] == "en"]
    print("\nSUMMARY  (same 13 lines, num_step=16, same Whisper turbo)")
    print(
        f"  English OV  raw {mean([r[1] for r in en]):.1%}  norm {mean([r[2] for r in en]):.1%}"
    )
    print(
        f"  English OG  raw {mean([r[3] for r in en]):.1%}  norm {mean([r[4] for r in en]):.1%}"
    )
    print(
        f"  All      OV  raw {mean([r[1] for r in rows]):.1%}  norm {mean([r[2] for r in rows]):.1%}"
    )
    print(
        f"  All      OG  raw {mean([r[3] for r in rows]):.1%}  norm {mean([r[4] for r in rows]):.1%}"
    )
    print(f"  wall {time.perf_counter() - t_all:.1f}s")
    print("OG_COMPARE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
