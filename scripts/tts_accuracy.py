#!/usr/bin/env python3
"""TTS accuracy: OmniVoice GPU speak → Whisper GPU transcribe.

Exclusive slot: TTS speaks every line while loaded, unloads, then Whisper
reads the resampled wavs. Reports per-line WER and a mean. English first
(the 0.0% claim). A few other languages if Whisper IR is present.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from athena.amc import AMC, AMCConfig, ModelId  # noqa: E402
from athena.amc.cli import _write_wav  # noqa: E402
from athena.amc.exceptions import GpuDeadError, ModelPathError  # noqa: E402
from athena.amc.tts import TTSOVBackend  # noqa: E402
from athena.noise import word_error_rate  # noqa: E402

DOC_MODELS = Path("/home/light/Documents/projectxi1/models")


def _models() -> Path:
    local = ROOT / "models"
    if (local / "omnivoice-fp16-ov" / "forward.xml").is_file():
        return local
    return DOC_MODELS


def _gb(n: object) -> float:
    try:
        return float(n) / (1024**3)
    except (TypeError, ValueError):
        return 0.0


def gpu_usm() -> tuple[float, float]:
    import openvino as ov

    core = ov.Core()
    try:
        raw = dict(core.get_property("GPU", "GPU_MEMORY_STATISTICS"))
    except Exception:
        return 0.0, 0.0
    usm = 0.0
    clmem = 0.0
    for key, val in raw.items():
        name = str(key).lower()
        if "usm_device" in name:
            usm = _gb(val)
        elif "cl_mem" in name or name == "clmem":
            clmem = _gb(val)
    return usm, clmem


def resample_24k_to_16k(pcm: list[float]) -> list[float]:
    x = np.asarray(pcm, dtype=np.float32)
    if x.size == 0:
        return []
    n_out = max(1, int(round(x.size * 16000 / 24000)))
    t_in = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32).tolist()


def _cfg(models: Path) -> AMCConfig:
    return AMCConfig(
        device="GPU",
        backend="openvino",
        llama_model_path=str(models / "llama-3.1-8b-instruct-int4-ov"),
        llama_int8_path=str(models / "llama-3.1-8b-instruct-int8-ov"),
        whisper_model_path=str(models / "whisper-large-v3-turbo-int8-ov"),
        translate_model_path=str(models / "translategemma-4b-it-int8-ov"),
        tts_model_path=str(models / "omnivoice-fp16-ov"),
        cache_dir=str(models / "ov_cache"),
        kv_cache_gb=4,
        kv_cache_gb_int8=2,
        job_timeout_s=1200.0,
        allow_cpu_fallback=False,
        tts_num_step=16,
    )

OUT = Path("/tmp/omnivoice_acc")

# Classroom + a few harder English lines. Same Whisper language tag as AMC.
EN = [
    "A fraction is a part of a whole.",
    "The numerator is the top number.",
    "The denominator is the bottom number.",
    "Two fractions are equivalent when they name the same amount.",
    "What is a fraction, in one short sentence?",
    "One half plus one half equals one.",
    "Look back at the definition in this section.",
    "A number split into equal pieces.",
    "Please read this sentence clearly.",
    "The quick brown fox jumps over the lazy dog.",
]

# Whisper tag must match the spoken language. Short lines only.
OTHER = [
    ("es", "<|es|>", "Una fracción es una parte de un todo."),
    ("hi", "<|hi|>", "भिन्न एक पूर्ण का भाग है।"),
    ("de", "<|de|>", "Ein Bruch ist ein Teil eines Ganzen."),
]


def _norm_wer(ref: str, hyp: str) -> float:
    """WER after dropping punctuation so whole. vs whole is not an error."""

    def toks(s: str) -> str:
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
        return " ".join(s.split())

    return word_error_rate(toks(ref), toks(hyp))


def _print_row(i: int, lang: str, gold: str, hyp: str, audio_s: float) -> tuple[float, float]:
    raw = word_error_rate(gold, hyp)
    norm = _norm_wer(gold, hyp)
    print(f"{i:2d}  {lang:2s}  raw {raw:6.1%}  norm {norm:6.1%}  {audio_s:4.2f}s")
    print(f"    gold: {gold}")
    print(f"    hyp:  {hyp!r}")
    return raw, norm


def main() -> int:
    models = _models()
    cfg = _cfg(models)
    tts_dir = Path(cfg.tts_model_path)
    whisper_bin = Path(cfg.whisper_model_path) / "openvino_encoder_model.bin"
    if not (tts_dir / "forward.xml").is_file():
        print(f"missing OmniVoice IR at {tts_dir}", file=sys.stderr)
        return 2
    if not whisper_bin.is_file():
        print(f"missing Whisper IR at {whisper_bin}", file=sys.stderr)
        return 2

    import openvino as ov

    core = ov.Core()
    if "GPU" not in list(core.available_devices):
        print(f"GPU not in OpenVINO devices: {core.available_devices}", file=sys.stderr)
        return 2
    print(f"device: GPU ({core.get_property('GPU', 'FULL_DEVICE_NAME')})")
    print("runtime: OpenVINO GPU — no PyTorch")
    print(f"tts {tts_dir}")
    print(f"stt {cfg.whisper_model_path}")
    OUT.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, str, str]] = [("en", "<|en|>", t) for t in EN]
    items.extend(OTHER)

    t_all = time.perf_counter()
    wavs: list[tuple[str, str, str, list[float], int]] = []

    print("\n1. TTS hop — all lines, one resident, then unload")
    backend = TTSOVBackend(cfg)
    backend.load()
    usm, kv = gpu_usm()
    print(f"   USM {usm:.3f} GiB  cl_mem {kv:.3f} GiB")
    if backend._fwd is None:
        raise SystemExit("TTS failed to load")
    try:
        for i, (lang, _tag, text) in enumerate(items, 1):
            t0 = time.perf_counter()
            speech = backend.speak(text, language=lang)
            dt = time.perf_counter() - t0
            audio_s = len(speech.pcm) / speech.sample_rate
            path = OUT / f"{i:02d}_{lang}.wav"
            _write_wav(path, speech.pcm, speech.sample_rate)
            print(
                f"   speak {i:02d} {lang}  {dt:.2f}s  audio {audio_s:.2f}s  {path.name}"
            )
            wavs.append((lang, _tag, text, list(speech.pcm), speech.sample_rate))
    finally:
        backend.unload()
    after = gpu_usm()
    print(f"   TTS unloaded  usm_device={after[0]:.3f} GiB")
    if after[0] > 0.05:
        print("   WARNING: USM not back near zero after TTS unload")

    print("\n2. Whisper hop — resample 24 kHz→16 kHz, one resident")
    raws: list[float] = []
    norms: list[float] = []
    en_raws: list[float] = []
    en_norms: list[float] = []
    try:
        with AMC(cfg) as amc:
            # First transcribe loads Whisper. Stay on Whisper for the rest.
            for i, (lang, tag, gold, pcm, sr) in enumerate(wavs, 1):
                amc.config.whisper_language = tag
                pcm16 = resample_24k_to_16k(pcm) if sr != 16000 else pcm
                hyp = amc.transcribe(pcm16)
                if amc._tts.is_loaded():
                    raise SystemExit("TTS still loaded during Whisper hop")
                if amc.status().resident_model is not ModelId.WHISPER:
                    raise SystemExit("expected Whisper resident")
                audio_s = len(pcm) / sr if sr else 0.0
                raw, norm = _print_row(i, lang, gold, hyp, audio_s)
                raws.append(raw)
                norms.append(norm)
                if lang == "en":
                    en_raws.append(raw)
                    en_norms.append(norm)
            if amc._tts.is_loaded() or amc._llama.is_loaded():
                raise SystemExit("exclusive slot broken after STT batch")
    except (ModelPathError, GpuDeadError) as exc:
        print(exc, file=sys.stderr)
        return 2

    print("\n3. AMC exclusive round-trip on the original gold line")
    gold = EN[0]
    try:
        with AMC(cfg) as amc:
            amc.config.whisper_language = "<|en|>"
            speech = amc.talk(gold, language="en")
            if amc.status().resident_model is not None or amc._tts.is_loaded():
                raise SystemExit("TTS must unload after talk()")
            hyp = amc.transcribe(resample_24k_to_16k(speech.pcm))
            if amc._tts.is_loaded():
                raise SystemExit("TTS loaded during Whisper")
            raw, norm = _print_row(0, "en", gold, hyp, len(speech.pcm) / speech.sample_rate)
    except (ModelPathError, GpuDeadError) as exc:
        print(exc, file=sys.stderr)
        return 2

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    print("\nSUMMARY")
    print(f"  English n={len(en_raws)}  mean raw WER {_mean(en_raws):.1%}  mean norm WER {_mean(en_norms):.1%}")
    print(f"  All      n={len(raws)}  mean raw WER {_mean(raws):.1%}  mean norm WER {_mean(norms):.1%}")
    print(f"  exclusive gold raw {raw:.1%}  norm {norm:.1%}")
    final = gpu_usm()
    print(
        f"  wall {time.perf_counter() - t_all:.1f}s  "
        f"final usm_device={final[0]:.3f} GiB"
    )
    print("TTS_ACCURACY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
