#!/usr/bin/env python3
"""Live OmniVoice TTS hop on the B580: OpenVINO GPU, exclusive slot, unload.

PyTorch is not used. forward.xml (embed+LLM+heads) and decode.xml run on GPU.
Optional Whisper WER: resample 24 kHz → 16 kHz, then STT hop after TTS unload.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOC_MODELS = Path("/home/light/Documents/projectxi1/models")

from athena.amc import AMC, AMCConfig, ModelId  # noqa: E402
from athena.amc.cli import _write_wav  # noqa: E402
from athena.amc.exceptions import GpuDeadError, ModelPathError  # noqa: E402
from athena.amc.tts import TTSOVBackend  # noqa: E402
from athena.noise import word_error_rate  # noqa: E402

GOLD = "A fraction is a part of a whole."


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


def main() -> int:
    models = _models()
    tts_dir = models / "omnivoice-fp16-ov"
    if not (tts_dir / "forward.xml").is_file():
        print(f"missing OmniVoice IR at {tts_dir}", file=sys.stderr)
        return 2

    import openvino as ov

    core = ov.Core()
    devices = list(core.available_devices)
    if "GPU" not in devices:
        print(f"GPU not in OpenVINO devices: {devices}", file=sys.stderr)
        return 2
    name = core.get_property("GPU", "FULL_DEVICE_NAME")
    print(f"device: GPU ({name})")
    print(f"ir:     {tts_dir}")
    print("runtime: OpenVINO GPU — no PyTorch")

    idle_usm, idle_kv = gpu_usm()
    print(f"idle usm_device={idle_usm:.3f} GiB  cl_mem={idle_kv:.3f} GiB")

    cfg = _cfg(models)
    t_all = time.perf_counter()

    print("1. backend boot / speak / unload (GPU graphs)")
    backend = TTSOVBackend(cfg)
    t0 = time.perf_counter()
    backend.load()
    boot_s = time.perf_counter() - t0
    usm_boot, kv_boot = gpu_usm()
    t1 = time.perf_counter()
    speech = backend.speak(GOLD, language="en")
    run_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    backend.unload()
    unload_s = time.perf_counter() - t2
    usm_after, kv_after = gpu_usm()
    audio_s = len(speech.pcm) / speech.sample_rate if speech.sample_rate else 0.0
    print(
        f"   boot {boot_s:.2f}s  run {run_s:.2f}s  unload {unload_s:.2f}s  "
        f"audio {audio_s:.2f}s ({len(speech.pcm)} samples @ {speech.sample_rate} Hz)"
    )
    print(
        f"   USM at boot {usm_boot:.3f} GiB  cl_mem {kv_boot:.3f} GiB  "
        f"after unload {usm_after:.3f} GiB"
    )
    wav = Path("/tmp/omnivoice_amc_talk.wav")
    _write_wav(wav, speech.pcm, speech.sample_rate)
    print(f"   wav {wav}")

    try:
        with AMC(cfg) as amc:
            print("2. AMC.talk() exclusive hop (cached compile)")
            t3 = time.perf_counter()
            speech2 = amc.talk(GOLD, language="en")
            talk_s = time.perf_counter() - t3
            st = amc.status()
            print(
                f"   wall {talk_s:.2f}s  samples {len(speech2.pcm)}  "
                f"resident={st.resident_model}"
            )
            if st.resident_model is not None:
                raise SystemExit("TTS must unload after talk()")
            if amc._tts.is_loaded() or amc._llama.is_loaded() or amc._whisper.is_loaded():
                raise SystemExit("exclusive slot still occupied after TTS hop")

            whisper_ir = Path(cfg.whisper_model_path) / "openvino_encoder_model.bin"
            if whisper_ir.is_file():
                print("3. Whisper STT hop on resampled 16 kHz (after TTS unload)")
                pcm16 = resample_24k_to_16k(speech2.pcm)
                t4 = time.perf_counter()
                hyp = amc.transcribe(pcm16)
                stt_s = time.perf_counter() - t4
                wer = word_error_rate(GOLD, hyp)
                print(f"   hyp: {hyp!r}  wall {stt_s:.2f}s  WER {wer:.1%}")
                if amc._tts.is_loaded():
                    raise SystemExit("TTS still loaded during Whisper hop")
                if amc.status().resident_model is not ModelId.WHISPER:
                    raise SystemExit("expected Whisper resident during STT")
            else:
                print("3. skip Whisper WER (no Whisper IR)")
    except (ModelPathError, GpuDeadError) as exc:
        print(exc, file=sys.stderr)
        return 2

    final = gpu_usm()
    print(
        f"wall {time.perf_counter() - t_all:.2f}s  "
        f"final usm_device={final[0]:.3f} GiB  cl_mem={final[1]:.3f} GiB"
    )
    print(
        "METRICS "
        f"boot={boot_s:.2f} run={run_s:.2f} unload={unload_s:.2f} "
        f"usm_boot={usm_boot:.3f} kv={kv_boot:.3f} usm_after={usm_after:.3f}"
    )
    print("TTS_LIVE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
