#!/usr/bin/env python3
"""Live GPU smoke test for AMC on Intel Arc.

Uses TinyLlama INT4 + Whisper-tiny INT8. Does not load Llama 3.1 8B.
Proves: OpenVINO sees GPU, LLM generate, Whisper transcribe, exclusive switch.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from athena.amc import AMC, AMCConfig, ModelId  # noqa: E402


LLAMA_DIR = ROOT / "models" / "smoke" / "TinyLlama-1.1B-Chat-v1.0-int4-ov"
WHISPER_DIR = ROOT / "models" / "smoke" / "whisper-tiny-int8-ov"


def require_ir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(
            f"missing {label} IR at {path}. Run scripts/download_smoke_models.py first."
        )


def gpu_name() -> str:
    import openvino as ov

    core = ov.Core()
    devices = list(core.available_devices)
    if "GPU" not in devices:
        raise SystemExit(f"GPU not in OpenVINO devices: {devices}")
    try:
        return str(core.get_property("GPU", "FULL_DEVICE_NAME"))
    except Exception:
        return "GPU"


def main() -> int:
    require_ir(LLAMA_DIR, "TinyLlama")
    require_ir(WHISPER_DIR, "Whisper-tiny")
    name = gpu_name()
    print(f"device: GPU ({name})")

    cfg = AMCConfig(
        device="GPU",
        backend="openvino",
        llama_model_path=str(LLAMA_DIR),
        whisper_model_path=str(WHISPER_DIR),
        cache_dir=str(ROOT / "models" / "ov_cache" / "smoke"),
        kv_cache_gb=1,
        allow_cpu_fallback=False,
        max_new_tokens_tutor=32,
        job_timeout_s=600.0,
    )

    t0 = time.perf_counter()
    with AMC(cfg) as amc:
        print("1. Event T / TinyLlama on GPU")
        amc.enter_event_t(
            {"id": "smoke-s1", "title": "Fractions", "body": "A fraction is a/b."}
        )
        st = amc.status()
        print(f"   resident={st.resident_model} role={st.role} device={st.device}")
        if st.resident_model is not ModelId.LLAMA_INT4:
            raise SystemExit("expected Llama resident after Event T")
        if st.device != "GPU":
            raise SystemExit(f"expected GPU, got {st.device}")

        print("2. tutor_ask")
        reply = amc.tutor_ask("What is a fraction, in one sentence?")
        print(f"   reply: {reply[:240]!r}")
        if not str(reply).strip():
            raise SystemExit("empty llama reply")

        print("3. speak -> Whisper (must unload Llama)")
        pcm = [0.0] * 16000
        text = amc.transcribe(pcm)
        st = amc.status()
        print(f"   transcript: {text!r}")
        print(f"   resident={st.resident_model}")
        if st.resident_model is not ModelId.WHISPER:
            raise SystemExit("expected Whisper resident after transcribe")

        print("4. tutor_ask again (reload Llama, keep text memory)")
        reply2 = amc.tutor_ask("Say that again more simply.")
        st = amc.status()
        print(f"   reply: {reply2[:240]!r}")
        print(f"   resident={st.resident_model} memory_turns={st.memory_turns}")
        if st.resident_model is not ModelId.LLAMA_INT4:
            raise SystemExit("expected Llama resident after second ask")
        if st.memory_turns < 2:
            raise SystemExit("expected tutor memory to survive Whisper")

    elapsed = time.perf_counter() - t0
    print(f"SMOKE OK on GPU ({name}) in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
