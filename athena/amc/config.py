from __future__ import annotations

import os
from dataclasses import dataclass, field


# Hugging Face IDs for a later, admin-approved fetch/convert step.
# Runtime never downloads these.
OFFICIAL_LLAMA_SOURCE = "meta-llama/Llama-3.1-8B-Instruct"
OFFICIAL_WHISPER_SOURCE = "OpenVINO/whisper-small-int8-ov"

# B580 12 GB: INT4 (~5 GB) can keep a 4 GB KV window for tutor chat.
# INT8 (~7.5 GB) cannot — 2 GB KV is enough for a quiz hint and still fits.
DEFAULT_KV_CACHE_GB_INT4 = 4
DEFAULT_KV_CACHE_GB_INT8 = 2


@dataclass
class AMCConfig:
    """Hardware and path settings for the Arc B580 box.

    device defaults to GPU (the discrete B580). The i5-10400F has no iGPU,
    so OpenVINO "GPU" is uniquely that card when the driver is up.
    """

    device: str = field(default_factory=lambda: os.environ.get("XILO_DEVICE", "GPU"))
    backend: str = field(
        default_factory=lambda: os.environ.get("XILO_AMC_BACKEND", "openvino")
    )
    llama_model_path: str = field(
        default_factory=lambda: os.environ.get(
            "XILO_LLAMA_PATH", "models/llama-3.1-8b-instruct-int4-ov"
        )
    )
    llama_int8_path: str = field(
        default_factory=lambda: os.environ.get(
            "XILO_LLAMA_INT8_PATH", "models/llama-3.1-8b-instruct-int8-ov"
        )
    )
    whisper_model_path: str = field(
        default_factory=lambda: os.environ.get(
            "XILO_WHISPER_PATH", "models/whisper-small-int8-ov"
        )
    )
    cache_dir: str = field(
        default_factory=lambda: os.environ.get("XILO_OV_CACHE", "models/ov_cache")
    )
    kv_cache_gb: int = DEFAULT_KV_CACHE_GB_INT4
    kv_cache_gb_int8: int = DEFAULT_KV_CACHE_GB_INT8
    allow_cpu_fallback: bool = False
    max_new_tokens_tutor: int = 512
    max_new_tokens_eval: int = 96
    tutor_temperature: float = 0.4
    eval_temperature: float = 0.1
    memory_keep_pairs: int = 3
    job_timeout_s: float = 180.0
    whisper_language: str = "<|en|>"

    @classmethod
    def mock(cls) -> AMCConfig:
        return cls(backend="mock", device="GPU", allow_cpu_fallback=False)
