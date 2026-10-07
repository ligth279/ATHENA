from __future__ import annotations

import os
from dataclasses import dataclass, field


# Hugging Face IDs for a later, admin-approved fetch/convert step.
# Runtime never downloads these.
OFFICIAL_LLAMA_SOURCE = "meta-llama/Llama-3.1-8B-Instruct"
OFFICIAL_WHISPER_SOURCE = "OpenVINO/whisper-large-v3-turbo-int8-ov"
OFFICIAL_TRANSLATE_SOURCE = "google/translategemma-4b-it"

# TranslateGemma card: 2K input tokens. Feed at most 75% so template +
# source + generated translation stay inside the window.
DEFAULT_TRANSLATE_MAX_INPUT = 2048
DEFAULT_TRANSLATE_FILL_RATIO = 0.75

# B580 12 GB: INT4 (~5 GB) can keep a 4 GB KV window for tutor chat.
# INT8 (~7.5 GB) cannot — 2 GB KV is enough for a quiz hint and still fits.
# TranslateGemma is Gemma 3 (config max_position 131072) but the card is 2K.
# VLMPipeline with cache_size=0 will try to size KV for that window and
# CL_OUT_OF_RESOURCES on generate even for a short sentence.
DEFAULT_KV_CACHE_GB_INT4 = 4
DEFAULT_KV_CACHE_GB_INT8 = 2
DEFAULT_KV_CACHE_GB_TRANSLATE = 1


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
            "XILO_WHISPER_PATH", "models/whisper-large-v3-turbo-int8-ov"
        )
    )
    translate_model_path: str = field(
        default_factory=lambda: os.environ.get(
            "XILO_TRANSLATE_PATH", "models/translategemma-4b-it-int8-ov"
        )
    )
    tts_model_path: str = field(
        default_factory=lambda: os.environ.get(
            "XILO_TTS_PATH", "models/omnivoice-fp16-ov"
        )
    )
    cache_dir: str = field(
        default_factory=lambda: os.environ.get("XILO_OV_CACHE", "models/ov_cache")
    )
    kv_cache_gb: int = DEFAULT_KV_CACHE_GB_INT4
    kv_cache_gb_int8: int = DEFAULT_KV_CACHE_GB_INT8
    kv_cache_gb_translate: int = DEFAULT_KV_CACHE_GB_TRANSLATE
    allow_cpu_fallback: bool = False
    max_new_tokens_tutor: int = 512
    max_new_tokens_eval: int = 96
    tutor_temperature: float = 0.4
    eval_temperature: float = 0.1
    memory_keep_pairs: int = 3
    job_timeout_s: float = 180.0
    whisper_language: str = "<|en|>"
    translate_max_input_tokens: int = DEFAULT_TRANSLATE_MAX_INPUT
    translate_fill_ratio: float = DEFAULT_TRANSLATE_FILL_RATIO
    max_new_tokens_translate: int = 512
    tts_num_step: int = 16  # OmniVoice default 32; 16 is a faster B580 hop
    # numpy = recommended (Seed-TTS 1.69%, hot 0.38 s/line).
    # gpu = experimental FP32 CFG graph (1.72%, hot 0.98 s/line).
    tts_cfg: str = "numpy"
    tts_guidance_scale: float = 2.0
    tts_t_shift: float = 0.1
    tts_layer_penalty: float = 5.0
    tts_position_temperature: float = 5.0

    @classmethod
    def mock(cls) -> AMCConfig:
        return cls(backend="mock", device="GPU", allow_cpu_fallback=False)
