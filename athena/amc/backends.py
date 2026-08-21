from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from athena.amc.config import AMCConfig
from athena.amc.exceptions import ModelPathError
from athena.amc.gpu import compile_cache_kwargs, resolve_device
from athena.amc.types import Streamer


def _as_text(result: Any) -> str:
    """OpenVINO 2026 generate() returns DecodedResults; older returns str."""

    if result is None:
        return ""
    texts = getattr(result, "texts", None)
    if texts:
        return str(texts[0])
    return str(result).strip()


class LlamaBackend(ABC):
    name = "llama"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def start_session(self, system_prompt: str) -> None: ...

    @abstractmethod
    def finish_session(self) -> None: ...

    @abstractmethod
    def generate(
        self,
        user_text: str,
        *,
        max_new_tokens: int,
        temperature: float,
        streamer: Streamer | None = None,
    ) -> str: ...


class WhisperBackend(ABC):
    name = "whisper"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def transcribe(self, pcm_16k: list[float], *, language: str) -> str: ...


class MockLlamaBackend(LlamaBackend):
    """No GPU. Used by tests and by --backend mock."""

    def __init__(self, name: str = "llama") -> None:
        self.name = name
        self._loaded = False
        self._session = False
        self.system_prompt = ""
        self.calls: list[str] = []

    def load(self) -> None:
        self._loaded = True
        self.calls.append("load")

    def unload(self) -> None:
        self._loaded = False
        self._session = False
        self.calls.append("unload")

    def is_loaded(self) -> bool:
        return self._loaded

    def start_session(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self._session = True
        self.calls.append("start_session")

    def finish_session(self) -> None:
        self._session = False
        self.calls.append("finish_session")

    def generate(
        self,
        user_text: str,
        *,
        max_new_tokens: int,
        temperature: float,
        streamer: Streamer | None = None,
    ) -> str:
        self.calls.append(f"generate:{user_text}")
        if "HINT_REQUEST" in user_text or "hint" in user_text.lower():
            text = "Look back at the definition in this section. What quantity is being asked for?"
        else:
            text = f"[tutor mock] {user_text}"
        if streamer:
            streamer(text)
        return text


class MockWhisperBackend(WhisperBackend):
    def __init__(self) -> None:
        self._loaded = False
        self.calls: list[str] = []
        self.last_pcm_len = 0

    def load(self) -> None:
        self._loaded = True
        self.calls.append("load")

    def unload(self) -> None:
        self._loaded = False
        self.calls.append("unload")

    def is_loaded(self) -> bool:
        return self._loaded

    def transcribe(self, pcm_16k: list[float], *, language: str) -> str:
        self.last_pcm_len = len(pcm_16k)
        self.calls.append("transcribe")
        return "what does this paragraph mean"


_LLAMA_IR_FILES = (
    "openvino_model.xml",
    "openvino_model.bin",
    "openvino_tokenizer.xml",
    "openvino_tokenizer.bin",
)
_WHISPER_IR_FILES = (
    "openvino_encoder_model.xml",
    "openvino_encoder_model.bin",
    "openvino_decoder_model.xml",
    "openvino_decoder_model.bin",
)


def _require_ir_dir(
    path: str,
    label: str,
    required: tuple[str, ...] = _LLAMA_IR_FILES,
) -> Path:
    root = Path(path)
    if not root.is_dir():
        raise ModelPathError(
            f"{label} IR directory does not exist: {root}. "
            "Convert or fetch the OpenVINO model after administrator approval."
        )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        parts = sorted(p.name for p in root.glob("*.part"))
        extra = f" Incomplete download ({', '.join(parts)})." if parts else ""
        raise ModelPathError(
            f"{label} IR is incomplete at {root}: missing {', '.join(missing)}.{extra} "
            "Finish the fetch or convert before loading."
        )
    return root


class LlamaOVBackend(LlamaBackend):
    def __init__(
        self,
        config: AMCConfig,
        *,
        model_path: str,
        kv_cache_gb: int,
        cache_subdir: str,
        name: str = "llama",
    ) -> None:
        self.config = config
        self.name = name
        self.model_path = model_path
        self.kv_cache_gb = kv_cache_gb
        self.cache_subdir = cache_subdir
        self.device = resolve_device(config)
        self._pipe: Any = None
        self._in_chat = False

    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        if self._pipe is not None:
            return
        model_dir = _require_ir_dir(self.model_path, f"Llama 3.1 ({self.name})")
        import openvino_genai as ov_genai

        kwargs: dict[str, Any] = dict(
            compile_cache_kwargs(self.config, self.device, self.cache_subdir)
        )
        scheduler = ov_genai.SchedulerConfig()
        scheduler.cache_size = int(self.kv_cache_gb)
        kwargs["scheduler_config"] = scheduler
        self._pipe = ov_genai.LLMPipeline(str(model_dir), self.device, **kwargs)

    def unload(self) -> None:
        if self._pipe is None:
            return
        try:
            if self._in_chat:
                self._pipe.finish_chat()
        except Exception:
            pass
        self._in_chat = False
        self._pipe = None
        gc.collect()

    def start_session(self, system_prompt: str) -> None:
        if self._pipe is None:
            raise RuntimeError("llama pipeline is not loaded")
        if self._in_chat:
            self._pipe.finish_chat()
        self._pipe.start_chat(system_prompt)
        self._in_chat = True

    def finish_session(self) -> None:
        if self._pipe is not None and self._in_chat:
            self._pipe.finish_chat()
        self._in_chat = False

    def generate(
        self,
        user_text: str,
        *,
        max_new_tokens: int,
        temperature: float,
        streamer: Streamer | None = None,
    ) -> str:
        if self._pipe is None:
            raise RuntimeError("llama pipeline is not loaded")

        cfg = self._pipe.get_generation_config()
        cfg.max_new_tokens = int(max_new_tokens)
        if temperature and temperature > 0:
            cfg.temperature = float(temperature)
            cfg.do_sample = True
        else:
            cfg.do_sample = False
        if hasattr(cfg, "return_decoded_results"):
            cfg.return_decoded_results = True
        result = self._pipe.generate(user_text, cfg, streamer)
        return _as_text(result)


class WhisperOVBackend(WhisperBackend):
    def __init__(self, config: AMCConfig) -> None:
        self.config = config
        self.device = resolve_device(config)
        self._pipe: Any = None

    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        if self._pipe is not None:
            return
        model_dir = _require_ir_dir(
            self.config.whisper_model_path, "Whisper", _WHISPER_IR_FILES
        )
        import openvino_genai as ov_genai

        kwargs = compile_cache_kwargs(self.config, self.device)
        self._pipe = ov_genai.WhisperPipeline(str(model_dir), self.device, **kwargs)

    def unload(self) -> None:
        self._pipe = None
        gc.collect()

    def transcribe(self, pcm_16k: list[float], *, language: str) -> str:
        if self._pipe is None:
            raise RuntimeError("whisper pipeline is not loaded")
        result = self._pipe.generate(
            list(pcm_16k),
            max_new_tokens=200,
            language=language,
            task="transcribe",
        )
        return _as_text(result)


def build_backends(
    config: AMCConfig,
) -> tuple[LlamaBackend, LlamaBackend, WhisperBackend]:
    if config.backend == "mock":
        return (
            MockLlamaBackend("llama_int4"),
            MockLlamaBackend("llama_int8"),
            MockWhisperBackend(),
        )
    int4 = LlamaOVBackend(
        config,
        model_path=config.llama_model_path,
        kv_cache_gb=config.kv_cache_gb,
        cache_subdir="int4",
        name="llama_int4",
    )
    int8 = LlamaOVBackend(
        config,
        model_path=config.llama_int8_path,
        kv_cache_gb=config.kv_cache_gb_int8,
        cache_subdir="int8",
        name="llama_int8",
    )
    return int4, int8, WhisperOVBackend(config)
