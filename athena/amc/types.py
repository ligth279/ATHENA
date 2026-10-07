from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class Event(str, Enum):
    """Athena event modes from the v6 docs."""

    T = "T"  # learning / tutor
    E = "E"  # quiz / evaluator


class LlamaRole(str, Enum):
    TUTOR = "TUTOR"
    EVALUATOR = "EVALUATOR"

    @classmethod
    def for_event(cls, event: Event) -> LlamaRole:
        if event is Event.T:
            return cls.TUTOR
        if event is Event.E:
            return cls.EVALUATOR
        raise ValueError(f"no llama role for event {event}")


class ModelId(str, Enum):
    LLAMA_INT4 = "llama_int4"  # tutor — 4 GB KV
    LLAMA_INT8 = "llama_int8"  # evaluator — 2 GB KV
    WHISPER = "whisper"
    TRANSLATE = "translate"  # Event G hop — TranslateGemma 4B INT8
    TTS = "tts"  # Event G hop — OmniVoice FP16, unload after speak


class JobKind(str, Enum):
    ENTER_EVENT = "enter_event"
    ASK = "ask"
    HINT = "hint"
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"
    TALK = "talk"
    LEAVE_ROLE = "leave_role"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class Turn:
    role: str  # "user" | "assistant"
    text: str


@dataclass(frozen=True)
class Speech:
    """24 kHz PCM from the TTS hop. Not Whisper (16 kHz STT)."""

    pcm: list[float]
    sample_rate: int = 24000


@dataclass
class AMCStatus:
    device: str
    resident_model: ModelId | None
    role: LlamaRole | None
    event: Event | None
    busy: bool
    queue_depth: int
    kv_warm: bool
    memory_turns: int


Streamer = Callable[[str], bool]
EventHandler = Callable[..., Any]
