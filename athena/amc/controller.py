from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from athena.amc.backends import (
    LlamaBackend,
    TranslatorBackend,
    WhisperBackend,
    build_backends,
)
from athena.amc.bus import EventBus
from athena.amc.config import AMCConfig
from athena.amc.exceptions import GpuDeadError, RoleError
from athena.amc.gpu import gpu_dead_message, is_gpu_dead_error, resolve_device
from athena.amc.memory import ConversationMemory
from athena.amc.roles import evaluator_system_prompt, tutor_system_prompt
from athena.amc.translate import (
    chunk_for_translator,
    format_translate_prompt,
    join_translations,
)
from athena.amc.tts import TTSBackend
from athena.amc.types import (
    AMCStatus,
    Event,
    JobKind,
    LlamaRole,
    ModelId,
    Speech,
    Streamer,
    Turn,
)


@dataclass
class _Job:
    kind: JobKind
    future: Future[Any]
    payload: dict[str, Any]


class AMC:
    """Exclusive GPU scheduler for Llama, Whisper, TranslateGemma, and TTS.

    All public methods enqueue work onto a single worker thread. A second
    request never loads a second model: it waits until the current job
    finishes, then the worker unloads and switches if needed.

    Conversation text is kept across a Whisper interrupt (speak button).
    KV cache is dropped on unload and the text history is fed back as
    system context when Llama is reloaded. Role end clears memory.

    STT / translator / TTS are stateless hops (product Event G): after the
    string is passed on, that model holds nothing. Only Llama keeps the
    same-section lesson thread.
    """

    def __init__(
        self,
        config: AMCConfig | None = None,
        *,
        llama: LlamaBackend | None = None,
        llama_int8: LlamaBackend | None = None,
        whisper: WhisperBackend | None = None,
        translator: TranslatorBackend | None = None,
        tts: TTSBackend | None = None,
        data_controller: Any | None = None,
    ) -> None:
        self.config = config or AMCConfig()
        self.device = resolve_device(self.config)
        if (
            llama is None
            or llama_int8 is None
            or whisper is None
            or translator is None
            or tts is None
        ):
            built = build_backends(self.config)
            llama = llama or built[0]
            llama_int8 = llama_int8 if llama_int8 is not None else built[1]
            whisper = whisper or built[2]
            translator = translator or built[3]
            tts = tts if tts is not None else built[4]
        self._llama_int4 = llama
        self._llama_int8 = llama_int8
        self._llama = llama  # tutor alias (INT4)
        self._whisper = whisper
        self._translator = translator
        self._tts = tts
        self._dc = data_controller
        self.bus = EventBus()

        self._resident: ModelId | None = None
        self._role: LlamaRole | None = None
        self._event: Event | None = None
        self._section: Any | None = None
        self._behavioral_level: int = 1
        self._kv_warm = False
        self._gpu_dead: str | None = None
        self._memory = ConversationMemory(keep_pairs=self.config.memory_keep_pairs)

        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._busy = False
        self._queue_depth = 0
        self._depth_lock = threading.Lock()
        self._worker = threading.Thread(
            target=self._run, name="amc-worker", daemon=True
        )
        self._stopped = threading.Event()
        self._worker.start()

    @classmethod
    def mock(cls) -> AMC:
        return cls(AMCConfig.mock())

    # ----- public API (all wait for the current job) -----

    def status(self) -> AMCStatus:
        return AMCStatus(
            device=self.device,
            resident_model=self._resident,
            role=self._role,
            event=self._event,
            busy=self._busy,
            queue_depth=self._queue_depth,
            kv_warm=self._kv_warm,
            memory_turns=len(self._memory),
        )

    def enter_event_t(self, section: Any, timeout: float | None = None) -> AMCStatus:
        return self._submit(
            JobKind.ENTER_EVENT,
            {"event": Event.T, "section": section},
            timeout,
        )

    def enter_event_e(
        self, behavioral_level: int = 1, timeout: float | None = None
    ) -> AMCStatus:
        return self._submit(
            JobKind.ENTER_EVENT,
            {"event": Event.E, "behavioral_level": behavioral_level},
            timeout,
        )

    def tutor_ask(
        self,
        question: str,
        *,
        streamer: Streamer | None = None,
        timeout: float | None = None,
    ) -> str:
        return self._submit(
            JobKind.ASK,
            {"question": question, "streamer": streamer},
            timeout,
        )

    def evaluator_hint(
        self,
        question: str,
        student_answer: str,
        *,
        allow_approximate: bool = False,
        timeout: float | None = None,
    ) -> str:
        return self._submit(
            JobKind.HINT,
            {
                "question": question,
                "student_answer": student_answer,
                "allow_approximate": allow_approximate,
            },
            timeout,
        )

    def transcribe(
        self, pcm_16k: list[float], timeout: float | None = None
    ) -> str:
        """Speak button — Whisper STT. Waits for Llama if it is mid-token."""

        return self._submit(JobKind.TRANSCRIBE, {"pcm": list(pcm_16k)}, timeout)

    def translate(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        timeout: float | None = None,
    ) -> str:
        """Event G translator hop. Chunks at 75% of 2K, sentence boundaries.

        Same path for student input → English and Llama output → selected
        language. Unloads after the string is returned (holds no knowledge).
        """

        return self._submit(
            JobKind.TRANSLATE,
            {
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
            },
            timeout,
        )

    def talk(
        self,
        text: str | None = None,
        *,
        language: str | None = None,
        timeout: float | None = None,
    ) -> Speech:
        """Talk button — OmniVoice TTS hop. Not Whisper (Whisper is STT).

        Speaks ``text``, or the last tutor reply if ``text`` is omitted.
        Unloads after the waveform is returned (holds no knowledge).
        """

        return self._submit(
            JobKind.TALK, {"text": text, "language": language}, timeout
        )

    def leave_role(self, timeout: float | None = None) -> list[Turn]:
        return self._submit(JobKind.LEAVE_ROLE, {}, timeout)

    def shutdown(self, timeout: float | None = None) -> None:
        if self._stopped.is_set():
            return
        self._submit(JobKind.SHUTDOWN, {}, timeout)
        self._stopped.set()
        self._jobs.put(None)
        self._worker.join(timeout=timeout or self.config.job_timeout_s)

    def close(self) -> None:
        self.shutdown()

    def __enter__(self) -> AMC:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    # ----- worker -----

    def _submit(self, kind: JobKind, payload: dict[str, Any], timeout: float | None) -> Any:
        if self._stopped.is_set() and kind is not JobKind.SHUTDOWN:
            raise RuntimeError("AMC is shut down")
        future: Future[Any] = Future()
        with self._depth_lock:
            self._queue_depth += 1
        self._jobs.put(_Job(kind=kind, future=future, payload=payload))
        self.bus.emit("job_queued", kind=kind.value)
        wait = self.config.job_timeout_s if timeout is None else timeout
        return future.result(timeout=wait)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            self._busy = True
            self.bus.emit("job_started", kind=job.kind.value)
            try:
                result = self._execute(job)
            except BaseException as exc:
                if is_gpu_dead_error(exc) and not isinstance(exc, GpuDeadError):
                    exc = self._poison_gpu(exc)
                job.future.set_exception(exc)
                self.bus.emit("job_failed", kind=job.kind.value, error=str(exc))
            else:
                job.future.set_result(result)
                self.bus.emit("job_finished", kind=job.kind.value)
            finally:
                self._busy = False
                with self._depth_lock:
                    self._queue_depth = max(0, self._queue_depth - 1)
                self._jobs.task_done()

    def _execute(self, job: _Job) -> Any:
        kind = job.kind
        p = job.payload
        if kind is JobKind.ENTER_EVENT:
            return self._enter_event(p["event"], p)
        if kind is JobKind.ASK:
            return self._ask(p["question"], p.get("streamer"))
        if kind is JobKind.HINT:
            return self._hint(
                p["question"],
                p["student_answer"],
                p["allow_approximate"],
            )
        if kind is JobKind.TRANSCRIBE:
            return self._transcribe(p["pcm"])
        if kind is JobKind.TRANSLATE:
            return self._translate(p["text"], p["source_lang"], p["target_lang"])
        if kind is JobKind.TALK:
            return self._talk(p.get("text"), p.get("language"))
        if kind is JobKind.LEAVE_ROLE:
            return self._leave_role()
        if kind is JobKind.SHUTDOWN:
            self._unload_resident()
            return None
        raise ValueError(f"unknown job {kind}")

    # ----- exclusive residency -----

    def _backend(
        self, model: ModelId
    ) -> LlamaBackend | WhisperBackend | TranslatorBackend | TTSBackend:
        if model is ModelId.LLAMA_INT4:
            return self._llama_int4
        if model is ModelId.LLAMA_INT8:
            return self._llama_int8
        if model is ModelId.TRANSLATE:
            return self._translator
        if model is ModelId.TTS:
            return self._tts
        return self._whisper

    @staticmethod
    def _is_llama(model: ModelId | None) -> bool:
        return model in (ModelId.LLAMA_INT4, ModelId.LLAMA_INT8)

    def _poison_gpu(self, exc: BaseException) -> GpuDeadError:
        """-5 OOM poisons the OpenCL queue; later compiles become -14.

        Do not load another model in this process. Unload what we can.
        """

        dead = GpuDeadError(gpu_dead_message(exc))
        self._gpu_dead = str(dead)
        try:
            self._unload_resident()
        except Exception:
            self._resident = None
        self.bus.emit("gpu_dead", error=str(exc))
        return dead

    def _ensure(self, model: ModelId) -> None:
        if self._gpu_dead:
            raise GpuDeadError(self._gpu_dead)
        if self._resident is model:
            return
        self._unload_resident()
        self.bus.emit("model_loading", model=model.value)
        backend = self._backend(model)
        try:
            backend.load()
        except BaseException as exc:
            try:
                backend.unload()
            except Exception:
                pass
            if is_gpu_dead_error(exc):
                raise self._poison_gpu(exc) from exc
            raise
        self._resident = model
        self.bus.emit("model_loaded", model=model.value)

    def _unload_resident(self) -> None:
        if self._resident is None:
            return
        leaving = self._resident
        backend = self._backend(leaving)
        if self._is_llama(leaving):
            backend.finish_session()  # type: ignore[union-attr]
            self._kv_warm = False
        backend.unload()
        self._resident = None
        self.bus.emit("model_unloaded", model=leaving.value)

    # ----- role / event -----

    def _enter_event(self, event: Event, payload: dict[str, Any]) -> AMCStatus:
        new_role = LlamaRole.for_event(event)
        if self._role is not None and self._role is not new_role:
            self._leave_role()
        self._event = event
        self._role = new_role
        if event is Event.T:
            self._section = payload.get("section")
            self._ensure(ModelId.LLAMA_INT4)
            self._start_tutor_session()
        else:
            self._behavioral_level = int(payload.get("behavioral_level", 1))
            self._ensure(ModelId.LLAMA_INT8)
            # Evaluator is per-question; no long chat cache.
            self._llama_int8.finish_session()
            self._kv_warm = False
        self.bus.emit("role_changed", role=new_role.value, event=event.value)
        return self.status()

    def _start_tutor_session(self) -> None:
        prompt = tutor_system_prompt(self._section, self._memory)
        self._llama_int4.start_session(prompt)
        self._kv_warm = True

    def _ask(self, question: str, streamer: Streamer | None) -> str:
        if self._role is not LlamaRole.TUTOR or self._event is not Event.T:
            raise RoleError("tutor_ask requires Event T / TUTOR role")
        self._ensure(ModelId.LLAMA_INT4)
        if not self._kv_warm:
            self._start_tutor_session()
        self._memory.add_user(question)
        reply = self._llama_int4.generate(
            question,
            max_new_tokens=self.config.max_new_tokens_tutor,
            temperature=self.config.tutor_temperature,
            streamer=streamer,
        )
        self._memory.add_assistant(reply)
        return reply

    def _hint(
        self,
        question: str,
        student_answer: str,
        allow_approximate: bool,
    ) -> str:
        if self._role is not LlamaRole.EVALUATOR or self._event is not Event.E:
            raise RoleError("evaluator_hint requires Event E / EVALUATOR role")
        self._ensure(ModelId.LLAMA_INT8)
        system = evaluator_system_prompt(
            behavioral_level=self._behavioral_level,
            question=question,
            student_answer=student_answer,
            allow_approximate=allow_approximate,
        )
        self._llama_int8.start_session(system)
        self._kv_warm = True
        hint = self._llama_int8.generate(
            "Give one hint. Do not include the answer.",
            max_new_tokens=self.config.max_new_tokens_eval,
            temperature=self.config.eval_temperature,
        )
        self._llama_int8.finish_session()
        self._kv_warm = False
        return hint

    def _transcribe(self, pcm: list[float]) -> str:
        self._ensure(ModelId.WHISPER)
        return self._whisper.transcribe(
            pcm, language=self.config.whisper_language
        )

    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not (text or "").strip():
            return ""
        self._ensure(ModelId.TRANSLATE)
        cfg = self.config
        chunks = chunk_for_translator(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
            count=self._translator.token_count,
            max_input_tokens=cfg.translate_max_input_tokens,
            fill_ratio=cfg.translate_fill_ratio,
        )
        hard = cfg.translate_max_input_tokens
        fill_cap = max(32, int(hard * cfg.translate_fill_ratio))
        pieces: list[str] = []
        try:
            for chunk in chunks:
                prompt = format_translate_prompt(
                    chunk.text, source_lang=source_lang, target_lang=target_lang
                )
                used = self._translator.token_count(prompt)
                max_new = min(
                    cfg.max_new_tokens_translate,
                    fill_cap,
                    max(32, hard - used),
                )
                pieces.append(
                    self._translator.generate(prompt, max_new_tokens=max_new)
                )
            return join_translations(chunks, pieces)
        finally:
            # Product: translator holds no knowledge after pass-on.
            self._unload_resident()

    def _talk(self, text: str | None, language: str | None) -> Speech:
        spoken = (text or "").strip()
        if not spoken:
            for turn in reversed(self._memory.turns):
                if turn.role == "assistant" and turn.text.strip():
                    spoken = turn.text.strip()
                    break
        if not spoken:
            raise ValueError("talk needs text or a last assistant reply")
        self._ensure(ModelId.TTS)
        try:
            return self._tts.speak(spoken, language=language)
        finally:
            # Product: TTS holds no knowledge after pass-on.
            self._unload_resident()

    def _leave_role(self) -> list[Turn]:
        snapshot = self._memory.clear()
        if self._dc is not None and snapshot:
            saver = getattr(self._dc, "save_last_chats", None)
            if callable(saver):
                saver(snapshot)
        if self._is_llama(self._resident):
            self._backend(self._resident).finish_session()  # type: ignore[union-attr]
            self._kv_warm = False
        self._role = None
        self._event = None
        self._section = None
        self.bus.emit("role_ended")
        return snapshot
