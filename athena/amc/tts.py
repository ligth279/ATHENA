"""OmniVoice FP16 OpenVINO TTS hop.

Runtime is OpenVINO device ``GPU``. PyTorch is not imported here.

Neural work lives in two compiled graphs, both on the B580:
  forward.xml  — embed preprocess + Qwen3 + audio heads (one unmask step)
  decode.xml   — Higgs acoustic decoder (codes → 24 kHz wav)

Python/numpy only: BPE lookup, duration heuristic, Gumbel+topk scatter,
and (default) CFG log-softmax + argmax.

``config.tts_cfg="gpu"`` compiles an experimental FP32 CFG graph. Seed-TTS
1.72% vs NumPy 1.69%; hot TTS 0.98 vs 0.38 s/line. Keep numpy as the hop.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from athena.amc.config import AMCConfig
from athena.amc.exceptions import ModelPathError
from athena.amc.gpu import compile_cache_kwargs, drop_pipeline, resolve_device
from athena.amc.tts_duration import RuleDurationEstimator
from athena.amc.tts_tokenizer import JsonBpeTokenizer
from athena.amc.types import Speech


CODEBOOKS = 8
AUDIO_MASK_ID = 1024
AUDIO_VOCAB = 1025
SAMPLE_RATE = 24000
HOP_LENGTH = 960  # decode.xml: T frames → T * 960 samples
FRAME_RATE = SAMPLE_RATE // HOP_LENGTH  # 25 Hz
MAX_TARGET_FRAMES = 375  # 15 s cap on the exclusive hop
_REF_TEXT = "Nice to meet you."
_REF_FRAMES = 25

_TTS_IR_FILES = (
    "forward.xml",
    "forward.bin",
    "decode.xml",
    "decode.bin",
    "tokenizer.json",
)

_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)


class TTSBackend(ABC):
    name = "tts"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def speak(self, text: str, *, language: str | None = None) -> Speech: ...


class MockTTSBackend(TTSBackend):
    """No GPU. Used by tests and by --backend mock."""

    def __init__(self) -> None:
        self._loaded = False
        self.calls: list[str] = []
        self.last_text = ""
        self.last_language: str | None = None

    def load(self) -> None:
        self._loaded = True
        self.calls.append("load")

    def unload(self) -> None:
        self._loaded = False
        self.calls.append("unload")

    def is_loaded(self) -> bool:
        return self._loaded

    def speak(self, text: str, *, language: str | None = None) -> Speech:
        self.last_text = text
        self.last_language = language
        self.calls.append("speak")
        n = max(24, min(2400, len(text) * 24))
        pcm = [0.0] * n
        return Speech(pcm=pcm, sample_rate=SAMPLE_RATE)


def _require_tts_ir(path: str) -> Path:
    root = Path(path)
    if not root.is_dir():
        raise ModelPathError(
            f"OmniVoice IR directory does not exist: {root}. "
            "Convert with scripts/convert_omnivoice_fp16.py after administrator approval."
        )
    missing = [name for name in _TTS_IR_FILES if not (root / name).is_file()]
    if missing:
        raise ModelPathError(
            f"OmniVoice IR is incomplete at {root}: missing {', '.join(missing)}."
        )
    return root


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def _time_steps(num_step: int, t_shift: float) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, num_step + 1, dtype=np.float64)
    return t_shift * ts / (1.0 + (t_shift - 1.0) * ts)


def _unmask_schedule(target_len: int, num_step: int, t_shift: float) -> list[int]:
    timesteps = _time_steps(num_step, t_shift)
    total_mask = target_len * CODEBOOKS
    rem = total_mask
    sched: list[int] = []
    for step in range(num_step):
        if step == num_step - 1:
            num = rem
        else:
            num = min(
                math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                rem,
            )
        sched.append(int(num))
        rem -= int(num)
    return sched


def _combine_text(text: str, ref_text: str | None = None) -> str:
    if ref_text:
        full = ref_text.strip() + " " + text.strip()
    else:
        full = text.strip()
    full = re.sub(r"[\r\n]+", "", full)
    full = full.replace("\uff08", "(").replace("\uff09", ")")
    full = re.sub(r"[ \t]+", " ", full)
    chinese = r"[\u4e00-\u9fff]"
    full = re.sub(rf"(?<={chinese})\s+|\s+(?={chinese})", "", full)
    return full


def _gumbel(scores: np.ndarray, temperature: float, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(scores.shape, dtype=np.float32).astype(np.float64)
    noise = -np.log(-np.log(u + 1e-10) + 1e-10)
    return scores / temperature + noise


def _build_cfg_model(guidance: float):
    """OpenVINO graph: CFG log-softmax + argmax on GPU. Vocab 1025, mask 1024."""
    import openvino as ov
    from openvino import opset13 as ops

    c_p = ops.parameter(
        ov.PartialShape([1, CODEBOOKS, -1, AUDIO_VOCAB]), ov.Type.f32, "c_logits"
    )
    u_p = ops.parameter(
        ov.PartialShape([1, CODEBOOKS, -1, AUDIO_VOCAB]), ov.Type.f32, "u_logits"
    )
    c = ops.convert(c_p, ov.Type.f32)
    u = ops.convert(u_p, ov.Type.f32)
    c_log = ops.log_softmax(c, -1)
    if guidance != 0.0:
        u_log = ops.log_softmax(u, -1)
        g = ops.constant(np.array(guidance, dtype=np.float32))
        mixed = ops.add(c_log, ops.multiply(ops.subtract(c_log, u_log), g))
        cfg = ops.log_softmax(mixed, -1)
    else:
        cfg = c_log
    start = ops.constant(np.array([0], dtype=np.int64))
    stop = ops.constant(np.array([AUDIO_MASK_ID], dtype=np.int64))
    step = ops.constant(np.array([1], dtype=np.int64))
    axes = ops.constant(np.array([-1], dtype=np.int64))
    cfg_ok = ops.slice(cfg, start, stop, step, axes)
    k = ops.constant(np.int32(1))
    top = ops.topk(cfg_ok, k, axis=-1, mode="max", sort="none")
    scores = ops.squeeze(top.output(0), np.array([-1], dtype=np.int64))
    pred = ops.squeeze(top.output(1), np.array([-1], dtype=np.int64))
    pred_i64 = ops.convert(pred, ov.Type.i64)
    return ov.Model([pred_i64, scores], [c_p, u_p], "tts_cfg")


def _lang_tag(language: str | None) -> str:
    if not language:
        return "None"
    code = language.strip()
    if not code or code.lower() == "none":
        return "None"
    return code.split("-", 1)[0]


class TTSOVBackend(TTSBackend):
    """Compile OmniVoice FP16 IR on OpenVINO GPU. No PyTorch at runtime."""

    def __init__(self, config: AMCConfig) -> None:
        self.config = config
        self.device = resolve_device(config)
        self._core: Any = None
        self._fwd: Any = None
        self._dec: Any = None
        self._cfg: Any = None
        self._fwd_req: Any = None
        self._dec_req: Any = None
        self._cfg_req: Any = None
        self._tokenizer: Any = None
        self._duration = RuleDurationEstimator()

    def is_loaded(self) -> bool:
        return self._fwd is not None and self._dec is not None

    def load(self) -> None:
        if self.is_loaded():
            return
        model_dir = _require_tts_ir(self.config.tts_model_path)
        if self.device.upper() != "GPU" and not self.config.allow_cpu_fallback:
            raise ModelPathError(
                f"TTS hop requires OpenVINO device GPU (got {self.device})."
            )
        import openvino as ov

        tokenizer = JsonBpeTokenizer(model_dir / "tokenizer.json")
        props = compile_cache_kwargs(self.config, self.device, "omnivoice")
        if self.device.upper() == "GPU":
            props = dict(props)
            # One serial unmask stream. THROUGHPUT would add empty streams.
            props["PERFORMANCE_HINT"] = "LATENCY"
            props["INFERENCE_PRECISION_HINT"] = "f16"
        print(
            f"loading OmniVoice TTS on {self.device} from {model_dir} "
            "(first GPU compile can take a minute; later loads use ov_cache/omnivoice)",
            flush=True,
        )
        core = fwd = dec = cfg = None
        fwd_req = dec_req = cfg_req = None
        try:
            core = ov.Core()
            fwd = core.compile_model(
                str(model_dir / "forward.xml"), self.device, props
            )
            dec = core.compile_model(
                str(model_dir / "decode.xml"), self.device, props
            )
            fwd_req = fwd.create_infer_request()
            dec_req = dec.create_infer_request()
            if str(getattr(self.config, "tts_cfg", "numpy")).lower() == "gpu":
                # CFG softmax must stay FP32: GPU default f16 overflows to NaN.
                cfg_props = dict(props)
                cfg_props["INFERENCE_PRECISION_HINT"] = "f32"
                cfg_ir = _build_cfg_model(float(self.config.tts_guidance_scale))
                cfg = core.compile_model(cfg_ir, self.device, cfg_props)
                cfg_req = cfg.create_infer_request()
                compiled_msg = "tts compiled (forward+cfg+decode on GPU)"
            else:
                compiled_msg = "tts compiled (forward+decode on GPU, numpy CFG)"
        except BaseException:
            fwd_req = dec_req = cfg_req = None
            drop_pipeline(fwd)
            drop_pipeline(dec)
            drop_pipeline(cfg)
            drop_pipeline(core)
            raise
        self._tokenizer = tokenizer
        self._core = core
        self._fwd = fwd
        self._dec = dec
        self._cfg = cfg
        self._fwd_req = fwd_req
        self._dec_req = dec_req
        self._cfg_req = cfg_req
        print(compiled_msg, flush=True)

    def unload(self) -> None:
        fwd, dec, cfg, core = self._fwd, self._dec, self._cfg, self._core
        self._fwd_req = None
        self._dec_req = None
        self._cfg_req = None
        self._fwd = None
        self._dec = None
        self._cfg = None
        self._core = None
        self._tokenizer = None
        drop_pipeline(fwd)
        drop_pipeline(dec)
        drop_pipeline(cfg)
        drop_pipeline(core)

    def speak(self, text: str, *, language: str | None = None) -> Speech:
        if not self.is_loaded():
            raise RuntimeError("tts pipeline is not loaded")
        spoken = (text or "").strip()
        if not spoken:
            raise ValueError("tts speak needs non-empty text")
        target = self._estimate_frames(spoken)
        packed = self._pack_inputs(spoken, target, language)
        codes = self._unmask(packed, target)
        wav = self._decode_gpu(codes)
        return Speech(pcm=[float(x) for x in wav.tolist()], sample_rate=SAMPLE_RATE)

    def _estimate_frames(self, text: str) -> int:
        est = self._duration.estimate_duration(text, _REF_TEXT, float(_REF_FRAMES))
        return max(1, min(MAX_TARGET_FRAMES, int(est)))

    def _encode(self, text: str) -> list[int]:
        tok = self._tokenizer
        ids: list[int] = []
        last = 0
        for m in _NONVERBAL_PATTERN.finditer(text):
            if m.start() > last:
                ids.extend(tok.encode(text[last : m.start()]))
            ids.extend(tok.encode(m.group()))
            last = m.end()
        if last < len(text):
            ids.extend(tok.encode(text[last:]))
        if not ids:
            ids = list(tok.encode(text))
        return ids

    def _pack_inputs(
        self, text: str, num_target: int, language: str | None
    ) -> dict[str, np.ndarray]:
        lang = _lang_tag(language)
        style = (
            f"<|lang_start|>{lang}<|lang_end|>"
            f"<|instruct_start|>None<|instruct_end|>"
        )
        style_ids = self._encode(style)
        wrapped = f"<|text_start|>{_combine_text(text)}<|text_end|>"
        text_ids = self._encode(wrapped)
        prefix = style_ids + text_ids
        seq = len(prefix) + num_target
        input_ids = np.full((1, CODEBOOKS, seq), AUDIO_MASK_ID, dtype=np.int64)
        for layer in range(CODEBOOKS):
            input_ids[0, layer, : len(prefix)] = np.asarray(prefix, dtype=np.int64)
        audio_mask = np.zeros((1, seq), dtype=bool)
        audio_mask[0, len(prefix) :] = True
        return {
            "input_ids": input_ids,
            "audio_mask": audio_mask,
            "prefix_len": np.array(len(prefix), dtype=np.int64),
        }

    def _gpu_forward(
        self,
        input_ids: np.ndarray,
        audio_mask: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        req = self._fwd_req
        req.infer([input_ids, audio_mask, attention_mask])
        return np.asarray(req.get_output_tensor(0).data, dtype=np.float32)

    def _unmask(self, packed: dict[str, np.ndarray], t_len: int) -> np.ndarray:
        cfg = self.config
        num_step = int(cfg.tts_num_step)
        guidance = float(cfg.tts_guidance_scale)
        t_shift = float(cfg.tts_t_shift)
        layer_pen = float(cfg.tts_layer_penalty)
        pos_temp = float(cfg.tts_position_temperature)
        rng = np.random.default_rng(0)

        cond_ids = packed["input_ids"]
        cond_mask = packed["audio_mask"]
        c_len = int(cond_ids.shape[2])
        u_len = t_len
        max_len = c_len

        batch_ids = np.full((2, CODEBOOKS, max_len), AUDIO_MASK_ID, dtype=np.int64)
        batch_audio = np.zeros((2, max_len), dtype=bool)
        batch_attn = np.zeros((2, 1, max_len, max_len), dtype=bool)

        batch_ids[0, :, :c_len] = cond_ids[0]
        batch_audio[0, :c_len] = cond_mask[0]
        batch_attn[0, :, :c_len, :c_len] = True

        batch_ids[1, :, :u_len] = cond_ids[0, :, c_len - u_len : c_len]
        batch_audio[1, :u_len] = cond_mask[0, c_len - u_len : c_len]
        batch_attn[1, :, :u_len, :u_len] = True
        if max_len > u_len:
            diag = np.arange(u_len, max_len)
            batch_attn[1, 0, diag, diag] = True

        tokens = np.full((CODEBOOKS, t_len), AUDIO_MASK_ID, dtype=np.int64)
        sched = _unmask_schedule(t_len, num_step, t_shift)
        layer_ids = np.arange(CODEBOOKS, dtype=np.float32).reshape(CODEBOOKS, 1)

        for step in range(num_step):
            k = sched[step]
            logits = self._gpu_forward(batch_ids, batch_audio, batch_attn)
            if k <= 0:
                continue
            c_logits = np.ascontiguousarray(
                logits[0:1, :, c_len - t_len : c_len, :], dtype=np.float32
            )
            u_logits = np.ascontiguousarray(
                logits[1:2, :, :t_len, :], dtype=np.float32
            )
            pred, scores = self._predict_gpu(c_logits, u_logits, guidance)
            scores = scores[0] - (layer_ids * layer_pen)
            if pos_temp > 0.0:
                scores = _gumbel(scores, pos_temp, rng)
            still = tokens == AUDIO_MASK_ID
            scores = np.where(still, scores, -np.inf)
            flat = scores.reshape(-1)
            k = min(k, int(np.isfinite(flat).sum()))
            if k <= 0:
                continue
            topk = np.argpartition(flat, -k)[-k:]
            pred_flat = pred[0].reshape(-1)
            tok_flat = tokens.reshape(-1)
            tok_flat[topk] = pred_flat[topk]
            tokens = tok_flat.reshape(CODEBOOKS, t_len)
            batch_ids[0, :, c_len - t_len : c_len] = tokens
            batch_ids[1, :, :t_len] = tokens
        return tokens

    def _predict_gpu(
        self, c_logits: np.ndarray, u_logits: np.ndarray, guidance: float
    ) -> tuple[np.ndarray, np.ndarray]:
        req = self._cfg_req
        if req is not None:
            req.infer([c_logits, u_logits])
            pred = np.asarray(req.get_output_tensor(0).data, dtype=np.int64).copy()
            scores = np.asarray(req.get_output_tensor(1).data, dtype=np.float32).copy()
            return pred, scores
        return self._predict(c_logits, u_logits, guidance)

    @staticmethod
    def _predict(
        c_logits: np.ndarray, u_logits: np.ndarray, guidance: float
    ) -> tuple[np.ndarray, np.ndarray]:
        c_log = _log_softmax(c_logits)
        if guidance != 0.0:
            u_log = _log_softmax(u_logits)
            log_probs = _log_softmax(c_log + guidance * (c_log - u_log))
        else:
            log_probs = c_log
        log_probs[..., AUDIO_MASK_ID] = -np.inf
        pred = np.argmax(log_probs, axis=-1).astype(np.int64)
        scores = np.max(log_probs, axis=-1)
        return pred, scores

    def _decode_gpu(self, codes: np.ndarray) -> np.ndarray:
        audio_codes = codes.reshape(1, CODEBOOKS, codes.shape[-1]).astype(np.int64)
        req = self._dec_req
        req.infer([audio_codes])
        wav = np.asarray(req.get_output_tensor(0).data, dtype=np.float32)
        return np.asarray(wav, dtype=np.float32).reshape(-1).copy()
