#!/usr/bin/env python3
"""Experimental GPU CFG speak script. Not the recommended path.

Same FP16 IR as main. CFG log-softmax runs on GPU in FP32.
Hot TTS: 0.98 s/line vs 0.38 s/line NumPy CFG on main.
Seed-TTS: 1.72% vs 1.69%. Use main talk.py unless you are reproducing this hop.

  python talk.py "A fraction is a part of a whole." --language en --wav out.wav
"""

from __future__ import annotations

import argparse
import math
import re
import wave
from pathlib import Path

import numpy as np

CODEBOOKS = 8
AUDIO_MASK_ID = 1024
AUDIO_VOCAB = 1025
SAMPLE_RATE = 24000
HOP_LENGTH = 960
MAX_TARGET_FRAMES = 375
_REF_TEXT = "Nice to meet you."
_REF_FRAMES = 25
NUM_STEP = 16
GUIDANCE = 2.0
T_SHIFT = 0.1
LAYER_PENALTY = 5.0
POS_TEMP = 5.0

_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)

ROOT = Path(__file__).resolve().parent


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def _unmask_schedule(target_len: int, num_step: int, t_shift: float) -> list[int]:
    ts = np.linspace(0.0, 1.0, num_step + 1, dtype=np.float64)
    ts = t_shift * ts / (1.0 + (t_shift - 1.0) * ts)
    total = target_len * CODEBOOKS
    rem = total
    sched: list[int] = []
    for step in range(num_step):
        if step == num_step - 1:
            num = rem
        else:
            num = min(math.ceil(total * (ts[step + 1] - ts[step])), rem)
        sched.append(int(num))
        rem -= int(num)
    return sched


def _combine_text(text: str) -> str:
    full = re.sub(r"[\r\n]+", "", text.strip())
    full = full.replace("\uff08", "(").replace("\uff09", ")")
    full = re.sub(r"[ \t]+", " ", full)
    chinese = r"[\u4e00-\u9fff]"
    return re.sub(rf"(?<={chinese})\s+|\s+(?={chinese})", "", full)


def _gumbel(scores: np.ndarray, temperature: float, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(scores.shape, dtype=np.float32).astype(np.float64)
    return scores / temperature + (-np.log(-np.log(u + 1e-10) + 1e-10))


def _lang_tag(language: str | None) -> str:
    if not language or language.strip().lower() == "none":
        return "None"
    return language.strip().split("-", 1)[0]


def _encode(tok, text: str) -> list[int]:
    ids: list[int] = []
    last = 0
    for m in _NONVERBAL_PATTERN.finditer(text):
        if m.start() > last:
            ids.extend(tok.encode(text[last : m.start()], add_special_tokens=False).ids)
        ids.extend(tok.encode(m.group(), add_special_tokens=False).ids)
        last = m.end()
    if last < len(text):
        ids.extend(tok.encode(text[last:], add_special_tokens=False).ids)
    return ids or list(tok.encode(text, add_special_tokens=False).ids)


def _build_cfg_model(guidance: float):
    """FP32 GPU graph: CFG log-softmax + argmax. Do not compile this f16."""
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
    u_log = ops.log_softmax(u, -1)
    g = ops.constant(np.array(guidance, dtype=np.float32))
    mixed = ops.add(c_log, ops.multiply(ops.subtract(c_log, u_log), g))
    cfg = ops.log_softmax(mixed, -1)
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


def _predict(c_logits: np.ndarray, u_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c_log = _log_softmax(c_logits)
    u_log = _log_softmax(u_logits)
    log_probs = _log_softmax(c_log + GUIDANCE * (c_log - u_log))
    log_probs[..., AUDIO_MASK_ID] = -np.inf
    pred = np.argmax(log_probs, axis=-1).astype(np.int64)
    scores = np.max(log_probs, axis=-1)
    return pred, scores


def _write_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    samples = np.clip(pcm.astype(np.float32), -1.0, 1.0)
    pcm16 = (samples * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def speak(text: str, language: str = "en", device: str = "GPU") -> np.ndarray:
    import openvino as ov
    from tokenizers import Tokenizer

    from tts_duration import RuleDurationEstimator

    for name in ("forward.xml", "forward.bin", "decode.xml", "decode.bin", "tokenizer.json"):
        if not (ROOT / name).is_file():
            raise SystemExit(f"missing {ROOT / name}")

    tok = Tokenizer.from_file(str(ROOT / "tokenizer.json"))
    core = ov.Core()
    if device == "GPU" and "GPU" not in list(core.available_devices):
        raise SystemExit(f"OpenVINO has no GPU (devices={core.available_devices})")
    props = {}
    cache = ROOT / "ov_cache"
    cache.mkdir(exist_ok=True)
    if device == "GPU":
        props = {
            "CACHE_DIR": str(cache),
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "f16",
        }
    print(f"compiling experimental GPU CFG on {device} (first time can take a minute)", flush=True)
    fwd = core.compile_model(str(ROOT / "forward.xml"), device, props)
    dec = core.compile_model(str(ROOT / "decode.xml"), device, props)
    cfg_props = dict(props)
    cfg_props["INFERENCE_PRECISION_HINT"] = "f32"
    cfg = core.compile_model(_build_cfg_model(GUIDANCE), device, cfg_props)
    cfg_req = cfg.create_infer_request()
    print("compiled forward+decode f16, cfg f32", flush=True)

    est = RuleDurationEstimator().estimate_duration(text, _REF_TEXT, float(_REF_FRAMES))
    t_len = max(1, min(MAX_TARGET_FRAMES, int(est)))
    lang = _lang_tag(language)
    style = f"<|lang_start|>{lang}<|lang_end|><|instruct_start|>None<|instruct_end|>"
    prefix = _encode(tok, style) + _encode(tok, f"<|text_start|>{_combine_text(text)}<|text_end|>")
    seq = len(prefix) + t_len
    input_ids = np.full((1, CODEBOOKS, seq), AUDIO_MASK_ID, dtype=np.int64)
    for layer in range(CODEBOOKS):
        input_ids[0, layer, : len(prefix)] = np.asarray(prefix, dtype=np.int64)
    audio_mask = np.zeros((1, seq), dtype=bool)
    audio_mask[0, len(prefix) :] = True

    c_len = seq
    u_len = t_len
    batch_ids = np.full((2, CODEBOOKS, c_len), AUDIO_MASK_ID, dtype=np.int64)
    batch_audio = np.zeros((2, c_len), dtype=bool)
    batch_attn = np.zeros((2, 1, c_len, c_len), dtype=bool)
    batch_ids[0, :, :c_len] = input_ids[0]
    batch_audio[0, :c_len] = audio_mask[0]
    batch_attn[0, :, :c_len, :c_len] = True
    batch_ids[1, :, :u_len] = input_ids[0, :, c_len - u_len : c_len]
    batch_audio[1, :u_len] = audio_mask[0, c_len - u_len : c_len]
    batch_attn[1, :, :u_len, :u_len] = True
    if c_len > u_len:
        diag = np.arange(u_len, c_len)
        batch_attn[1, 0, diag, diag] = True

    tokens = np.full((CODEBOOKS, t_len), AUDIO_MASK_ID, dtype=np.int64)
    sched = _unmask_schedule(t_len, NUM_STEP, T_SHIFT)
    layer_ids = np.arange(CODEBOOKS, dtype=np.float32).reshape(CODEBOOKS, 1)
    rng = np.random.default_rng(0)
    fwd_req = fwd.create_infer_request()
    for step in range(NUM_STEP):
        k = sched[step]
        fwd_req.infer([batch_ids, batch_audio, batch_attn])
        logits = np.asarray(fwd_req.get_output_tensor(0).data, dtype=np.float32)
        if k <= 0:
            continue
        c_logits = np.ascontiguousarray(
            logits[0:1, :, c_len - t_len : c_len, :], dtype=np.float32
        )
        u_logits = np.ascontiguousarray(
            logits[1:2, :, :t_len, :], dtype=np.float32
        )
        cfg_req.infer([c_logits, u_logits])
        pred = np.asarray(cfg_req.get_output_tensor(0).data, dtype=np.int64).copy()
        scores = np.asarray(cfg_req.get_output_tensor(1).data, dtype=np.float32).copy()
        scores = scores[0] - (layer_ids * LAYER_PENALTY)
        if POS_TEMP > 0.0:
            scores = _gumbel(scores, POS_TEMP, rng)
        still = tokens == AUDIO_MASK_ID
        scores = np.where(still, scores, -np.inf)
        flat = scores.reshape(-1)
        k = min(k, int(np.isfinite(flat).sum()))
        if k <= 0:
            continue
        topk = np.argpartition(flat, -k)[-k:]
        tok_flat = tokens.reshape(-1)
        tok_flat[topk] = pred[0].reshape(-1)[topk]
        tokens = tok_flat.reshape(CODEBOOKS, t_len)
        batch_ids[0, :, c_len - t_len : c_len] = tokens
        batch_ids[1, :, :t_len] = tokens

    codes = tokens.reshape(1, CODEBOOKS, t_len).astype(np.int64)
    dec_req = dec.create_infer_request()
    dec_req.infer([codes])
    wav = np.asarray(dec_req.get_output_tensor(0).data, dtype=np.float32).reshape(-1)
    return wav.copy()


def main() -> int:
    p = argparse.ArgumentParser(description="OmniVoice OpenVINO FP16 talk")
    p.add_argument("text")
    p.add_argument("--language", default="en")
    p.add_argument("--wav", default="out.wav")
    p.add_argument("--device", default="GPU", help="OpenVINO device (GPU or CPU)")
    args = p.parse_args()
    wav = speak(args.text, language=args.language, device=args.device)
    _write_wav(Path(args.wav), wav, SAMPLE_RATE)
    print(f"wrote {args.wav}  {len(wav)} samples @ {SAMPLE_RATE} Hz  ({len(wav)/SAMPLE_RATE:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
