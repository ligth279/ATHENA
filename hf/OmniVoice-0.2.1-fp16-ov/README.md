---
license: cc-by-nc-4.0
base_model: k2-fsa/OmniVoice
base_model_relation: quantized
tags:
  - openvino
  - text-to-speech
  - intel-gpu
  - fp16
  - candidate
  - omnivoice
library_name: openvino
pipeline_tag: text-to-speech
language:
  - multilingual
---

# OmniVoice 0.2.1 — OpenVINO FP16 (candidate)

Unofficial **OpenVINO FP16** export of [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) **0.2.1**.

This is a **candidate** IR: Intel Arc GPU (`OpenVINO` device `"GPU"`), no PyTorch at runtime. Auto-voice only in this folder (no clone graph). Weights stay under the **upstream CC-BY-NC** license.

**Not a fine-tune.** Same 0.2.1 weights, exported to OpenVINO FP16. Hugging Face lists anything with `base_model:` as a finetune unless `base_model_relation: quantized` is set — that is a Hub tag, not training.

Not affiliated with Xiaomi / k2-fsa.

## Files

| File | What it is |
|---|---|
| `forward.xml` / `forward.bin` | Embed + Qwen3-0.6B + 8 audio heads (one unmask step) |
| `decode.xml` / `decode.bin` | Higgs acoustic decoder, codes → 24 kHz wav |
| `tokenizer.json` | Qwen3 BPE (CPU lookup) |
| `talk.py` | Recommended speak script (OpenVINO GPU + **NumPy CFG**) |
| `tts_duration.py` | Duration heuristic from OmniVoice (CPU) |

PyTorch was used **only** to export. Do not `pip install omnivoice` for this IR (that package pulls CUDA torch).

## How to use

Intel GPU with OpenVINO. Tested: **Arc B580 12 GB**, OpenVINO **2026.4.1**.

```bash
pip install openvino numpy tokenizers
hf download light434/OmniVoice-0.2.1-fp16-ov --local-dir OmniVoice-0.2.1-fp16-ov
cd OmniVoice-0.2.1-fp16-ov
python talk.py "A fraction is a part of a whole." --language en --wav out.wav
```

`talk.py` on **`main`** compiles both graphs on `"GPU"`, runs **16** unmask steps with **NumPy CFG** (CPU log-softmax), writes 24 kHz PCM. This is the recommended path.

With [ATHENA AMC](https://github.com/ligth279/ATHENA) (exclusive GPU slot: load → speak → unload):

```bash
export XILO_TTS_PATH=OmniVoice-0.2.1-fp16-ov
python -m athena.amc talk "A fraction is a part of a whole." --language en --wav out.wav
```

Device is OpenVINO `"GPU"` (Intel). Not NVIDIA CUDA.

## Metrics

Hardware: Intel Arc B580 12 GB, i5-10400F, 16 GB DDR4. OpenVINO 2026.4.1, device `"GPU"`. This IR, **auto-voice**, **16** unmask steps.

### Seed-TTS English (official 1088-line list)

ASR: Whisper **large-v3 turbo** INT8 (AMC hop). Scorer: OmniVoice `seedtts` post_process (lowercase, strip punctuation). Corpus WER = (S+D+I) / words.

| System | n | Corpus WER | Notes |
|---|---|---|---|
| **This IR (`main`, NumPy CFG)** | 1088 | **1.69%** | S=167 D=28 I=4 W=11806. Auto-voice, 16 steps, turbo |
| Branch `gpu-cfg-fp32` (experimental) | 1088 | **1.72%** | Same IR, FP32 GPU CFG. Not the default. |
| k2-fsa paper OmniVoice | Seed-TTS en | **1.60%** | Voice clone, 32 steps, Whisper large-v3 |

Mean sentence raw WER **4.04%** (punctuation still on). Mean sentence norm WER **1.73%**. Wall **1272.8 s** (~1.17 s/line including TTS + ASR).

Hot TTS only (8 lines, cached kernels, same `forward`/`decode`): NumPy CFG **0.38 s/line**, RTF **0.094**. That is the production baseline.

Same 1088 texts as the paper. This run does **not** clone a prompt wav.

### Same-line check vs original OmniVoice (CPU torch)

13 lines (10 English classroom + es/hi/de), `num_step=16`, same Whisper turbo:

| Set | This IR | Original OmniVoice 0.2.1 (CPU FP32) |
|---|---|---|
| English WER | 6.5% | 5.5% |
| All 13 WER | 5.0% | 5.5% |

The 1 point English gap was one word (`Two` → Whisper `2.`). Hindi was exact on this IR.

### Classroom 10 (not Seed-TTS)

Mean raw WER 6.5% / punct-stripped 5.3%. Gold line *A fraction is a part of a whole.* exclusive hop **0.0%** WER vs Whisper turbo.

### Hop timing (B580, cached compile)

| | Boot | Run | Unload | USM at boot | After unload |
|---|---|---|---|---|---|
| OmniVoice FP16 | 0.69 s | 2.61 s | 0.08 s | 1.18 GiB | 0 |

Xe engines while speaking: **RCS (3D) 0%**, CCS ~13%, BCS ~6%. VRAM ~1.56 GiB of 12 GiB.

## Experimental GPU CFG

**Not the recommended path.** Same FP16 IR as `main`. Implementation lives on branch [`gpu-cfg-fp32`](https://huggingface.co/light434/OmniVoice-0.2.1-fp16-ov/tree/gpu-cfg-fp32).

```bash
hf download light434/OmniVoice-0.2.1-fp16-ov --revision gpu-cfg-fp32 --local-dir OmniVoice-gpu-cfg
```

Experimental GPU CFG: FP32 GPU CFG was evaluated on the official 1,088-line Seed-TTS English corpus. It produced 1.72% corpus WER versus 1.69% for the NumPy CFG reference. Controlled hot-TTS testing showed 0.98 s/line versus 0.38 s/line for NumPy CFG, so the GPU-CFG path is retained as an experimental baseline rather than the default implementation.

Cause: each unmask step copies logits GPU→CPU, then CPU→GPU for a tiny softmax graph, then GPU→CPU again. FP16 softmax on this graph overflows (NaN → token 0). CFG must compile with `INFERENCE_PRECISION_HINT=f32`; `forward`/`decode` stay FP16.

| | NumPy CFG (`main`) | GPU CFG FP32 (`gpu-cfg-fp32`) |
|---|---:|---:|
| Seed-TTS corpus WER | **1.69%** | 1.72% |
| Hot TTS wall | **0.38 s/line** | 0.98 s/line |
| Hot RTF | **0.094** | 0.245 |
| Codebook tok/s | **2126** | 817 |
| Neural precision | FP16 | FP16 + FP32 CFG |

A fused on-device CFG (no host logit round-trip) is the next experiment. It belongs on a later `gpu-cfg-fused` revision, and only replaces `main` if it beats **0.38 s/line** at ~1.69% WER.

## Credit

- **Model:** [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) 0.2.1 — Xiaomi / Next-gen Kaldi
- **Paper:** [OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models](https://huggingface.co/papers/2604.00688) (Zhu et al., 2026)
- **Code:** [github.com/k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)
- **This repo:** FP16 OpenVINO export + Intel GPU measurements only. No extra training.

```bibtex
@article{zhu2026omnivoice,
  title={OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models},
  author={Zhu, Han and Ye, Lingxuan and Kang, Wei and Yao, Zengwei and Guo, Liyong
          and Kuang, Fangjun and Han, Zhifeng and Zhuang, Weiji and Lin, Long and Povey, Daniel},
  journal={arXiv preprint arXiv:2604.00688},
  year={2026}
}
```

## License

Upstream **code** is Apache-2.0. Upstream **pretrained weights** (and therefore this IR) are **CC-BY-NC** because of training data (e.g. Emilia), as stated on [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice).

Non-commercial use only, with credit to k2-fsa / the paper authors.

Do not use this model for unauthorized voice cloning, impersonation, fraud, or other illegal activity.

## Disclaimer

Candidate export. Not an official k2-fsa release. Voice clone / 32-step paper protocol is not in these two graphs.
