# AMC — AI Model Controller (xilo v6)

Maintainer map for `athena/amc/`. Product rules: **section 2** of `docu.txt`. This file is how the code actually works.

**New-chat handoff (IR sizes, INT8 vs embeddings, compile, next commands):** `docs/amc-context.md`.

**Keep the B580 clean:** Generate `-5` on TranslateGemma was **SigLIP** (even INT8 ~0.4 GB) in the VLM folder, not token count. Runtime vision is a **1.3 MB stub**; real SigLIP stays in `models/translategemma-4b-it-int8-ov-vision-bak`. Do not copy it back. After a `-5`, Xe CCS resets and the *next* compile is `-14`/`-58` until a **new Python** (reboot only if CCS stays dead). Gemma **does** leave: `GPU_MEMORY_STATISTICS` `usm_device` 4.25 GiB → **0** after `translate()` unload. See `docs/amc-context.md`.

Hardware: **Intel Arc B580 12 GB** + **i5-10400F** (no iGPU) + **16 GB DDR4**. OpenVINO device `"GPU"` is the B580.

Runtime: OpenVINO **2026.4.1** + `openvino-genai`. Not PyTorch. Not IPEX. Not bitsandbytes. Not NVIDIA.

---

## What it does

One GPU resident at a time: **Llama INT4** or **Llama INT8** or **Whisper** or **TranslateGemma** or **OmniVoice TTS**. A single worker queue waits out the current job, then unloads and switches.

| Role | Event | IR | KV | Why |
|---|---|---|---|---|
| TUTOR | T | INT4 ~5.1 GB | **4 GB** | Chat needs a window. 5+4 fits 12 GB. |
| EVALUATOR | E | INT8 ~7.5 GB | **2 GB** | Quiz turns are short. 7.5+4 would OOM. |
| Speak (STT) | — | Whisper turbo INT8 | n/a | Exclusive hop |
| Translator | G hop | TranslateGemma 4B language **INT8** ~3.62 GB + text emb **INT8** ~0.63 GB + **vision stub** ~1.3 MB. | n/a | Stateless. 2K, ≤75%, sentence chunks. `VLMPipeline` still opens a vision xml; SigLIP must not be that file. |
| Talk (TTS) | G hop | OmniVoice 0.2.1 FP16 (`forward.xml` + `decode.xml`) | n/a | Stateless. OpenVINO `"GPU"` embed + unmask + decode. Unloads after the wav. Not Whisper. |

Never two of these in VRAM. Compile caches: `models/ov_cache/int4`, `int8`, `translate`, `omnivoice`.

Talk/TTS is **not** Whisper. `amc.talk()` speaks a string (or the last tutor reply) on OmniVoice FP16, then unloads.

---

## Unique execution path

Do not add a second load path.

```
caller → AMC.public_method() → queue → amc-worker
                                       ├─ _execute(job)
                                       ├─ _ensure(model)   # unload other, load this
                                       └─ INT4 / INT8 / Whisper / Translate / TTS backend
```

Public methods only enqueue. Only `_ensure` / `_unload_resident` touch VRAM.

---

## File map

| File | Job |
|---|---|
| `controller.py` | Queue, exclusive slot, Event T/E + G hops |
| `backends.py` | OpenVINO GenAI INT4, INT8, Whisper, TranslateGemma + OmniVoice TTS + mocks |
| `tts.py` | OmniVoice FP16: GPU `forward` + GPU `decode`, numpy unmask loop |
| `translate.py` | Official TranslateGemma prompt + 75% sentence chunker |
| `roles.py` | Tutor / evaluator **prompts** |
| `memory.py` | RAM conversation text (not GPU KV) |
| `config.py` | Paths, KV budgets, token/temp knobs |
| `gpu.py` | Device `"GPU"`, per-model `CACHE_DIR` |
| `cli.py` | `python -m athena.amc status\|tutor\|eval\|translate\|talk` |
| `types.py` | `ModelId` includes `TRANSLATE` and `TTS` |
| `event_g.py` (Athena) | Event G hop planner + combinator |
| `noise.py` | Word-slip / WER helpers for hop stacking |
| `tests/test_amc.py` | Mock tests (no GPU) |
| `tests/test_event_g.py` | Combinator + noise (no GPU) |

---

## Parameters (knobs)

| Knob | Default | Used for |
|---|---|---|
| `kv_cache_gb` | 4 | INT4 tutor KV (GB) |
| `kv_cache_gb_int8` | 2 | INT8 evaluator KV (GB) |
| `kv_cache_gb_translate` | 1 | Present on `AMCConfig`. **Not passed** into `VLMPipeline` — `scheduler_config` switches GenAI to the continuous-batching adapter, which has no chat mode and MatMul-crashes text-only generate. |
| `max_new_tokens_tutor` | 512 | tutor generation cap |
| `max_new_tokens_eval` | 96 | hint cap |
| `tutor_temperature` | 0.4 | tutor sampling |
| `eval_temperature` | 0.1 | hints, greedy-ish |
| `memory_keep_pairs` | 3 | last N Q/A dumped to DC |
| `job_timeout_s` | 180 | wait for a job (CLI uses 900 for first compile) |
| `device` | `GPU` | no silent CPU fallback |
| `whisper_language` | `<\|en\|>` | STT language tag |
| `translate_max_input_tokens` | 2048 | TranslateGemma card input window |
| `translate_fill_ratio` | 0.75 | Never feed more than this fraction |
| `max_new_tokens_translate` | 512 | per-chunk generate cap |
| `tts_num_step` | 16 | OmniVoice unmask steps (paper default 32) |

Env: `XILO_DEVICE`, `XILO_AMC_BACKEND`, `XILO_LLAMA_PATH`, `XILO_LLAMA_INT8_PATH`, `XILO_WHISPER_PATH`, `XILO_TRANSLATE_PATH`, `XILO_OV_CACHE`.

---

## Public API

```python
from athena.amc import AMC, AMCConfig

amc = AMC.mock()                            # tests
# amc = AMC(AMCConfig())                    # real GPU

amc.enter_event_t(section_json)             # load INT4, TUTOR
reply = amc.tutor_ask("why is this?")

amc.transcribe(pcm_16k)                     # unload Llama, Whisper STT
amc.translate(text, source_lang="hi", target_lang="en")  # G hop; unloads after

amc.enter_event_e(behavioral_level=2)       # unload INT4, load INT8
hint = amc.evaluator_hint(q, student, allow_approximate=False)

amc.leave_role()
amc.shutdown()
```

Constructor: `llama=` (INT4), `llama_int8=` (INT8), `whisper=`, `translator=`, `data_controller=` (`save_last_chats`).

---

## Switch table

| Current | Request | What happens |
|---|---|---|
| idle | Event T | load INT4 (4 GB KV), start tutor chat |
| INT4 tutor | `tutor_ask` | reuse KV + RAM history |
| any busy | anything | **wait** |
| INT4 tutor | `transcribe` | unload INT4, load Whisper; **Llama RAM history kept**; Whisper holds nothing |
| Whisper done | next hop | **unload Whisper immediately** (no STT KV / no STT memory). Text is passed on. |
| any | `translate` | unload resident, load TranslateGemma, chunk at 75% of 2K on **full sentences**, generate each chunk, **unload** (holds nothing) |
| any | `talk` | unload resident, load OmniVoice FP16, speak, **unload** (holds nothing). 24 kHz PCM. |
| Event T | Event E | clear tutor memory, unload INT4, load INT8 (2 GB KV) |
| Event E | `evaluator_hint` | one-shot `start_chat` / generate / `finish_chat` |
| any | `shutdown` | unload resident, stop worker |

Two memories, and they are **only Llama’s**:

- GPU KV — dies on Llama unload / `finish_chat`
- `ConversationMemory` — Llama tutor RAM. Survives a **same-section** STT hop. Cleared on quiz / role end / successful quiz (no doubt carryover)

**STT, TTS, translator are stateless hops.** After the string (or audio) is handed to the next stage, unload that model. The next STT / TTS / translation does **not** use the last one’s KV or chat. Event G “multiple models” means a **queue of hops** on the exclusive slot, not coresident VRAM.

---

## Models on disk

| Path | Status |
|---|---|
| `models/llama-3.1-8b-instruct-int4-ov` | **Ready.** NNCF mixed INT4 (CelesteImperia). Data-free AWQ. GPU-tested. |
| `models/llama-3.1-8b-instruct-int8-ov` | **Ready.** `openvino_model.bin` = 8035958805 bytes. GPU-tested. |
| `models/whisper-large-v3-turbo-int8-ov` | **Ready.** Encoder 645332592, decoder 172534710. GPU-tested (JFK). |
| `models/translategemma-4b-it-int8-ov` | Language INT8 ~3.62 GB, text emb INT8 ~0.63 GB, **vision stub ~1.3 MB** (SigLIP 407 MB in `*-vision-bak`). Culprit of generate `-5` was SigLIP, not sentence length. Do not restore SigLIP into the runtime folder. |
| `models/omnivoice-fp16-ov` | **Ready.** OmniVoice 0.2.1 FP16 IR: `forward.xml` + `decode.xml`. GPU embed/decode. PyTorch was export-only. |
| `models/ov_cache/int4` | ~5.2 GB blob. Cached INT4 load ~4–8 s |
| `models/ov_cache/int8` | **7.5 GB blob.** If this file is missing/truncated (~6 MB leftover), INT8 “first compile” is minutes and can look hung. Cached INT8 ~4.4 s |
| `models/ov_cache/translate` | VLM kernels. Safe to delete after a `-5`; next load rebuilds |

Official Intel zoo does **not** host Llama 3.1 8B Instruct IR (Meta license). Official data-aware INT4 is `optimum-cli export … --awq --scale-estimation --dataset wikitext2` (needs HF token + torch/nncf; 16 GB RAM is tight for that conversion). INT8 NNCF is data-free by design.

Measured on this B580 (cached, exclusive, one process):

| Step | Time |
|---|---|
| Whisper turbo transcribe | **1.0–2.3 s** |
| TranslateGemma hop (cached compile + generate) | **~8–16 s** |
| INT4 tutor_ask | **~8 s** |
| INT8 hint | **~8 s** (0.7 s was an earlier shorter hint) |
| INT4 → INT8 switch | **5.5 s** INT8 after INT4 |
| Event G `translate_in, llama, translate_out` | **~25–41 s** |
| All four models exclusive then sequential + WER | **174.7 s** |

`python scripts/event_g_live.py` uses **512** tutor + translate caps (not 96). 96 cut a 5-sentence Hindi paragraph at `पहले एक`.

---

## Live WER (channel hops only)

Do **not** WER Llama’s answer against the question. That number only means the tutor answered.

| Hop | Gold | WER | What it actually is |
|---|---|---|---|
| Whisper (espeak 16 kHz) | `A fraction is a part of a whole.` | **12.5%** | 1/8: *whole → bowl* (synthetic TTS, not a dead model) |
| Gemma hi→en of `HI_Q` | `What is a fraction? Tell me in one short sentence.` | **60%** | `भिन्न` → “difference” (general Hindi) vs fraction (math class) |
| Gemma paragraph en→hi→en | original EN paragraph | **33%** | paraphrase; last sentence (“common denominator”) survived at cap 512 |
| Gemma sentence round-trip | `EN_Q` | **125%** | compounding: bad `भिन्न` sense fed back |
| Event G `translate_in` | same `HI_Q` gold | **60%** | same `भिन्न` miss |

### How to improve WER (ranked)

1. **Classroom glossary on the translator hop (biggest real miss).** `भिन्न` is “fraction” in this section and “difference” in general Hindi. The official TranslateGemma user-turn in `translate.py` has **no section terms**. The hop stays stateless if Athena passes a one-shot glossary *with the text* (do not keep translator KV). Do not silently rewrite the official prompt in AMC without a live A/B. UI can also say **भिन्न संख्या**.
2. **Stop scoring the wrong thing.** Round-trip WER and Llama-vs-question WER will stay ugly even when the student heard a good answer. Score `translate_in` vs a gold English of *that* Hindi, and Whisper vs a real 16 kHz recording.
3. **Whisper 12.5%.** Use the website’s 16 kHz PCM, not espeak. AMC does not resample. `whisper_language` must stay `<|en|>` (plain `en` raises `lang_to_id`).
4. **Cap 512, chunk at 75% of 2K.** Already in config. Live script must not override to 96/80.
5. **Do not restore SigLIP** to chase quality. It `-5`s generate and then every later hop is `-14`.
6. **TranslateGemma 12B INT8** fills the 12 GB card — skip. 4B + stub is the hop that fits.
7. **RBA closeness** is the product grade for answers, not WER. Wait for RBA.

---

Live switch (`python scripts/switch_models.py`): never two backends loaded.

---

## CLI

```bash
cd /home/light/Documents/projectxi1
source .venv/bin/activate

python -m athena.amc status
python -m athena.amc tutor "What is a fraction, in one short sentence?"
python -m athena.amc tutor                  # REPL, type quit
python -m athena.amc eval --question "What is 2+2?" --answer "5"
python -m athena.amc translate --source en --target hi "A fraction is a part of a whole."
python -m athena.amc talk "A fraction is a part of a whole." --language en
python scripts/download_translategemma.py --compress-only   # INT8 emb + vision stub; no Kaggle re-fetch
python scripts/event_g_live.py             # Llama T + TranslateGemma; 512-token caps
python scripts/tts_live.py                 # OmniVoice TTS hop + optional Whisper WER
python scripts/switch_models.py             # INT4 ↔ INT8 exclusive slot
python -m unittest tests.test_amc tests.test_event_g -q
```

---

## Backends

`build_backends` returns `(int4, int8, whisper, translator, tts)`. Mock mode adds `MockTranslatorBackend` (identity) and `MockTTSBackend`. OpenVINO imported only inside `load()`. PyTorch is not imported at runtime.

Translator: raw TranslateGemma user-turn string (not GenAI's Gemma 3 chat wrapper). Chunker in `translate.py` applies to **both** student→English and Llama-answer→selected-language. Cut only after `. ? ! … । ॥` (and CJK/Arabic enders). `Mr.` / `3.14` are not cuts. A single sentence over budget is last-resort split on `; ,` then words.

Llama load: `LLMPipeline(path, "GPU", CACHE_DIR=ov_cache/<int4|int8>, scheduler_config.cache_size=kv_gb)`.

Whisper: 16 kHz float PCM. No resample in AMC.

---

## How to change it

- Llama prompts → `roles.py` only. Translator user-turn → `translate.py` (official TranslateGemma shape).
- New exclusive model → `ModelId` + backend + `JobKind` in `_execute` → `_ensure`. Never a second thread on generate. Translator chunks live in `translate.py`, not in the backend.
- Do not keep INT4 loaded to “save time” while INT8 or Whisper loads.

---

## Left to do (AMC)

Parked on purpose until the named owner exists, or still missing IR.

### Must fetch / wire in AMC

| Item | Status | Notes |
|---|---|---|
| Whisper STT (`OpenVINO/whisper-large-v3-turbo-int8-ov`) | **Ready, exclusive + sequential GPU-tested** | 16 kHz PCM. Language tag `<\|en\|>`. espeak WER 12.5% (*whole→bowl*). |
| Audio resample | Not in AMC | Website must send 16 kHz, or add a helper later. |
| TTS / `talk()` | **Wired.** OmniVoice 0.2.1 FP16 IR on GPU | Event G hop: speak the string → **unload**. Not Whisper. Embed + decode graphs on `"GPU"`. |
| Translator | **Wired.** Language+emb INT8, vision stub. Sentence, paragraph, Event G GPU-tested. | hi→en **60% WER** on `भिन्न` vs “fraction”. Glossary is an Athena/Event G hop payload, not a second AMC model. |
| Event G combinator | **Mimic in Athena** | Exclusive hops work (USM 0 between). `python scripts/event_g_live.py`. TTS hop is `talk()`. |
| Per-language Whisper | One path in config | Swap `whisper_model_path` later; still one resident, still stateless. |

### Wait for RBA / ALE / Athena (do not guess in AMC now)

| Item | Why wait |
|---|---|
| Question types + closeness (math exact / one-word spelling / paragraph fuzzy) | Product path: answer → **RBA** exact → else LLM closeness only if type allows → else wrong → then INT8 **hint**. AMC only has `evaluator_hint()` + `allow_approximate` (prompt text, not a grade). |
| Answer key in the hint prompt | Optional `correct_answer` for grounded hints; never echo the key. After RBA exists. |
| New section vs same section RAM | **Llama only.** Same-section STT: keep Llama chat. Quiz / successful quiz: clear (no doubt carryover). New section id: wait for Athena/RBA. STT/TTS/translator never keep the thread. |
| DC restore last 3 chats | `save_last_chats` on role end exists. No `load_last_chats` / seed on `enter_event_t`. |
| ALE behavioral gear | Integer `behavioral_level` stub only. |

### Optional / later

| Item | Notes |
|---|---|
| Official data-aware INT4 | Intel recipe: `optimum-cli … --awq --scale-estimation --dataset wikitext2`. Needs HF token + torch/nncf. 16 GB host RAM is tight for conversion. Community INT4 is data-free NNCF. |
| Job timeout vs GPU cancel | Default `job_timeout_s=180`; CLI uses 900. `future.result(timeout)` does **not** abort the worker; GPU stays busy. |
| CWD-relative model paths | `AMC(AMCConfig())` needs repo root or `XILO_*_PATH`. CLI already uses absolute paths. |
| Athena REST | Planned in section 1; AMC is in-process today. |

### Done (do not re-open)

Exclusive GPU slot; INT4 tutor (4 GB KV); INT8 evaluator (2 GB KV); queue waits; IR completeness (no `.part` as ready); CLI `status` / `tutor` / `eval` / `translate` / `talk`; `scripts/switch_models.py`; live B580 switch INT4 ↔ INT8; OmniVoice TTS hop unloads; quiz clears **Llama** RAM; same-section STT keeps **Llama** RAM only (Whisper itself keeps nothing).

---

## Tests that must stay green

`python -m unittest tests.test_amc -v`

| Test | Guards |
|---|---|
| `test_never_two_models_loaded` | INT4 / INT8 / Whisper / Translate / TTS exclusive |
| `test_event_e_uses_int8_not_int4` | quiz loads INT8, unloads INT4 |
| `test_second_request_waits_for_first` | queue |
| `test_tutor_after_whisper_reloads_llama_with_history` | RAM history |
| `test_talk_is_not_whisper` | TTS not STT; unloads after speak |
