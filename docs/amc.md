# AMC — AI Model Controller (xilo v6)

Maintainer map for `athena/amc/`. Product rules: **section 2** of `2333.txt`. This file is how the code actually works.

Hardware: **Intel Arc B580 12 GB** + **i5-10400F** (no iGPU) + **16 GB DDR4**. OpenVINO device `"GPU"` is the B580.

Runtime: OpenVINO **2026.3** + `openvino-genai`. Not PyTorch. Not IPEX. Not bitsandbytes. Not NVIDIA.

---

## What it does

One GPU resident at a time: **Llama INT4** or **Llama INT8** or **Whisper**. A single worker queue waits out the current job, then unloads and switches.

| Role | Event | IR | KV | Why |
|---|---|---|---|---|
| TUTOR | T | INT4 ~5.1 GB | **4 GB** | Chat needs a window. 5+4 fits 12 GB. |
| EVALUATOR | E | INT8 ~7.5 GB | **2 GB** | Quiz turns are short. 7.5+4 would OOM. |
| Speak (STT) | — | Whisper (IR **not fetched**) | n/a | Exclusive switch |

Never INT4 + INT8 in VRAM. Never Llama + Whisper. Compile caches are split: `models/ov_cache/int4` and `models/ov_cache/int8`.

Talk/TTS is **not** Whisper. `amc.talk()` raises `TTSNotWiredError`.

---

## Unique execution path

Do not add a second load path.

```
caller → AMC.public_method() → queue → amc-worker
                                       ├─ _execute(job)
                                       ├─ _ensure(model)   # unload other, load this
                                       └─ INT4 / INT8 / Whisper backend
```

Public methods only enqueue. Only `_ensure` / `_unload_resident` touch VRAM.

---

## File map

| File | Job |
|---|---|
| `controller.py` | Queue, exclusive slot, Event T/E |
| `backends.py` | OpenVINO GenAI INT4, INT8, Whisper + mocks |
| `roles.py` | Tutor / evaluator **prompts** |
| `memory.py` | RAM conversation text (not GPU KV) |
| `config.py` | Paths, KV budgets, token/temp knobs |
| `gpu.py` | Device `"GPU"`, per-model `CACHE_DIR` |
| `cli.py` | `python -m athena.amc status\|tutor` |
| `types.py` | `Event`, `LlamaRole`, `ModelId` (`LLAMA_INT4`, `LLAMA_INT8`, `WHISPER`) |
| `tests/test_amc.py` | Mock tests (no GPU) |

---

## Parameters (knobs)

| Knob | Default | Used for |
|---|---|---|
| `kv_cache_gb` | 4 | INT4 tutor KV (GB) |
| `kv_cache_gb_int8` | 2 | INT8 evaluator KV (GB) |
| `max_new_tokens_tutor` | 512 | tutor generation cap |
| `max_new_tokens_eval` | 96 | hint cap |
| `tutor_temperature` | 0.4 | tutor sampling |
| `eval_temperature` | 0.1 | hints, greedy-ish |
| `memory_keep_pairs` | 3 | last N Q/A dumped to DC |
| `job_timeout_s` | 180 | wait for a job (CLI uses 900 for first compile) |
| `device` | `GPU` | no silent CPU fallback |
| `whisper_language` | `<\|en\|>` | STT language tag |

Env: `XILO_DEVICE`, `XILO_AMC_BACKEND`, `XILO_LLAMA_PATH`, `XILO_LLAMA_INT8_PATH`, `XILO_WHISPER_PATH`, `XILO_OV_CACHE`.

---

## Public API

```python
from athena.amc import AMC, AMCConfig

amc = AMC.mock()                            # tests
# amc = AMC(AMCConfig())                    # real GPU

amc.enter_event_t(section_json)             # load INT4, TUTOR
reply = amc.tutor_ask("why is this?")

amc.transcribe(pcm_16k)                     # unload Llama, Whisper STT

amc.enter_event_e(behavioral_level=2)       # unload INT4, load INT8
hint = amc.evaluator_hint(q, student, allow_approximate=False)

amc.leave_role()
amc.shutdown()
```

Constructor: `llama=` (INT4), `llama_int8=` (INT8), `whisper=`, `data_controller=` (`save_last_chats`).

---

## Switch table

| Current | Request | What happens |
|---|---|---|
| idle | Event T | load INT4 (4 GB KV), start tutor chat |
| INT4 tutor | `tutor_ask` | reuse KV + RAM history |
| any busy | anything | **wait** |
| INT4 tutor | `transcribe` | unload INT4, load Whisper; **keep RAM history** |
| Whisper | `tutor_ask` | unload Whisper, load INT4, rebuild prompt from history |
| Event T | Event E | clear tutor memory, unload INT4, load INT8 (2 GB KV) |
| Event E | `evaluator_hint` | one-shot `start_chat` / generate / `finish_chat` |
| any | `shutdown` | unload resident, stop worker |

Two memories: GPU KV (dies on unload) vs `ConversationMemory` (survives Whisper, dies on role end). Last 3 pairs go to DC on role end.

---

## Models on disk

| Path | Status |
|---|---|
| `models/llama-3.1-8b-instruct-int4-ov` | **Ready.** NNCF mixed INT4 (CelesteImperia). Data-free AWQ. GPU-tested. |
| `models/llama-3.1-8b-instruct-int8-ov` | **Ready.** `openvino_model.bin` = 8035958805 bytes. GPU-tested. |
| `models/whisper-small-int8-ov` | **Not fetched** |
| `models/ov_cache/int4` | Compile cache from live INT4 |
| `models/ov_cache/int8` | Compile cache from live INT8 |

Official Intel zoo does **not** host Llama 3.1 8B Instruct IR (Meta license). Official data-aware INT4 is `optimum-cli export … --awq --scale-estimation --dataset wikitext2` (needs HF token + torch/nncf; 16 GB RAM is tight for that conversion). INT8 NNCF is data-free by design.

Measured on this B580:

| Step | Time |
|---|---|
| INT4 first compile | **58.6 s** |
| INT4 generate (~64 tokens, first run) | **20.1 s** |
| INT8 first compile | **80.1 s** |
| INT8 hint | **0.7 s** |
| Cached INT4 load | **~4 s** |
| Cached T→E switch (unload INT4, load INT8) | **2.3 s** |
| Cached E→T switch | **3.9 s** |
| Cached tutor_ask | **~1.1–1.3 s** |

Live switch (`python scripts/switch_models.py`): never two backends loaded. Whisper skipped (no IR).

---

## CLI

```bash
cd /home/light/Documents/projectxi1
source .venv/bin/activate

python -m athena.amc status
python -m athena.amc tutor "What is a fraction, in one short sentence?"
python -m athena.amc tutor                  # REPL, type quit
python -m athena.amc eval --question "What is 2+2?" --answer "5"
python scripts/switch_models.py             # INT4 ↔ INT8 exclusive slot
python -m unittest tests.test_amc -v
```

---

## Backends

`build_backends` returns `(int4, int8, whisper)`. Mock mode: two `MockLlamaBackend`s + Whisper mock. OpenVINO imported only inside `load()`.

Llama load: `LLMPipeline(path, "GPU", CACHE_DIR=ov_cache/<int4|int8>, scheduler_config.cache_size=kv_gb)`.

Whisper: 16 kHz float PCM. No resample in AMC.

---

## How to change it

- Prompts → `roles.py` only.
- New exclusive model → `ModelId` + backend + `JobKind` in `_execute` → `_ensure`. Never a second thread on generate.
- Do not keep INT4 loaded to “save time” while INT8 or Whisper loads.

---

## Left to do (AMC)

Parked on purpose until the named owner exists, or still missing IR.

### Must fetch / wire in AMC

| Item | Status | Notes |
|---|---|---|
| Whisper STT (`OpenVINO/whisper-small-int8-ov`) | Not fetched | Speak button. Same exclusive slot. 16 kHz PCM. |
| Audio resample | Not in AMC | Website must send 16 kHz, or add a helper later. |
| TTS / `talk()` | Raises `TTSNotWiredError` | **Not Whisper.** Needs OpenVINO `Text2SpeechPipeline` as a third exclusive model. |
| Per-language Whisper | One path in config | Swap `whisper_model_path` later; still one resident. |

### Wait for RBA / ALE / Athena (do not guess in AMC now)

| Item | Why wait |
|---|---|
| Question types + closeness (math exact / one-word spelling / paragraph fuzzy) | Product path: answer → **RBA** exact → else LLM closeness only if type allows → else wrong → then INT8 **hint**. AMC only has `evaluator_hint()` + `allow_approximate` (prompt text, not a grade). |
| Answer key in the hint prompt | Optional `correct_answer` for grounded hints; never echo the key. After RBA exists. |
| New section vs same section RAM | Speak/same Event T: **keep** chat. Quiz T→E: **already clears**. New section id: Athena/RBA must say “same vs new” or AMC will wipe revise-this-section. |
| DC restore last 3 chats | `save_last_chats` on role end exists. No `load_last_chats` / seed on `enter_event_t`. |
| ALE behavioral gear | Integer `behavioral_level` stub only. |
| Event G | Forgotten in `2333.txt`. |

### Optional / later

| Item | Notes |
|---|---|
| Official data-aware INT4 | Intel recipe: `optimum-cli … --awq --scale-estimation --dataset wikitext2`. Needs HF token + torch/nncf. 16 GB host RAM is tight for conversion. Community INT4 is data-free NNCF. |
| Job timeout vs GPU cancel | Default `job_timeout_s=180`; CLI uses 900. `future.result(timeout)` does **not** abort the worker; GPU stays busy. |
| CWD-relative model paths | `AMC(AMCConfig())` needs repo root or `XILO_*_PATH`. CLI already uses absolute paths. |
| Athena REST | Planned in section 1; AMC is in-process today. |

### Done (do not re-open)

Exclusive GPU slot; INT4 tutor (4 GB KV); INT8 evaluator (2 GB KV); queue waits; IR completeness (no `.part` as ready); CLI `status` / `tutor` / `eval`; `scripts/switch_models.py`; live B580 switch INT4 ↔ INT8; talk is not Whisper; quiz clears RAM; speak keeps RAM (Whisper path untested on GPU until IR is fetched).

---

## Tests that must stay green

`python -m unittest tests.test_amc -v`

| Test | Guards |
|---|---|
| `test_never_two_models_loaded` | INT4 / INT8 / Whisper exclusive |
| `test_event_e_uses_int8_not_int4` | quiz loads INT8, unloads INT4 |
| `test_second_request_waits_for_first` | queue |
| `test_tutor_after_whisper_reloads_llama_with_history` | RAM history |
| `test_talk_is_not_whisper` | TTS not STT |
