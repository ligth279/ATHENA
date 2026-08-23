# AMC handoff (read this after compression / in a new chat)

Product: `docu.txt` Event G (~177). Maintainer map: `docs/amc.md`. Hardware: Arc **B580 12 GB**, no NVIDIA/CUDA/IPEX/bitsandbytes. OpenVINO 2026.3 + genai.

This file exists because session context was full. After INT8 embedding compress (or if a live compile already ran), do **not** re-download 12.8 GB from Kaggle.

---

## Rule: keep the B580 clean — never enter a dead GPU state

**Yes.** Exclusive slot is not enough if a hop **OOMs**. `-5` is the *first* bug; `-14` is only the aftershock.

### Why `-5 CL_OUT_OF_RESOURCES` happened first

1. `VLMPipeline` compiles **three** graphs even for text: language + text embeddings + **vision**.
2. The hop **used to be** ~**7.4 GB** (3.9 INT8 language + **2.6 FP32 embeddings** + **0.85 BF16 vision**). Optimum skipped embedding INT8 because those graphs are **under 1e9 params**. `--compress-only` used to return early once language was INT8 — that skip is gone.
3. **Now on disk:** language INT8 ~3.62 GB, text embeddings INT8 ~0.63 GB, **vision stub ~1.3 MB**. Real SigLIP (~407 MB INT8) is in `models/translategemma-4b-it-int8-ov-vision-bak`. Do not copy it back.
4. Generate still allocates activations / KV on top. The B580 is **12 GB and the display GPU**.
5. A leftover fat hop that OOM'd left a failed event on the **process-wide** GPU queue.

### Why Llama T then dies with `-14 WAIT_LIST`

`-14` means “a previous GPU command already failed.” Same OpenVINO GPU plugin / OpenCL context. AMC `resident=None` is Python. It does **not** reset OpenCL. So `LLMPipeline` INT4 (~5.1 GB + 4 GB KV) compiles on a poisoned, maybe still-full card.

**Prevent `-5` (do not wait for `-14`):**

- AMC **refuses** GPU TranslateGemma while text embeddings are not INT8, or the **SigLIP** vision bin is still in the runtime folder (>10 MB, even INT8), or weights + display + scratch exceed 12 GB (`require_translator_gpu_safe`).
- After any `-5`/`-14`, AMC sets **gpu dead** and will not load another model in this process (`GpuDeadError`).
- Hop on disk is already compressed. Live Event G: a **new** Python (`python scripts/event_g_live.py`). Do not reuse a shell that printed `-5`/`-14`.

---

| Code | Meaning | What we saw |
|---|---|---|
| `-5 CL_OUT_OF_RESOURCES` | GPU RAM gone during a command | VLM generate / first live abort (exit 134) |
| `-14 CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST` | A **previous** GPU command already failed; wait list is poisoned | Llama T `ProgramBuilder` after a working translator hop |

Once `-5` or `-14` prints: **do not** load the next model in that same Python process. Unload/gc will not unpoison the OpenCL context. **Exit the process** (new `python scripts/event_g_live.py`). Then shrink the hop so OOM cannot happen again.

**Prevent dead state (in order):**

1. **Never two residents.** Queue hops. Translator `finally: _unload_resident()`.
2. **Unload must destroy the C++ pipeline**, not only `self._pipe = None`. Use `drop_pipeline()` in `athena/amc/gpu.py` (del + double `gc`).
3. **Translator must fit with headroom** on a **12 GB display GPU**. The generate `-5` culprit is **SigLIP**, not token count. Llama T is ~5.1 GB USM + **4 GB cl_mem KV**.
4. **Gemma *does* leave.** Measured `GPU_MEMORY_STATISTICS`: VLM load 4.25 GiB `usm_device` → after AMC `translate()` finally-unload **0.000**. Extra Python refs (`pipe = self._pipe` kept across unload) keep the 4.25 GiB; `drop_pipeline` + no extra refs returns it. On a **clean** process: whisper→gemma→llama INT4 **OK**; Event G `translate_in, llama, translate_out` **OK** 25.5s, USM 0 after. The Gemma-then-Llama `-5` was a **poisoned Xe CCS** after SigLIP generate OOM, not leftover VLM.
5. `--compress-only` INT8s embeddings and **installs the vision stub** (backs up SigLIP). Do not restore SigLIP into the runtime folder.
6. After any `-5`/`-14`: **new process**. Then compress if not done. Do not “retry Llama” in the crashed shell.
7. B580 is **card0 with HDMI/DP** — desktop framebuffer always eats a slice. Do not budget 12.0 GB as free.

**Llama “reconcile MT against the section” is not wired.** Event G still sends the raw `translate_in` English to `tutor_ask`. Discussed only: it can hide `भिन्न`→difference from the student (Llama already has section JSON) but can **raise** error if it over-corrects a real “difference” question, and it does **not** lower `translate_in` WER. English Event T must not get that line.

**Do not** treat malloc_trim as a GPU reset. It is host RAM. Dead state is OpenCL on the device.

---

## What is already true

### Exclusive GPU (never stack)

One resident: Llama INT4 **or** Llama INT8 **or** Whisper **or** TranslateGemma. Event G is a **queue of hops**, not coresident VRAM.

- **Llama T** = INT4 tutor (`enter_event_t` / `tutor_ask`). Event G uses this.
- **Llama E** = INT8 evaluator. **Do not** use it for Event G.
- Llama T **must** turn question → answer. That change is **not** hop noise. Do not WER the answer against the question.

### Event G combinator (Athena, not AMC)

`athena/event_g.py` — `plan_hops` + `EventG.run()`.

| Combo | Hops |
|---|---|
| en type → en out | llama |
| en type → hi out | llama → translate_out |
| hi type → hi out | translate_in → llama → translate_out |
| hi speech → hi out | stt → translate_in → llama → translate_out |
| + speak | same + tts (`tts_pending`; `talk()` still `TTSNotWiredError`) |

Connect a hop only if that flag needs it. STT/translator unload after pass-on.

### TranslateGemma IR on disk

Dir: `models/translategemma-4b-it-int8-ov`

Gemma 3 only exports as **image-text-to-text** (VLM split). There is no `openvino_model.bin`.

| File | Last known | INT8? |
|---|---|---|
| `openvino_language_model.bin` | **~3.62 GB**, U8 constants | **Yes** |
| `openvino_text_embeddings_model.bin` | **~0.63 GB U8** (was 2.69 GB FP32; Optimum skipped &lt;1e9 params) | **Yes** |
| `openvino_vision_embeddings_model.bin` | **~1.3 MB stub** in the runtime folder. Real SigLIP ~407 MB INT8 in `*-vision-bak`. | Stub, not SigLIP |
| tokenizer / detokenizer | small | U8 tables, fine |

First export passed `ov_config=INT8` into `main_export()`, which **skips** the auto INT8 pass (CLI is supposed to call `_main_quantize` after). Script printed `done int8` because files existed. Language was compressed later. A second `--compress-only` used to **return early** because language already had U8 — embeddings stayed FP32. That skip is fixed; embeddings are INT8 and vision is stashed.

Kaggle source (already downloaded, do not fetch again):

`~/.cache/kagglehub/models/google/translategemma/transformers/translategemma-4b-it/1`

Auth: `~/.kaggle/access_token` (new token), not Hugging Face. Gemma terms still apply.

### Backend wiring (already in code)

- `TranslateOVBackend`: `VLMPipeline`, IR files = language xml/bin + tokenizer (not `openvino_model.*`).
- Official TranslateGemma `chat_template.jinja` **breaks MiniJinja** (huge language map, error at `zu-ZA`). We `set_chat_template("{{ bos_token }}{{ messages[-1]['content'] }}")` and emit the trained user-turn in `athena/amc/translate.py` `format_translate_prompt()`.
- Load prints: `loading translator on GPU…` then `translator compiled`. First compile **minutes**; later `models/ov_cache/translate`.
- Chunker: 75% of 2K, sentence boundaries, both directions. `Mr.` / `3.14` not splits.
- Live test: `scripts/event_g_live.py` — translator-only en↔hi, then Event G with **Llama T only**. Scores channel vs tutor separately.

### Why a live run sat ~3 min with no text

Blocked in `VLMPipeline(...)` first GPU compile of **three** graphs (including unused vision). NVMe timeouts showed up in the journal. Process was then **killed**. Not a finished translation.

Mock noise table (injected STT/MT slips **only** on channel hops, not Llama): more hops ⇒ higher WER. `python scripts/event_g_noise.py`.

---

## First live GPU run (already happened)

`scripts/event_g_live.py` **did compile** (`translator compiled` ~25 s this time — cache may have helped), then **aborted on generate**:

```
[GPU] clFinish, error code: -5 CL_OUT_OF_RESOURCES
exit 134 SIGABRT
```

So VLMPipeline **loads**, MiniJinja passthrough is OK, then **VRAM blows** when it actually runs (language 3.9 + text emb 2.6 FP32 + vision 0.85 + KV/runtime on 12 GB). Exclusive slot was respected (Llama not loaded). **Do not retry live until `--compress-only` + stash vision.**

### Live run that got past generate, then died on Llama T

Translator-only hi→en **worked** (`resident=None` after hop). Then `enter_event_t` → `LLMPipeline` INT4 **ProgramBuilder failed**:

`CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST` (-14)

**Why (not “Llama is broken”):**

1. Translator hop still ~**7.4 GB** (INT8 language + **FP32 text embeddings** + **BF16 vision**). B580 is 12 GB **and the display GPU**.
2. Unload only dropped the Python handle. If the VLM C++ object or OpenCL buffers linger, VRAM is not empty.
3. Llama T then asks for INT4 weights **~5.1 GB + 4 GB KV** during compile. 7.4 leftover + 9 does not fit. Even a clean 9 GB + desktop framebuffer is tight.
4. `-14 WAIT_LIST` means a **previous GPU command already failed** (this process or the earlier `-5 CL_OUT_OF_RESOURCES` abort). ProgramBuilder then fails; it is not a bad INT4 IR (that IR already GPU-tested).

**87.5% WER** was a **test bug**: scored HI_Q translation against EN_Q (a different English sentence). Also `भिन्न` → “difference” is a real classroom lexical miss (math Hindi: fraction). Fixed live script uses `HI_Q_EN_GOLD`.

**Fix order:** `--compress-only` (embeddings INT8 + stash vision) so the hop is ~5–6 GB, then live again. Unload now `del`s the pipeline so the destructor can free GPU. Do not retry Llama T in the same process after `-5`/`-14` without that compress — the OpenCL context can stay poisoned until the process exits.

## After this first compile finishes

1. Read `scripts/event_g_live.py` stdout. Expect:
   - Step 1–2: translator only, Llama **not** loaded.
   - Step 3: plan `llama, translate_out`. English Q ≠ English A (tutor). Hindi is the translated **answer**.
   - Step 4: plan `translate_in, llama, translate_out`. Never `llama_int8`.
2. If **VLMPipeline / MiniJinja** error: template already patched; re-check `set_chat_template`.
3. If **OOM**: hop is still ~7.4 GB on disk (3.9+2.6+0.85). Then do embedding INT8 + stash vision (below). Do **not** load Llama at the same time.
4. If **success**: keep `models/ov_cache/translate`. Next load should be seconds.

## After compression (embeddings) — do this, in order

Do **not** re-run Kaggle download.

```bash
cd ~/Documents/projectxi1
source .venv/bin/activate
python scripts/download_translategemma.py --compress-only
```

That script (updated) should:

1. Force NNCF INT8 on language (skip if already U8) **and** text/vision embedding IRs (bypass 1B skip).
2. Stash `openvino_vision_embeddings_model.*` → `models/translategemma-4b-it-int8-ov-vision-bak`.
3. If `VLMPipeline` then refuses to load: copy those two files back.

Expected sizes after: language ~3.9 GB, text emb ~1.3 GB, vision gone from runtime (or ~0.4 GB if restored INT8).

Then re-run:

```bash
python scripts/event_g_live.py
```

First compile after file change will be slow **once** (new cache keys). Then cache.

Verify INT8:

```bash
python3 -c "
from pathlib import Path
p=Path('models/translategemma-4b-it-int8-ov')
for n in ['openvino_language_model.xml','openvino_text_embeddings_model.xml']:
    t=(p/n).read_text(errors='ignore')
    print(n, 'U8', t.count('precision=\"U8\"'), 'I8', t.count('precision=\"I8\"'), 'bin', (p/n).with_suffix('.bin').stat().st_size)
"
```

Language must have U8. Text embeddings should too after `--compress-only`.

---

## Do not

- NVIDIA / CUDA / IPEX / bitsandbytes. CPU torch only for convert. AMC runtime = OpenVINO.
- Re-export from Hugging Face (`HF_TOKEN=hf_...` was a placeholder; 401).
- Use Llama E on Event G.
- WER Llama’s answer vs the question.
- Coreside models.
- Re-download 12.8 GB.

---

## Still open (not this compile)

| Item | Owner |
|---|---|
| Embedding INT8 + vision stub | **Done.** Text emb 0.63 GB U8. Runtime vision is the 1.3 MB stub. SigLIP stays in `*-vision-bak`. |
| Live GPU Event G after KV cap | All four models exclusive + sequential GPU-tested (174.7 s). Channel WER: Whisper 12.5% (*whole→bowl* espeak), Gemma hi→en **60%** (`भिन्न`→difference). Do not WER Llama vs the question. Caps: 512. See `docs/amc.md` WER section. |
| TTS `Text2SpeechPipeline` | later |
| Athena REST combinator (real, not mimic) | Athena |
| RBA closeness / new-section RAM | wait for RBA |
| Official data-aware Llama INT4 | optional, RAM-tight |

Code map: `athena/amc/*`, `athena/event_g.py`, `athena/noise.py`, `scripts/download_translategemma.py`, `scripts/event_g_live.py`, `scripts/event_g_noise.py`, `tests/test_amc.py`, `tests/test_event_g.py`.
