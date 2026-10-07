# Athena

AMC (AI Model Controller) loads one model on the GPU, runs it, unloads it, then starts the next.

Hardware this code has been tested: **Intel Arc B580 12 GB**, **i5-10400F**, **16 GB DDR4**. OpenVINO device `"GPU"` is the B580.

Stack: **OpenVINO 2026.4.1** + **openvino-genai** + **oneAPI**. Regular PyTorch only where conversion needs it. IPEX is unsupported. Do not add bitsandbytes, NVIDIA, or CUDA packages. Do not install new dependencies without administrator approval.

---

## Current done

### Exclusive GPU slot

The B580 holds **one** model at a time. A single worker queue waits until the resident job finishes, unloads that model, then starts the next.

| Resident | Event | IR | Reserved GPU KV |
|---|---|---|---|
| Llama 3.1 8B INT4 | T | `models/llama-3.1-8b-instruct-int4-ov` | 4 GB (`cl_mem`) |
| Llama 3.1 8B INT8 | E | `models/llama-3.1-8b-instruct-int8-ov` | 2 GB (`cl_mem`) |
| Whisper large-v3 turbo INT8 | G hop | `models/whisper-large-v3-turbo-int8-ov` | none |
| TranslateGemma 4B INT8 | G hop | `models/translategemma-4b-it-int8-ov` | none |
| OmniVoice 0.2.1 FP16 | G hop | `models/omnivoice-fp16-ov` | none |

Never two in VRAM. Compile caches: `models/ov_cache/int4`, `int8`, `translate`, `omnivoice`. TranslateGemma language + embeddings are INT8; vision is a **1.3 MB stub** (SigLIP stays in `*-vision-bak`). OmniVoice TTS is two OpenVINO graphs (`forward.xml` embed+LLM+heads, `decode.xml` acoustic). PyTorch is export-only.

Llama keeps a reserved KV window for chat (`scheduler_config.cache_size`). Whisper and TranslateGemma do not: they run, return a string, and AMC **unloads** them. TranslateGemma is not given a KV `cache_size` — that switch crashes `VLMPipeline` on this card. While it is resident, `cl_mem` stays ~0; weights sit in `usm_device` (~4.25 GiB) and drop to **0** on unload.

### Wait, then start the next (verified)

Mock backends (`python -m unittest tests.test_amc tests.test_event_g -q`). Exclusive-slot tests **ok**.

| Test | What it proves |
|---|---|
| `test_never_two_models_loaded` | INT4 → Whisper → TranslateGemma → TTS → INT8. Never two loaded. |
| `test_second_request_waits_for_first` | Next request waits until generate finishes, then switches. |
| `test_event_e_uses_int8_not_int4` | Event E unloads INT4, loads INT8. |
| `test_speak_unloads_llama_and_loads_whisper` | STT hop: Llama off, Whisper on. |
| `test_tutor_after_whisper_reloads_llama_with_history` | After STT, INT4 reloads; Whisper off. |
| `test_translate_unloads_llama_and_unloads_after` | Translate unloads Llama, generates, unloads itself. |
| `test_roundtrip_calls_translate_twice_and_unloads` | `translate_in → llama → translate_out`. Translator and Llama unloaded. |
| `test_speech_uses_whisper_then_translate_in` | `stt → translate_in → llama`. Whisper unloaded after pass-on. |
| `test_english_only_skips_translator` | English in/out is Llama only. |
| `test_talk_is_not_whisper` | Talk is OmniVoice TTS, unloads, not Whisper. |
| `test_speak_runs_tts_and_unloads` | Event G `+ speak` runs TTS then unloads. |
| `test_shutdown_unloads` | Shutdown leaves no resident. |

### Live GPU metrics (one resident at a time)

Same AMC worker, **Intel(R) Arc(TM) B580 Graphics (dGPU)**, cached IR. Order: Whisper → TranslateGemma → Llama INT4 → Llama INT8. Exclusive check passed at every step. Wall **44.33 s**. Final `usm_device` **0**.

| Model | Boot | Run | Unload | USM at boot | KV (`cl_mem`) while resident | USM after unload |
|---|---|---|---|---|---|---|
| Whisper large-v3 turbo INT8 | 2.09 s | 0.48 s | 0.05 s | 0.76 GiB | 0.01 GiB | 0 |
| TranslateGemma 4B INT8 | 18.01 s | 1.07 s | 0.15 s | 4.25 GiB | ~0 (no reserved KV; AMC unloads after the hop) | 0 |
| Llama 3.1 8B INT4 | 10.60 s | 0.89 s | 0.23 s | 5.13 GiB | 4.00 GiB | 0 |
| Llama 3.1 8B INT8 | 10.15 s | 0.50 s | 0.11 s | 7.48 GiB | 2.00 GiB | 0 |

TranslateGemma `cl_mem` during generate was ~0; `usm_device` 4.25 → 4.28 GiB, then **0** after AMC unload.

OmniVoice TTS was measured in its own exclusive process (`python scripts/tts_live.py`). OpenVINO `"GPU"` only — no PyTorch. Speak *A fraction is a part of a whole.*, unload, then Whisper STT on 24 kHz→16 kHz resample.

| Model | Boot | Run | Unload | USM at boot | KV (`cl_mem`) while resident | USM after unload |
|---|---|---|---|---|---|---|
| OmniVoice 0.2.1 FP16 | 0.69 s | 2.61 s | 0.08 s | 1.18 GiB | ~0 | 0 |

`amc.talk()` cached exclusive hop **1.90 s**, resident **None** after unload. Whisper hyp matched the gold sentence (**0.0% WER**). Final `usm_device` **0**.

Accuracy follow-up (`python scripts/tts_accuracy.py`): 10 English lines + es/hi/de, TTS resident then Whisper resident, exclusive gold round-trip last. TTS USM **1.18 GiB → 0**. Wall **36.0 s**.

| Set | n | Mean raw WER | Mean punct-stripped WER |
|---|---|---|---|
| English | 10 | 6.5% | 5.3% |
| English + es/hi/de | 13 | 5.0% | 4.1% |
| Gold exclusive hop | 1 | 0.0% | 0.0% |

Seven English lines matched exactly. The three misses were Whisper writing digits/punctuation (`Two`→`2.`, `One half`→`1 half`, dropped `,?`). Spanish, Hindi, and German lines were exact. Gold line still **0.0%** on the exclusive talk→transcribe hop.

Official Seed-TTS English (`python scripts/seedtts_turbo.py`): same 1088 texts as OmniVoice’s paper eval, our OV hop (auto-voice, 16 steps), Whisper **turbo**, OmniVoice `seedtts` post_process. Wall **1272.8 s**.

| Set | n | Corpus WER | Paper (clone, 32 step, Whisper large-v3) |
|---|---|---|---|
| Seed-TTS English | 1088 | **1.69%** (S=167 D=28 I=4 W=11806) | **1.60%** |

Mean sentence raw WER 4.04% (punctuation still on). Mean sentence norm WER 1.73%. Turbo + official sentences sit **0.09 points** above the paper number. The 6.5% classroom set was the short n=10 list, not this benchmark.

### Events

- **T** — load INT4, `tutor_ask`
- **E** — unload INT4, load INT8, `evaluator_hint` (one-shot)
- **G** — `athena/event_g.py` adds STT / translate / TTS hops only when needed. Those hops unload after pass-on.

| Combo | Hops |
|---|---|
| type English → English | llama |
| type English → other language | llama → translate_out |
| type other language → that language | translate_in → llama → translate_out |
| speech in other language | stt → translate_in → llama → translate_out |
| English speech | stt → llama |
| + speak | same + tts |

### Public API

```python
from athena.amc import AMC, AMCConfig

amc = AMC.mock()
# amc = AMC(AMCConfig())

amc.enter_event_t(section_json)
reply = amc.tutor_ask("why is this?")

amc.transcribe(pcm_16k)
amc.translate(text, source_lang="hi", target_lang="en")
amc.talk("A fraction is a part of a whole.", language="en")

amc.enter_event_e(behavioral_level=2)
hint = amc.evaluator_hint(q, student, allow_approximate=False)

amc.leave_role()
amc.shutdown()
```

### CLI

```bash
python -m athena.amc status
python -m athena.amc tutor "What is a fraction, in one short sentence?"
python -m athena.amc eval --question "What is 2+2?" --answer "5"
python -m athena.amc translate --source en --target hi "A fraction is a part of a whole."
python -m athena.amc talk "A fraction is a part of a whole."
python scripts/switch_models.py
python scripts/event_g_live.py
python scripts/tts_live.py
python -m unittest tests.test_amc tests.test_event_g -q
```

Runtime extras: `requirements-amc.txt` (install only after administrator approval).

IR / compile notes: `docs/amc.md`, `docs/amc-context.md`.

---

## Coming soon

This will come soon.
