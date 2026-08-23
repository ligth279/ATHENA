#!/usr/bin/env python3
"""Live Event G on the B580: TranslateGemma + Llama T (INT4 tutor).

Does **not** use Llama E. Llama T is supposed to turn the question into an
answer — that change is not scored as hop noise.

Scores:
  1) translator-only (en↔hi) — channel quality
  2) Event G: en question → Llama T → hi answer  (translate_out only)
  3) Event G: hi question → translate_in → Llama T → translate_out
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from athena.amc import AMC, AMCConfig, ModelId
from athena.amc.exceptions import GpuDeadError, ModelPathError
from athena.event_g import EventG
from athena.noise import word_error_rate

SECTION = {
    "id": "g-live-s1",
    "title": "Fractions",
    "body": (
        "A fraction a/b is a parts of a whole split into b equal pieces. "
        "The top number is the numerator, the bottom is the denominator."
    ),
}

EN_Q = "What is a fraction, in one short sentence?"
HI_Q = "भिन्न क्या है? एक छोटे वाक्य में बताओ।"
# Gold for HI_Q only. Do not WER HI→EN against EN_Q (different sentence).
HI_Q_EN_GOLD = "What is a fraction? Tell me in one short sentence."
EN_PARA = (
    "A fraction a/b is a parts of a whole split into b equal pieces. "
    "The top number is the numerator, the bottom is the denominator. "
    "Two fractions are equivalent when they name the same amount. "
    "To add fractions with the same denominator, add the numerators and keep the denominator. "
    "To add fractions with different denominators, first find a common denominator."
)


def _cfg() -> AMCConfig:
    return AMCConfig(
        device="GPU",
        backend="openvino",
        llama_model_path=str(ROOT / "models" / "llama-3.1-8b-instruct-int4-ov"),
        llama_int8_path=str(ROOT / "models" / "llama-3.1-8b-instruct-int8-ov"),
        whisper_model_path=str(ROOT / "models" / "whisper-large-v3-turbo-int8-ov"),
        translate_model_path=str(ROOT / "models" / "translategemma-4b-it-int8-ov"),
        cache_dir=str(ROOT / "models" / "ov_cache"),
        kv_cache_gb=4,
        kv_cache_gb_int8=2,
        kv_cache_gb_translate=1,
        max_new_tokens_tutor=512,
        max_new_tokens_translate=512,
        job_timeout_s=1200.0,
        allow_cpu_fallback=False,
    )


def _banner(title: str) -> None:
    print()
    print("==", title)


def main() -> int:
    cfg = _cfg()
    ir = Path(cfg.translate_model_path)
    lang = ir / "openvino_language_model.bin"
    if not lang.is_file():
        print(f"missing TranslateGemma IR at {ir}", file=sys.stderr)
        return 2
    print("xilo Event G live  GPU  Llama T (INT4) + TranslateGemma")
    print(f"  language IR: {lang.stat().st_size} bytes")
    print("  evaluator INT8 is not loaded in this test")

    t_all = time.perf_counter()
    try:
        with AMC(cfg) as amc:
            _banner("1 translator only  en → hi  (no Llama)")
            t = time.perf_counter()
            hi = amc.translate(EN_Q, source_lang="en", target_lang="hi")
            print(f"  [{time.perf_counter()-t:.1f}s] resident={amc.status().resident_model}")
            print(f"  en: {EN_Q}")
            print(f"  hi: {hi}")
            if amc._llama_int4.is_loaded() or amc._llama_int8.is_loaded():
                raise SystemExit("Llama loaded during translator-only hop")
            if not hi.strip():
                raise SystemExit("empty translation")
            if hi.strip() == EN_Q:
                print("  WARN: hi output identical to English source")

            _banner("1b translator only  en paragraph → hi  (chunked, no Llama)")
            t = time.perf_counter()
            hi_para = amc.translate(EN_PARA, source_lang="en", target_lang="hi")
            print(f"  [{time.perf_counter()-t:.1f}s] resident={amc.status().resident_model}")
            print(f"  en chars={len(EN_PARA)} sentences~{EN_PARA.count('.')}")
            print(f"  hi: {hi_para}")
            if not hi_para.strip():
                raise SystemExit("empty paragraph translation")
            if amc._llama_int4.is_loaded() or amc._llama_int8.is_loaded():
                raise SystemExit("Llama loaded during translator paragraph hop")

            _banner("2 translator only  hi → en  (no Llama)")
            t = time.perf_counter()
            en_back = amc.translate(HI_Q, source_lang="hi", target_lang="en")
            print(f"  [{time.perf_counter()-t:.1f}s] resident={amc.status().resident_model}")
            print(f"  hi: {HI_Q}")
            print(f"  en: {en_back}")
            print(f"  gold for this Hindi (not the English test Q): {HI_Q_EN_GOLD}")
            print(f"  WER vs that gold: {word_error_rate(HI_Q_EN_GOLD, en_back):.1%}")
            if "difference" in en_back.lower() and "fraction" not in en_back.lower():
                print(
                    "  note: भिन्न→difference is a lexical miss in math class "
                    "(भिन्न = fraction in the section, = difference in general Hindi)"
                )

            g = EventG(amc, SECTION)
            _banner("3 Event G  en type → hi out  (Llama T + translate_out)")
            t = time.perf_counter()
            r = g.run(text=EN_Q, source_lang="en", target_lang="hi")
            print(f"  [{time.perf_counter()-t:.1f}s] plan={r.plan}")
            print(f"  resident after={amc.status().resident_model}")
            print(f"  Q (en, into Llama T): {r.english_question}")
            print(f"  A (en, Llama T):      {r.english_answer}")
            print(f"  A (hi, translate_out): {r.text}")
            if amc.status().resident_model is ModelId.LLAMA_INT8:
                raise SystemExit("used Llama E — this test is tutor INT4 only")
            if r.english_answer.strip() == r.english_question.strip():
                raise SystemExit("Llama T echoed the question; expected an answer")
            print(
                f"  note: WER(question, Llama answer) is the wrong metric "
                f"({word_error_rate(r.english_question, r.english_answer):.0%} "
                f"just means the tutor answered)"
            )

            _banner("4 Event G  hi type → hi out  (in + Llama T + out)")
            t = time.perf_counter()
            r2 = g.run(text=HI_Q, source_lang="hi", target_lang="hi")
            print(f"  [{time.perf_counter()-t:.1f}s] plan={r2.plan}")
            print(f"  translate_in  → {r2.english_question}")
            print(f"  Llama T (en)  → {r2.english_answer}")
            print(f"  translate_out → {r2.text}")
            if r2.english_answer.strip() == r2.english_question.strip():
                raise SystemExit("Llama T echoed the translated question")
            print(f"  exclusive llama_e loaded={amc._llama_int8.is_loaded()}")
    except GpuDeadError as exc:
        print(exc, file=sys.stderr)
        print(
            "GPU OpenCL is poisoned by a failed command. Do not retry in this process.\n"
            "Exit. A reboot only clears a reset Xe CCS engine; the generate -5 "
            "comes back if translator KV is uncapped. Then a NEW "
            "python scripts/event_g_live.py",
            file=sys.stderr,
        )
        return 3
    except ModelPathError as exc:
        print(exc, file=sys.stderr)
        print(
            "If this is TranslateGemma VRAM: "
            "python scripts/download_translategemma.py --compress-only",
            file=sys.stderr,
        )
        return 2
    print(f"\nLIVE OK in {time.perf_counter()-t_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
