#!/usr/bin/env python3
"""Live exclusive-slot switch: INT4 tutor ↔ INT8 evaluator on GPU."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from athena.amc import AMC, AMCConfig, ModelId  # noqa: E402


SECTION = {
    "id": "switch-s1",
    "title": "Fractions",
    "body": "A fraction a/b is a parts of a whole split into b equal pieces.",
}


def main() -> int:
    cfg = AMCConfig(
        device="GPU",
        backend="openvino",
        llama_model_path=str(ROOT / "models" / "llama-3.1-8b-instruct-int4-ov"),
        llama_int8_path=str(ROOT / "models" / "llama-3.1-8b-instruct-int8-ov"),
        cache_dir=str(ROOT / "models" / "ov_cache"),
        kv_cache_gb=4,
        kv_cache_gb_int8=2,
        max_new_tokens_tutor=48,
        max_new_tokens_eval=48,
        job_timeout_s=900.0,
        allow_cpu_fallback=False,
    )
    log: list[str] = []

    def snap(amc: AMC, label: str, t0: float) -> None:
        st = amc.status()
        line = (
            f"{label:24}  {time.perf_counter() - t0:6.1f}s  "
            f"resident={st.resident_model}  "
            f"int4={amc._llama_int4.is_loaded()}  "
            f"int8={amc._llama_int8.is_loaded()}  "
            f"whisper={amc._whisper.is_loaded()}"
        )
        print(line)
        log.append(line)
        both = amc._llama_int4.is_loaded() + amc._llama_int8.is_loaded() + amc._whisper.is_loaded()
        if both != 1:
            raise SystemExit(f"exclusive slot broken: {both} backends loaded")

    t_all = time.perf_counter()
    with AMC(cfg) as amc:
        t = time.perf_counter()
        amc.enter_event_t(SECTION)
        snap(amc, "1 Event T / INT4", t)
        if amc.status().resident_model is not ModelId.LLAMA_INT4:
            raise SystemExit("expected INT4")

        t = time.perf_counter()
        reply = amc.tutor_ask("What is a fraction, one sentence?")
        snap(amc, "2 tutor_ask", t)
        print(f"   tutor: {reply[:180]!r}")

        t = time.perf_counter()
        amc.enter_event_e(behavioral_level=2)
        snap(amc, "3 Event E / INT8", t)
        if amc.status().resident_model is not ModelId.LLAMA_INT8:
            raise SystemExit("expected INT8")
        if amc._llama_int4.is_loaded():
            raise SystemExit("INT4 still loaded during Event E")

        t = time.perf_counter()
        hint = amc.evaluator_hint("What is 2+2?", "5")
        snap(amc, "4 evaluator_hint", t)
        print(f"   hint:  {hint[:180]!r}")

        t = time.perf_counter()
        amc.enter_event_t(SECTION)
        snap(amc, "5 back to INT4", t)
        if amc.status().resident_model is not ModelId.LLAMA_INT4:
            raise SystemExit("expected INT4 after return")
        if amc._llama_int8.is_loaded():
            raise SystemExit("INT8 still loaded during Event T")

        t = time.perf_counter()
        reply2 = amc.tutor_ask("Give one example of a fraction.")
        snap(amc, "6 tutor_ask again", t)
        print(f"   tutor: {reply2[:180]!r}")

    print(f"SWITCH OK in {time.perf_counter() - t_all:.1f}s")
    print("(Whisper not on disk — skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
