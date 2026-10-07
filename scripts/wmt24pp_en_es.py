"""One-pair WMT24++ en→es_MX generate + chrF/BLEU for the local INT8 IR.

Keeps VLMPipeline loaded (AMC.translate unloads every call). Writes hyps
incrementally so a CCS reset is not a total loss. Not the 55-lang MetricX run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from athena.amc.backends import TranslateOVBackend
from athena.amc.config import AMCConfig
from athena.amc.translate import (
    chunk_for_translator,
    format_translate_prompt,
    join_translations,
)

SRC = ROOT / "models" / "eval" / "en-es_MX.jsonl"
HYP = ROOT / "models" / "eval" / "wmt24pp_en-es_MX_hyps.jsonl"
SCORE = ROOT / "models" / "eval" / "wmt24pp_en-es_MX_scores.json"
SOURCE_LANG = "en"
TARGET_LANG = "es-MX"


def _rows() -> list[dict]:
    out = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("domain") == "canary" or row.get("is_bad_source"):
            continue
        out.append(row)
    return out


def _done_ids() -> set[int]:
    ids: set[int] = set()
    if not HYP.exists():
        return ids
    for line in HYP.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(int(json.loads(line)["segment_id"]))
    return ids


def _generate(backend: TranslateOVBackend, prompt: str, max_new: int) -> str:
    """Do not pass images=[]. That path OOMs generate on this 12 GB card (-5)."""

    from athena.amc.backends import _as_text

    cfg = backend._pipe.get_generation_config()
    cfg.max_new_tokens = int(max_new)
    cfg.do_sample = False
    if hasattr(cfg, "max_length"):
        cfg.max_length = int(backend.config.translate_max_input_tokens) + int(max_new)
    if hasattr(cfg, "stop_token_ids"):
        cfg.stop_token_ids = {1, 106}  # <eos>, <end_of_turn>
    if hasattr(cfg, "apply_chat_template"):
        cfg.apply_chat_template = False
    return _as_text(backend._pipe.generate(prompt, generation_config=cfg))


def _translate(backend: TranslateOVBackend, cfg: AMCConfig, text: str) -> str:
    chunks = chunk_for_translator(
        text,
        source_lang=SOURCE_LANG,
        target_lang=TARGET_LANG,
        count=backend.token_count,
        max_input_tokens=cfg.translate_max_input_tokens,
        fill_ratio=cfg.translate_fill_ratio,
    )
    hard = cfg.translate_max_input_tokens
    fill_cap = max(32, int(hard * cfg.translate_fill_ratio))
    pieces: list[str] = []
    for chunk in chunks:
        prompt = format_translate_prompt(
            chunk.text, source_lang=SOURCE_LANG, target_lang=TARGET_LANG
        )
        used = backend.token_count(prompt)
        max_new = min(
            cfg.max_new_tokens_translate,
            fill_cap,
            max(32, hard - used),
        )
        pieces.append(_generate(backend, prompt, max_new))
    return join_translations(chunks, pieces)


def score() -> dict:
    import sacrebleu

    refs: list[str] = []
    hyps: list[str] = []
    by_id = {
        json.loads(line)["segment_id"]: json.loads(line)
        for line in HYP.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for row in _rows():
        rec = by_id.get(int(row["segment_id"]))
        if rec is None:
            continue
        refs.append(row["target"])
        hyps.append(rec["hyp"])
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    chrfpp = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    out = {
        "pair": "en-es_MX",
        "dataset": "google/wmt24pp",
        "n_scored": len(hyps),
        "n_source_file": 998,
        "skipped_canary_or_bad": 998 - len(_rows()),
        "bleu": round(float(bleu.score), 2),
        "chrf": round(float(chrf.score), 2),
        "chrfpp": round(float(chrfpp.score), 2),
        "metricx": None,
        "comet": None,
        "note": "INT8 OpenVINO IR; greedy; not 55-lang MetricX/COMET",
    }
    SCORE.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)
    return out


def main() -> int:
    rows = _rows()
    done = _done_ids()
    print(f"segments {len(rows)} already {len(done)}", flush=True)
    cfg = AMCConfig()
    # Do not reuse ov_cache/translate: it still holds 7.3 GB blobs from the
    # pre-stub fat IR. Those graphs generate -5 on this 12 GB card.
    cfg.cache_dir = str(ROOT / "models" / "ov_cache" / "wmt24")
    backend = TranslateOVBackend(cfg)
    HYP.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        backend.load()
        with HYP.open("a", encoding="utf-8") as fh:
            for i, row in enumerate(rows, 1):
                sid = int(row["segment_id"])
                if sid in done:
                    continue
                t1 = time.time()
                hyp = _translate(backend, cfg, row["source"])
                rec = {
                    "segment_id": sid,
                    "domain": row.get("domain"),
                    "source": row["source"],
                    "ref": row["target"],
                    "hyp": hyp,
                    "sec": round(time.time() - t1, 2),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done.add(sid)
                if i % 10 == 0 or i == len(rows):
                    elapsed = time.time() - t0
                    print(
                        f"{len(done)}/{len(rows)} last_sid={sid} "
                        f"last_s={rec['sec']} elapsed_min={elapsed/60:.1f}",
                        flush=True,
                    )
    finally:
        backend.unload()
    if len(_done_ids()) < len(rows):
        print("INCOMPLETE", len(_done_ids()), len(rows), flush=True)
        return 2
    score()
    print("WMT24PP_EN_ES_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
