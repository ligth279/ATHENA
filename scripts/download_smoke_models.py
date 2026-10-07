#!/usr/bin/env python3
"""Fetch small OpenVINO IRs for a GPU smoke test. Not Llama 3.1 8B."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "models" / "smoke"

MODELS = {
    "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov": [
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "openvino_detokenizer.bin",
        "openvino_detokenizer.xml",
        "openvino_model.bin",
        "openvino_model.xml",
        "openvino_tokenizer.bin",
        "openvino_tokenizer.xml",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ],
    "OpenVINO/whisper-tiny-int8-ov": [
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "normalizer.json",
        "openvino_config.json",
        "openvino_decoder_model.bin",
        "openvino_decoder_model.xml",
        "openvino_detokenizer.bin",
        "openvino_detokenizer.xml",
        "openvino_encoder_model.bin",
        "openvino_encoder_model.xml",
        "openvino_tokenizer.bin",
        "openvino_tokenizer.xml",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
}

UA = "xilo-v6-amc-smoke/1.0"


def dest_dir(repo: str) -> Path:
    return SMOKE / repo.split("/", 1)[1]


def download_file(repo: str, name: str) -> None:
    target = dest_dir(repo) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"skip {repo}/{name} ({target.stat().st_size} bytes)")
        return
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    print(f"get  {url}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(target)
    print(f"ok   {target} ({target.stat().st_size} bytes)")


def main() -> int:
    for repo, files in MODELS.items():
        for name in files:
            download_file(repo, name)
    print(f"smoke IRs in {SMOKE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
