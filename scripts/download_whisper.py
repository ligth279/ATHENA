#!/usr/bin/env python3
"""Download OpenVINO multilingual Whisper large-v3-turbo INT8.

English-only Distil-Whisper is not used. Event G STT hop: transcribe, pass
text, unload. ~0.8 GB.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "whisper-large-v3-turbo-int8-ov"
REPO = "OpenVINO/whisper-large-v3-turbo-int8-ov"

EXPECTED = {
    "openvino_encoder_model.bin": 645332592,
    "openvino_decoder_model.bin": 172534710,
}

FILES = [
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
]


def fetch(name: str, expected: int | None = None) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / name
    if target.exists() and target.stat().st_size > 0:
        size = target.stat().st_size
        if expected is None or size == expected:
            print(f"skip {name} ({size} bytes)")
            return
        print(f"bad  {name} ({size} bytes, expected {expected}); re-fetch")
        target.unlink()
        part = DEST / f"{name}.part"
        if part.exists():
            part.unlink()
    url = f"https://huggingface.co/{REPO}/resolve/main/{name}"
    part = DEST / f"{name}.part"
    print(f"get  {url}")
    subprocess.check_call(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "8",
            "--retry-all-errors",
            "-C",
            "-",
            "-A",
            "xilo-v6-amc/1.0",
            "-o",
            str(part),
            url,
        ]
    )
    if part.exists():
        part.replace(target)
    elif not target.exists():
        raise FileNotFoundError(f"curl finished but neither {part} nor {target} exists")
    size = target.stat().st_size
    if expected is not None and size != expected:
        raise SystemExit(f"{target} is {size} bytes, expected {expected}")
    print(f"ok   {target} ({size} bytes)")


def main() -> int:
    print(f"=== whisper turbo int8: {REPO} -> {DEST} ===")
    for name in FILES:
        fetch(name, EXPECTED.get(name))
    print("done whisper-large-v3-turbo-int8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
