#!/usr/bin/env python3
"""Download OmniVoice 0.2.1 weights (k2-fsa/OmniVoice).

Latest GitHub release: 0.2.1 (16 Jul 2026). Weights live on Hugging Face
`k2-fsa/OmniVoice` (~3.3 GB: 2.45 GB LM + 806 MB audio tokenizer).

This script only fetches the checkpoint. It does not pip-install `omnivoice`
(that stack pulls CUDA/XPU PyTorch). AMC runtime stays OpenVINO after NNCF.

  python scripts/download_omnivoice.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "omnivoice"
REPO = "k2-fsa/OmniVoice"
# Pin the Hugging Face snapshot that matches the 0.2.1-era hub card.
REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"

# Exact sizes from the Hub. Re-fetch if a .part / truncated file is left.
EXPECTED = {
    "model.safetensors": 2_450_000_000,  # ~2.45 GB; Hub reports Xet, check min
    "tokenizer.json": 11_400_000,
    "audio_tokenizer/model.safetensors": 806_000_000,
}

# Minimum accepted size (Hub lists rounded; Xet can differ by a few MB).
MIN_BYTES = {
    "model.safetensors": 2_300_000_000,
    "tokenizer.json": 10_000_000,
    "audio_tokenizer/model.safetensors": 750_000_000,
}

FILES = [
    "chat_template.jinja",
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "audio_tokenizer/config.json",
    "audio_tokenizer/model.safetensors",
    "audio_tokenizer/preprocessor_config.json",
]


def _hub_base() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def fetch(name: str) -> None:
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    min_size = MIN_BYTES.get(name)
    if target.exists() and target.stat().st_size > 0:
        size = target.stat().st_size
        if min_size is None or size >= min_size:
            print(f"skip {name} ({size} bytes)")
            return
        print(f"bad  {name} ({size} bytes, min {min_size}); re-fetch")
        target.unlink()
        part = DEST / f"{name}.part"
        if part.exists():
            part.unlink()
    url = f"{_hub_base()}/{REPO}/resolve/{REVISION}/{name}"
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
            "athena-amc/omnivoice-0.2.1",
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
    if min_size is not None and size < min_size:
        raise SystemExit(f"{target} is {size} bytes, expected at least {min_size}")
    print(f"ok   {target} ({size} bytes)")


def main() -> int:
    print(f"=== OmniVoice 0.2.1  {REPO}@{REVISION[:8]} -> {DEST} ===")
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "VERSION").write_text("omnivoice==0.2.1\n" + REVISION + "\n")
    for name in FILES:
        fetch(name)
    print("done OmniVoice 0.2.1 (PyTorch weights; NNCF/OpenVINO later)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
