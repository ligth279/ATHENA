#!/usr/bin/env python3
"""Download NNCF OpenVINO IR for Llama 3.1 8B Instruct INT4 + INT8.

Sources (optimum-cli / NNCF, not GGUF):
  CelesteImperia/Llama-3.1-8B-Instruct-OpenVINO-INT4  (~5.1 GB weights)
  CelesteImperia/Llama-3.1-8B-Instruct-OpenVINO-INT8  (~7.5 GB weights)

Resumes .part files. Does not fetch Whisper.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

REPOS = {
    "int4": (
        "CelesteImperia/Llama-3.1-8B-Instruct-OpenVINO-INT4",
        MODELS / "llama-3.1-8b-instruct-int4-ov",
        {"openvino_model.bin": 5459614805},
    ),
    "int8": (
        "CelesteImperia/Llama-3.1-8B-Instruct-OpenVINO-INT8",
        MODELS / "llama-3.1-8b-instruct-int8-ov",
        {"openvino_model.bin": 8035958805},
    ),
}

FILES = [
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "openvino_config.json",
    "openvino_detokenizer.bin",
    "openvino_detokenizer.xml",
    "openvino_model.bin",
    "openvino_model.xml",
    "openvino_tokenizer.bin",
    "openvino_tokenizer.xml",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def fetch(repo: str, dest: Path, name: str, expected: int | None = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / name
    if target.exists() and target.stat().st_size > 0:
        size = target.stat().st_size
        if expected is None or size == expected:
            print(f"skip {target.name} ({size} bytes)")
            return
        print(f"bad  {target.name} ({size} bytes, expected {expected}); re-fetch")
        target.unlink()
        part = dest / f"{name}.part"
        if part.exists():
            part.unlink()
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    part = dest / f"{name}.part"
    print(f"get  {url}")
    cmd = [
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
    subprocess.check_call(cmd)
    if part.exists():
        part.replace(target)
    elif not target.exists():
        raise FileNotFoundError(f"curl finished but neither {part} nor {target} exists")
    size = target.stat().st_size
    if expected is not None and size != expected:
        raise SystemExit(
            f"{target} is {size} bytes, expected {expected}. "
            "Two downloads probably raced; delete the file and run again once."
        )
    print(f"ok   {target} ({size} bytes)")


def main() -> int:
    which = (sys.argv[1] if len(sys.argv) > 1 else "int4").lower()
    if which not in REPOS:
        print(f"usage: {sys.argv[0]} int4|int8", file=sys.stderr)
        return 2
    repo, dest, expected = REPOS[which]
    print(f"=== {which}: {repo} -> {dest} ===")
    for name in FILES:
        fetch(repo, dest, name, expected.get(name))
    print(f"done {which}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
