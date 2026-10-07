#!/usr/bin/env python3
"""Create/update light434/OmniVoice-0.2.1-fp16-ov on Hugging Face.

Needs: hf auth login   (write token for user light434)
Staging folder: /tmp/hf_omnivoice_fp16_ov
"""

from __future__ import annotations

from pathlib import Path

REPO = "light434/OmniVoice-0.2.1-fp16-ov"
STAGING = Path("/tmp/hf_omnivoice_fp16_ov")


def main() -> int:
    from huggingface_hub import HfApi, create_repo, login, whoami

    info = whoami()
    name = info.get("name") or info.get("fullname")
    print("logged in as", info.get("name"), flush=True)
    if info.get("name") != "light434":
        print(f"warning: expected light434, got {info.get('name')}")

    create_repo(REPO, repo_type="model", exist_ok=True, private=False)
    api = HfApi()
    api.upload_folder(
        folder_path=str(STAGING),
        repo_id=REPO,
        repo_type="model",
        commit_message="candidate: OmniVoice 0.2.1 OpenVINO FP16 IR (Seed-TTS en WER 1.69%)",
        ignore_patterns=["ov_cache/**", "__pycache__/**", "talk_gpu_cfg.py"],
    )
    print(f"https://huggingface.co/{REPO}")
    print(f"https://huggingface.co/{REPO}/tree/gpu-cfg-fp32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
