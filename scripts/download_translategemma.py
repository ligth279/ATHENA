#!/usr/bin/env python3
"""Export TranslateGemma 4B INT8 OpenVINO IR for the Event G hop.

Uses the Optimum Python API (not optimum-cli). Fedora venvs put the same
package on both lib/ and lib64/; optimum-cli then registers `export openvino`
twice and dies with `conflicting subparser: openvino`.

Conversion runs on **CPU** (regular PyTorch +cpu). Inference later is
OpenVINO on the Arc B580. NVIDIA/CUDA is forbidden.

Prefer Kaggle if you already accepted Gemma there (no Hugging Face token):

  # ~/.kaggle/kaggle.json from https://www.kaggle.com/settings (API token)
  python scripts/download_translategemma.py --from-kaggle
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "translategemma-4b-it-int8-ov"
HF_REPO = "google/translategemma-4b-it"
KAGGLE_HANDLE = "google/translategemma/transformers/translategemma-4b-it"

_NVIDIA_MARKERS = (
    "nvidia-",
    "cuda",
    "cublas",
    "cudnn",
    "cufft",
    "nvinfer",
)


def _ir_complete(dest: Path) -> bool:
    if not dest.is_dir():
        return False
    xmls = list(dest.glob("*.xml"))
    bins = [
        p
        for p in dest.glob("*.bin")
        if "tokenizer" not in p.name and "detokenizer" not in p.name
    ]
    return bool(xmls) and any(p.stat().st_size > 100_000_000 for p in bins)


def _xml_has_int8(xml: Path) -> bool:
    if not xml.is_file() or xml.stat().st_size == 0:
        return False
    text = xml.read_text(errors="ignore")
    return 'precision="I8"' in text or 'precision="U8"' in text


def _has_int8_weights(dest: Path) -> bool:
    """True if the language IR actually stores INT8 constants."""

    return _xml_has_int8(dest / "openvino_language_model.xml")


def _text_embeddings_int8(dest: Path) -> bool:
    return _xml_has_int8(dest / "openvino_text_embeddings_model.xml")


def _vision_is_stub(dest: Path) -> bool:
    xml = dest / "openvino_vision_embeddings_model.xml"
    binp = dest / "openvino_vision_embeddings_model.bin"
    return (
        xml.is_file()
        and binp.is_file()
        and binp.stat().st_size <= 10_000_000
    )


def _hop_gpu_ready(dest: Path) -> bool:
    """Language INT8 + text embeddings INT8 + SigLIP replaced by the stub.

    VLMPipeline still opens the vision xml. The real SigLIP encoder is the
    generate -5 (short sentence, INT8 weights, 12 GB display GPU).
    """

    return (
        _has_int8_weights(dest)
        and _text_embeddings_int8(dest)
        and _vision_is_stub(dest)
    )


def _force_int8_xml(xml: Path) -> None:
    """INT8 even when the graph is under Optimum's 1B-parameter skip."""

    if _xml_has_int8(xml):
        print(f"skip already INT8 {xml.name}")
        return
    from openvino import Core, save_model
    from optimum.intel.openvino.quantization import _weight_only_quantization

    core = Core()
    print(f"NNCF INT8 {xml.name} ({xml.with_suffix('.bin').stat().st_size} bytes)")
    model = core.read_model(str(xml))
    _weight_only_quantization(
        model, {"bits": 8, "sym": False, "ratio": 1.0, "group_size": -1}
    )
    compressed = xml.with_name(xml.stem + "_compressed.xml")
    save_model(model, compressed, compress_to_fp16=False)
    if not _xml_has_int8(compressed):
        compressed.unlink(missing_ok=True)
        compressed.with_suffix(".bin").unlink(missing_ok=True)
        raise SystemExit(f"NNCF did not produce INT8 constants in {xml.name}")
    xml.unlink()
    xml.with_suffix(".bin").unlink()
    compressed.rename(xml)
    compressed.with_suffix(".bin").rename(xml.with_suffix(".bin"))
    print(f"  now {xml.with_suffix('.bin').stat().st_size} bytes")


def _backup_real_vision(dest: Path) -> None:
    """Copy the real SigLIP IR aside before replacing it with the stub."""

    import shutil

    binp = dest / "openvino_vision_embeddings_model.bin"
    if not binp.is_file() or binp.stat().st_size <= 10_000_000:
        return
    bak = dest.parent / (dest.name + "-vision-bak")
    bak.mkdir(exist_ok=True)
    copied = []
    for p in dest.glob("openvino_vision_embeddings_model.*"):
        target = bak / p.name
        if target.exists() and target.stat().st_size == p.stat().st_size:
            continue
        shutil.copy2(p, target)
        copied.append(p.name)
    if copied:
        print(f"backed up SigLIP {copied} -> {bak}")


def _install_vision_stub(dest: Path) -> None:
    """VLMPipeline requires the vision xml. Event G must not run SigLIP."""

    if _vision_is_stub(dest):
        print("vision stub already in runtime folder")
        return
    import numpy as np
    import openvino as ov
    from openvino import opset8 as op

    _backup_real_vision(dest)
    pixel = op.parameter(ov.PartialShape([-1, 3, -1, -1]), ov.Type.f32, "pixel_values")
    zeros = op.constant(np.zeros((1, 256, 2560), np.float32))
    zeros.output(0).set_names({"last_hidden_state"})
    model = ov.Model([zeros], [pixel], "vision_stub")
    xml = dest / "openvino_vision_embeddings_model.xml"
    ov.save_model(model, str(xml), compress_to_fp16=True)
    print(
        f"installed text-only vision stub "
        f"({xml.with_suffix('.bin').stat().st_size} bytes)"
    )


def _compress_ir(dest: Path) -> None:
    """NNCF INT8 on language (already done) and embedding IRs (1B skip)."""

    xmls = [
        dest / "openvino_language_model.xml",
        dest / "openvino_text_embeddings_model.xml",
    ]
    found = [p for p in xmls if p.is_file() and p.stat().st_size > 0]
    if not found:
        raise SystemExit(f"no OpenVINO submodel XML under {dest}")
    for xml in found:
        _force_int8_xml(xml)
    if not _has_int8_weights(dest):
        raise SystemExit(
            "compress finished but language IR still has no I8 constants"
        )


def _refuse_nvidia() -> str | None:
    try:
        import importlib.metadata as metadata
    except ImportError:
        return None
    bad = []
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower()
        if name.startswith("nvidia-") or "cuda" in name:
            bad.append(name)
    if bad:
        return "NVIDIA/CUDA packages in this venv: " + ", ".join(sorted(set(bad)))
    try:
        import torch
    except ImportError:
        return None
    ver = str(getattr(torch, "__version__", ""))
    if bool(getattr(torch, "cuda", None) and torch.cuda.is_available()):
        return f"this venv has CUDA PyTorch ({ver}). Use CPU torch only."
    if "cu" in ver and "+cpu" not in ver:
        return f"this venv has CUDA PyTorch ({ver}). Use CPU torch only."
    return None


def _int8_ov_config():
    from optimum.intel.openvino.configuration import OVConfig

    return OVConfig(
        quantization_config={
            "bits": 8,
            "ratio": 1.0,
            "group_size": -1,
            "sym": False,
            "quant_method": "default",
            "dtype": "int8",
        }
    )


def _is_placeholder_token(token: str) -> bool:
    stripped = token.strip()
    return stripped in {"hf_...", "hf_", "...", "YOUR_TOKEN", "your_token"}


def _is_auth_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "gated repo" in text or "401" in text or "restricted" in text


def _looks_like_hf_checkpoint(path: Path) -> bool:
    return (path / "config.json").is_file() and any(path.glob("*.safetensors"))


def _kaggle_creds_ok() -> bool:
    home = Path.home() / ".kaggle"
    if (home / "kaggle.json").is_file():
        return True
    token_file = home / "access_token"
    txt_file = home / "access_token.txt"
    if token_file.is_file() and token_file.stat().st_size > 8:
        return True
    if txt_file.is_file() and txt_file.stat().st_size > 8:
        return True
    env_token = os.environ.get("KAGGLE_API_TOKEN")
    if env_token and not _is_placeholder_token(env_token):
        return True
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def _download_kaggle() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is not installed. One-shot (CPU, no NVIDIA):\n"
            "  pip install kagglehub"
        ) from exc
    if not _kaggle_creds_ok():
        raise SystemExit(
            "No Kaggle credentials (the token file was never written).\n"
            "Quotes must wrap ONLY the token. The > stays outside:\n"
            "  mkdir -p ~/.kaggle\n"
            "  echo 'PASTE_TOKEN_HERE' > ~/.kaggle/access_token\n"
            "  chmod 600 ~/.kaggle/access_token\n"
            "  python scripts/download_translategemma.py --from-kaggle"
        )
    print(f"kagglehub download  {KAGGLE_HANDLE}")
    raw = Path(kagglehub.model_download(KAGGLE_HANDLE))
    print(f"kaggle files at {raw}")
    if not _looks_like_hf_checkpoint(raw):
        raise SystemExit(f"Kaggle download has no config.json / safetensors at {raw}")
    return raw


def _export(source: str, dest: Path, token: str | None) -> None:
    try:
        from optimum.exporters.openvino import main_export
    except ImportError as exc:
        raise SystemExit(
            "optimum-intel is not importable in this interpreter.\n"
            f"{exc}\n"
            "CPU torch only. Do not pip-install CUDA."
        ) from exc

    dest.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {}
    if token:
        kwargs["token"] = token
    last_err: BaseException | None = None
    # Gemma 3 only exports as image-text-to-text. Do NOT pass ov_config here:
    # a set quantization_config makes main_export skip its >1B INT8 pass.
    for task in ("image-text-to-text", "auto"):
        print(f"export  {source}  task={task}  device=cpu  (then NNCF INT8)")
        try:
            main_export(
                model_name_or_path=source,
                output=str(dest),
                task=task,
                device="cpu",
                framework="pt",
                trust_remote_code=True,
                ov_config=None,
                stateful=True,
                convert_tokenizer=True,
                **kwargs,
            )
            return
        except BaseException as exc:
            last_err = exc
            print(f"task {task} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            if _is_auth_error(exc):
                raise SystemExit(
                    "Hugging Face rejected the token.\n"
                    "Use Kaggle instead (same official weights):\n"
                    "  python scripts/download_translategemma.py --from-kaggle"
                ) from exc
    raise SystemExit(f"all export tasks failed: {last_err}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="TranslateGemma 4B INT8 OpenVINO export (Arc B580, no CUDA)"
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--from-kaggle",
        action="store_true",
        help="download official 4B from Kaggle, then convert (no HF token)",
    )
    src.add_argument(
        "--local",
        metavar="DIR",
        help="already-downloaded HuggingFace/Kaggle checkpoint directory",
    )
    src.add_argument(
        "--compress-only",
        action="store_true",
        help="NNCF INT8 the IR already in models/ (no re-download)",
    )
    args = parser.parse_args()

    nvidia = _refuse_nvidia()
    if nvidia:
        print(nvidia, file=sys.stderr)
        print("Arc B580 / OpenVINO only. CPU torch for conversion.", file=sys.stderr)
        return 2

    print(f"=== translategemma 4b int8 -> {DEST} ===")
    if args.compress_only:
        if not _ir_complete(DEST):
            print(f"no IR at {DEST} to compress", file=sys.stderr)
            return 2
        # Language INT8 is not enough. SigLIP in the runtime folder is the
        # generate -5 (even INT8, even a short sentence).
        if _hop_gpu_ready(DEST):
            print("hop already GPU-ready (language+embeddings INT8, vision stub)")
            return 0
        _compress_ir(DEST)
        _install_vision_stub(DEST)
        if not _hop_gpu_ready(DEST):
            print(
                "compress finished but hop is still GPU-unsafe "
                f"(language_int8={_has_int8_weights(DEST)} "
                f"emb_int8={_text_embeddings_int8(DEST)} "
                f"vision_stub={_vision_is_stub(DEST)})",
                file=sys.stderr,
            )
            return 1
        for p in sorted(DEST.glob("*.bin")):
            print(f"ok   {p} ({p.stat().st_size} bytes)")
        print("done translategemma-4b-it-int8 compress")
        return 0

    if _ir_complete(DEST) and _hop_gpu_ready(DEST):
        sizes = [
            f"{p.name}={p.stat().st_size}"
            for p in sorted(DEST.glob("*.bin"))
        ]
        print("skip already GPU-ready INT8 (" + ", ".join(sizes) + ")")
        print("done translategemma-4b-it-int8")
        return 0
    if _ir_complete(DEST):
        print(
            "IR on disk is not GPU-ready (embeddings still FP and/or "
            "SigLIP still in the runtime folder). Compressing embeddings, "
            "installing the text-only vision stub."
        )
        _compress_ir(DEST)
        _install_vision_stub(DEST)
        if not _hop_gpu_ready(DEST):
            print("compress finished but hop is still GPU-unsafe", file=sys.stderr)
            return 1
        for p in sorted(DEST.glob("*.bin")):
            print(f"ok   {p} ({p.stat().st_size} bytes)")
        print("done translategemma-4b-it-int8 compress")
        return 0

    token: str | None = None
    if args.from_kaggle:
        source = str(_download_kaggle())
    elif args.local:
        local = Path(args.local).expanduser().resolve()
        if not _looks_like_hf_checkpoint(local):
            print(f"not a checkpoint (need config.json + *.safetensors): {local}", file=sys.stderr)
            return 2
        source = str(local)
    else:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token or _is_placeholder_token(token):
            print(
                "Hugging Face is gated. You already accepted Gemma on Kaggle — use that:\n"
                "  python scripts/download_translategemma.py --from-kaggle\n"
                "Or a real HF token after agreeing on huggingface.co/google/translategemma-4b-it",
                file=sys.stderr,
            )
            return 2
        source = HF_REPO

    _export(source, DEST, token)
    if _ir_complete(DEST) and not _hop_gpu_ready(DEST):
        _compress_ir(DEST)
        _install_vision_stub(DEST)
    if not _ir_complete(DEST):
        found = sorted(p.name for p in DEST.iterdir()) if DEST.is_dir() else []
        print(f"export finished but IR incomplete. files={found}", file=sys.stderr)
        return 1
    if not _hop_gpu_ready(DEST):
        print("export finished but hop is still GPU-unsafe", file=sys.stderr)
        return 1
    for p in sorted(DEST.glob("*.bin")):
        print(f"ok   {p} ({p.stat().st_size} bytes)")
    print("done translategemma-4b-it-int8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
