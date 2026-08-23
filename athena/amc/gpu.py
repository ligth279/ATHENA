from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from athena.amc.config import AMCConfig
from athena.amc.exceptions import DeviceError, ModelPathError


def list_openvino_devices() -> list[str]:
    try:
        import openvino as ov
    except ImportError:
        return []
    core = ov.Core()
    try:
        return list(core.available_devices)
    except Exception:
        return list(core.get_available_devices())


def resolve_device(config: AMCConfig) -> str:
    """Pick the inference device.

    Default is GPU (Arc B580). CPU fallback is off unless the admin
    sets allow_cpu_fallback — silent CPU would hide a driver problem.
    """

    wanted = config.device.upper()
    devices = list_openvino_devices()
    if not devices:
        if config.backend == "mock":
            return wanted
        # OpenVINO not installed yet — keep the configured name so load()
        # is the one that fails with a useful import error.
        return wanted
    if wanted in devices:
        return wanted
    if config.allow_cpu_fallback and "CPU" in devices:
        return "CPU"
    raise DeviceError(
        f"OpenVINO device {wanted!r} not found. Available: {devices}. "
        "Arc B580 should appear as GPU once Intel GPU drivers / oneAPI "
        "Level Zero are working. CPU fallback is disabled."
    )


def compile_cache_kwargs(
    config: AMCConfig, device: str, subdir: str = ""
) -> dict[str, str]:
    if device.upper() != "GPU":
        return {}
    cache = str(Path(config.cache_dir) / subdir) if subdir else config.cache_dir
    return {"CACHE_DIR": cache}


# Arc B580 is 12 GB and drives the displays. Do not treat 12.0 as free.
B580_VRAM_GB = 12.0
DISPLAY_RESERVE_GB = 1.5
GENERATE_SCRATCH_GB = 1.5
# Must match AMCConfig.kv_cache_gb_translate. Uncapped VLM KV was the
# generate -5 on a 10-token sentence after INT8 weights already fitted.
TRANSLATE_KV_RESERVE_GB = 1.0

_GPU_DEAD_MARKERS = (
    "CL_OUT_OF_RESOURCES",
    "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
    "CL_INVALID_EVENT",
    "OUT_OF_RESOURCES",
)


def is_gpu_dead_error(exc: BaseException) -> bool:
    """True if OpenCL reported OOM or a poisoned wait list."""

    text = str(exc)
    return any(m in text for m in _GPU_DEAD_MARKERS)


def _xml_has_int8(xml: Path) -> bool:
    if not xml.is_file() or xml.stat().st_size == 0:
        return False
    text = xml.read_text(errors="ignore")
    return 'precision="I8"' in text or 'precision="U8"' in text


def translator_weight_gb(model_dir: Path) -> float:
    total = 0
    for p in model_dir.glob("openvino_*_model.bin"):
        total += p.stat().st_size
    return total / (1024**3)


def translator_unsafe_reason(model_dir: Path | str) -> str | None:
    """Why loading this IR on the B580 would likely hit -5 and poison OpenCL.

    None means the hop is small enough to try. This is the *first* cause
    of dead-state: generate OOM, then every later compile is -14.
    """

    root = Path(model_dir)
    if not root.is_dir():
        return f"translator IR missing: {root}"
    vision = root / "openvino_vision_embeddings_model.bin"
    vision_xml = root / "openvino_vision_embeddings_model.xml"
    if not vision_xml.is_file():
        return (
            "vision IR missing. VLMPipeline still opens "
            "openvino_vision_embeddings_model.xml for a text hop. "
            "Run: python scripts/download_translategemma.py --compress-only"
        )
    # Actual culprit: generate runs SigLIP (896², ~0.4 GB INT8 still) and
    # hits -5 on a short sentence. The text-only stub is ~1.3 MB.
    if vision.is_file() and vision.stat().st_size > 10_000_000:
        return (
            "SigLIP vision encoder is still in the runtime folder "
            f"({vision.stat().st_size} bytes). Event G is text-only; "
            "generate of that graph OOMs the B580. Run: "
            "python scripts/download_translategemma.py --compress-only"
        )
    text_xml = root / "openvino_text_embeddings_model.xml"
    if text_xml.is_file() and not _xml_has_int8(text_xml):
        return (
            "text embeddings are not INT8 (still ~2.6 GB FP32). "
            "Optimum skipped them (<1B params). Run: "
            "python scripts/download_translategemma.py --compress-only"
        )
    gb = translator_weight_gb(root)
    need = (
        gb
        + DISPLAY_RESERVE_GB
        + GENERATE_SCRATCH_GB
        + TRANSLATE_KV_RESERVE_GB
    )
    if need > B580_VRAM_GB:
        return (
            f"translator weights {gb:.1f} GiB + display/scratch/KV "
            f"{DISPLAY_RESERVE_GB + GENERATE_SCRATCH_GB + TRANSLATE_KV_RESERVE_GB:.1f} GiB "
            f"exceed the {B580_VRAM_GB:.0f} GB B580. Compress embeddings first."
        )
    return None


def require_translator_gpu_safe(model_dir: Path | str) -> None:
    reason = translator_unsafe_reason(model_dir)
    if reason:
        raise ModelPathError(
            "Refusing to load TranslateGemma on GPU: " + reason + " "
            "Loading it anyway caused CL_OUT_OF_RESOURCES (-5) on generate, "
            "which poisons the OpenCL wait list so the next hop (Llama T) "
            "fails with WAIT_LIST (-14)."
        )


def gpu_dead_message(exc: BaseException) -> str:
    return (
        "Intel GPU OpenCL context is dead after: "
        f"{exc}. "
        "A -5 OUT_OF_RESOURCES (or the -14 WAIT_LIST / -58 INVALID_EVENT "
        "that follows) is not recoverable in this process. Exit Python. "
        "Do not load another model on this GPU in this process. "
        "Reboot only recovers a reset Xe CCS engine; it does not stop the "
        "next generate from OOM'ing. Translator KV must stay capped."
    )


def drop_pipeline(pipe: Any) -> None:
    """Drop an OpenVINO GenAI pipeline so the C++ destructor can free GPU.

    Measured: VLM ~4.25 GiB usm_device returns to 0 when this is the last
    Python ref. Keeping any extra name (`pipe = self._pipe` across unload)
    holds the 4.25 GiB. Does not unpoison OpenCL after a -5; that dies
    with the process.
    """

    del pipe
    gc.collect()
    gc.collect()
