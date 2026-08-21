from __future__ import annotations

from pathlib import Path

from athena.amc.config import AMCConfig
from athena.amc.exceptions import DeviceError


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
