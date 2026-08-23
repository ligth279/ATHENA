class AMCError(Exception):
    """Base error for the AI Model Controller."""


class DeviceError(AMCError):
    """Intel GPU / OpenVINO device is missing or unusable."""


class ModelPathError(AMCError):
    """Converted OpenVINO IR model is not on disk."""


class RoleError(AMCError):
    """Requested llama role/event is not active."""


class TTSNotWiredError(AMCError):
    """Talk/TTS is not Whisper. Whisper is speech-to-text only."""


class GpuDeadError(AMCError):
    """Intel GPU OpenCL context is poisoned (-5 / -14). Stop this process."""
