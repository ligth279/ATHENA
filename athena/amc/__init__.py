"""AI Model Controller (AMC) — section 2.

Owns exclusive GPU residency for Llama 3.1 and Whisper on Intel Arc.
Only one model is resident at a time. Role/session memory lives here;
KV cache is dropped on unload and rebuilt from text history.

Maintainer map (how this module works, where to change it): docs/amc.md
Product logic: 2333.txt section 2.
"""

from athena.amc.config import AMCConfig
from athena.amc.controller import AMC
from athena.amc.types import Event, LlamaRole, ModelId

__all__ = ["AMC", "AMCConfig", "Event", "LlamaRole", "ModelId"]
