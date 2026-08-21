from __future__ import annotations

from collections import defaultdict
from typing import Any

from athena.amc.types import EventHandler


class EventBus:
    """Tiny in-process bus. Athena (later) subscribes to AMC lifecycle."""

    def __init__(self) -> None:
        self._subs: dict[str, list[EventHandler]] = defaultdict(list)

    def on(self, name: str, handler: EventHandler) -> None:
        self._subs[name].append(handler)

    def emit(self, name: str, **payload: Any) -> None:
        for handler in list(self._subs.get(name, ())):
            handler(**payload)
