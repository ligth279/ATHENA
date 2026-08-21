from __future__ import annotations

from athena.amc.types import Turn


class ConversationMemory:
    """Text-side session memory for llama roles.

    GPU KV cache is *not* this object. KV dies on unload; this history is
    what we replay into the system prompt after a Whisper round-trip.
    """

    def __init__(self, keep_pairs: int = 3) -> None:
        self.keep_pairs = keep_pairs
        self._turns: list[Turn] = []

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def turns(self) -> tuple[Turn, ...]:
        return tuple(self._turns)

    def add_user(self, text: str) -> None:
        self._turns.append(Turn(role="user", text=text))

    def add_assistant(self, text: str) -> None:
        self._turns.append(Turn(role="assistant", text=text))

    def last_pairs(self, n: int | None = None) -> list[Turn]:
        n = self.keep_pairs if n is None else n
        # a pair is user+assistant, so 3 pairs ≈ 6 turns
        return list(self._turns[-(n * 2) :])

    def render(self) -> str:
        if not self._turns:
            return ""
        lines = []
        for turn in self._turns:
            label = "Student" if turn.role == "user" else "Tutor"
            lines.append(f"{label}: {turn.text}")
        return "\n".join(lines)

    def snapshot_for_dc(self) -> list[Turn]:
        return self.last_pairs(self.keep_pairs)

    def clear(self) -> list[Turn]:
        snapshot = self.snapshot_for_dc()
        self._turns.clear()
        return snapshot
