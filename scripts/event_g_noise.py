#!/usr/bin/env python3
"""Measure stacked *channel* noise on Event G (mock AMC).

STT and translator are lossy pipes (word slips). Llama is the tutor: it
is supposed to change the text (answer the question). This script does
**not** scramble Llama — that would fake “Llama WER” against the question.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from athena.event_g import plan_hops
from athena.noise import inject_word_noise, word_error_rate

GOLD = (
    "What is a fraction of a whole. "
    "Name the numerator and the denominator."
)
RATE = 0.12
TRIALS = 40


_CHANNEL = {"stt", "translate_in", "translate_out"}


def _combo(name: str, **flags: object) -> tuple[str, list[str], list[str]]:
    plan = plan_hops(
        speech=bool(flags.get("speech", False)),
        speak=bool(flags.get("speak", False)),
        source_lang=str(flags.get("source_lang", "en")),
        target_lang=str(flags.get("target_lang", "en")),
    )
    channel = [h for h in plan if h in _CHANNEL]
    return name, plan, channel


def _run(channel: list[str], rng: random.Random) -> str:
    text = GOLD
    for _hop in channel:
        text = inject_word_noise(text, RATE, rng)
    return text


def main() -> int:
    combos = [
        _combo("A  en type → en out (llama only)", source_lang="en", target_lang="en"),
        _combo("B  en type → hi out (+translate_out)", source_lang="en", target_lang="hi"),
        _combo("C  hi type → hi out (in+llama+out)", source_lang="hi", target_lang="hi"),
        _combo("D  hi speech → hi out (stt+in+llama+out)", speech=True, source_lang="hi", target_lang="hi"),
        _combo(
            "E  hi speech+speak (tts not counted)",
            speech=True,
            speak=True,
            source_lang="hi",
            target_lang="hi",
        ),
    ]
    print(f"gold: {GOLD}")
    print(f"channel slip {RATE:.0%} per STT/translate hop, {TRIALS} trials")
    print("Llama is not a channel; it is not noised here.")
    print(f"{'path':48} ch  mean WER")
    for name, _plan, channel in combos:
        wers: list[float] = []
        for i in range(TRIALS):
            hyp = _run(channel, random.Random(1 + i))
            wers.append(word_error_rate(GOLD, hyp))
        mean = sum(wers) / len(wers)
        print(f"{name:48} {len(channel):2d}  {mean:7.1%}")
    print("TTS is not wired; speak adds no extra text noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
