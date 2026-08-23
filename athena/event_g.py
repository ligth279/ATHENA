"""Event G sub-process combinator (Athena, not AMC / RBA).

Product: language / speech / speak are a momentary sub-state. Combine
buttons by adding hops, do not rerun the whole pipeline. AMC only runs
one exclusive hop at a time and kills STT / translator after pass-on.
Llama keeps the same-section lesson thread.

Connect a hop only when that flag requires it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from athena.amc.controller import AMC
from athena.amc.exceptions import TTSNotWiredError
from athena.amc.translate import normalize_lang
from athena.amc.types import Event


def _base_lang(code: str) -> str:
    return normalize_lang(code).split("-", 1)[0].lower()


def plan_hops(
    *,
    speech: bool,
    speak: bool,
    source_lang: str,
    target_lang: str,
) -> list[str]:
    """Which hops this button combo needs. Order is fixed; presence is not."""

    hops: list[str] = []
    if speech:
        hops.append("stt")
    if _base_lang(source_lang) != "en":
        hops.append("translate_in")
    hops.append("llama")
    if _base_lang(target_lang) != "en":
        hops.append("translate_out")
    if speak:
        hops.append("tts")
    return hops


@dataclass(frozen=True)
class GHop:
    name: str
    input_text: str
    output_text: str


@dataclass
class GResult:
    text: str
    english_question: str
    english_answer: str
    hops: list[GHop] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    tts_pending: bool = False


class EventG:
    """Mimic of the Event G sub-controller. Calls AMC public methods only."""

    def __init__(self, amc: AMC, section: Any | None = None) -> None:
        self._amc = amc
        self._section = section or {
            "id": "g-s1",
            "title": "Fractions",
            "body": (
                "A fraction a/b is a parts of a whole split into b equal pieces. "
                "The top number is the numerator, the bottom is the denominator."
            ),
        }

    def run(
        self,
        *,
        source_lang: str = "en",
        target_lang: str = "en",
        speech: bool = False,
        speak: bool = False,
        text: str | None = None,
        pcm: list[float] | None = None,
    ) -> GResult:
        src = normalize_lang(source_lang)
        tgt = normalize_lang(target_lang)
        plan = plan_hops(
            speech=speech, speak=speak, source_lang=src, target_lang=tgt
        )
        done: list[GHop] = []

        if speech:
            if pcm is None:
                raise ValueError("speech hop needs 16 kHz PCM")
            spoken = self._amc.transcribe(pcm)
            done.append(GHop("stt", "<pcm>", spoken))
            student = spoken
        else:
            student = (text or "").strip()
            if not student:
                raise ValueError("typed Event G needs text")

        if "translate_in" in plan:
            en_q = self._amc.translate(
                student, source_lang=src, target_lang="en"
            )
            done.append(GHop("translate_in", student, en_q))
        else:
            en_q = student

        if self._amc.status().event is not Event.T:
            self._amc.enter_event_t(self._section)
        en_a = self._amc.tutor_ask(en_q)
        done.append(GHop("llama", en_q, en_a))

        if "translate_out" in plan:
            out = self._amc.translate(en_a, source_lang="en", target_lang=tgt)
            done.append(GHop("translate_out", en_a, out))
        else:
            out = en_a

        tts_pending = False
        if "tts" in plan:
            try:
                self._amc.talk(out)
            except TTSNotWiredError:
                tts_pending = True
                done.append(GHop("tts", out, "<tts not wired>"))

        return GResult(
            text=out,
            english_question=en_q,
            english_answer=en_a,
            hops=done,
            plan=plan,
            tts_pending=tts_pending,
        )
