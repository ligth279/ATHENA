"""Tiny word-error helpers for Event G hop-noise experiments."""

from __future__ import annotations

import random

# Classroom / STT-style near misses. Not a full homophone lexicon.
_HOMOPHONE = {
    "whole": "hole",
    "hole": "whole",
    "two": "too",
    "too": "two",
    "to": "too",
    "right": "write",
    "write": "right",
    "sum": "some",
    "some": "sum",
    "one": "won",
    "won": "one",
    "four": "for",
    "for": "four",
    "eight": "ate",
    "ate": "eight",
    "know": "no",
    "no": "know",
    "numerator": "numerater",
    "denominator": "denominater",
    "fraction": "faction",
}


def inject_word_noise(text: str, rate: float, rng: random.Random) -> str:
    """Corrupt `rate` of alphabetic tokens (STT / MT slip simulation)."""

    if rate <= 0 or not text:
        return text
    words = text.split()
    out: list[str] = []
    for word in words:
        core = word.strip(".,;:?!\"'")
        if core.isalpha() and rng.random() < rate:
            key = core.lower()
            if key in _HOMOPHONE:
                repl = _HOMOPHONE[key]
                if core[0].isupper():
                    repl = repl.capitalize()
                word = word.replace(core, repl, 1)
            elif len(core) > 3:
                i = rng.randrange(len(core) - 1)
                chars = list(core)
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                word = word.replace(core, "".join(chars), 1)
        out.append(word)
    return " ".join(out)


def word_error_rate(ref: str, hyp: str) -> float:
    """Token Levenshtein / |ref|. 0 = identical, 1+ = fully wrong."""

    a = ref.split()
    b = hyp.split()
    if not a:
        return 0.0 if not b else 1.0
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i, wa in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, wb in enumerate(b, 1):
            cur = dp[j]
            cost = 0 if wa.lower() == wb.lower() else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m] / n
