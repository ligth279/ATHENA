"""TranslateGemma hop helpers: official prompt + 75% context packing.

TranslateGemma 4B/12B/27B share a 2K *input* window. We never send more
than fill_ratio (default 0.75) of that, and we only cut after a sentence
so a chunk is still readable on its own. Used for both Event G directions:
student text → English, and Llama English → selected language.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import ceil
from typing import Callable

# Official model card: total input context of 2K tokens.
TRANSLATE_CONTEXT_TOKENS = 2048
TRANSLATE_FILL_RATIO = 0.75

CountTokens = Callable[[str], int]

_ABBREV = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "inc",
    "ltd",
    "st",
    "ave",
    "fig",
    "eq",
    "al",
    "approx",
    "dept",
    "est",
    "no",
}

# Period not in 3.14; keep following space with the sentence.
_TERMINATOR = re.compile(
    r"(?:"
    r"(?<!\d)[.!?…](?!\d)[\"'»”’)\]]*"
    r"|[।॥]+"
    r"|[。！？]+"
    r"|[؟۔]+"
    r")"
    r"(?:\s+|$)"
    r"|\n{2,}"
)

# English names the official TranslateGemma chat template expects.
_LANG_NAMES: dict[str, str] = {
    "ar": "Arabic",
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "cs": "Czech",
    "de": "German",
    "doi": "Dogri",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "gu": "Gujarati",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "kok": "Konkani",
    "ks": "Kashmiri",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "nl": "Dutch",
    "od": "Odia",
    "or": "Odia",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


@dataclass(frozen=True)
class TextChunk:
    text: str
    paragraph_before: bool = False


class ApproxTokenCounter:
    """Conservative stand-in until the IR tokenizer is on disk.

    2 characters/token over-counts English and is closer to Indic fertility,
    so packing stays under the real window.
    """

    def __init__(self, chars_per_token: float = 2.0) -> None:
        self.chars_per_token = float(chars_per_token)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(ceil(len(text) / self.chars_per_token)))


def language_name(code: str) -> str:
    raw = (code or "").strip()
    if not raw:
        return "Unknown"
    key = raw.replace("_", "-")
    lower = key.lower()
    if lower in _LANG_NAMES:
        return _LANG_NAMES[lower]
    prefix = lower.split("-", 1)[0]
    return _LANG_NAMES.get(prefix, raw)


def normalize_lang(code: str) -> str:
    raw = (code or "").strip().replace("_", "-")
    if not raw:
        return "en"
    parts = raw.split("-")
    if len(parts) == 1:
        return parts[0].lower()
    return f"{parts[0].lower()}-{parts[1].upper()}"


def format_translate_prompt(text: str, *, source_lang: str, target_lang: str) -> str:
    """Byte-stable copy of the official TranslateGemma user turn.

    OpenVINO GenAI's built-in Gemma 3 chat template rejects the structured
    TranslateGemma messages, so we send this string as a raw generate().
    """

    src = normalize_lang(source_lang)
    tgt = normalize_lang(target_lang)
    src_name = language_name(src)
    tgt_name = language_name(tgt)
    body = text.strip()
    return (
        f"<start_of_turn>user\n"
        f"You are a professional {src_name} ({src}) to {tgt_name} ({tgt}) "
        f"translator. Your goal is to accurately convey the meaning and "
        f"nuances of the original {src_name} text while adhering to "
        f"{tgt_name} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {tgt_name} translation, without any additional "
        f"explanations or commentary. Please translate the following "
        f"{src_name} text into {tgt_name}:\n\n\n"
        f"{body}"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def prompt_token_limit(
    max_input_tokens: int = TRANSLATE_CONTEXT_TOKENS,
    fill_ratio: float = TRANSLATE_FILL_RATIO,
) -> int:
    return max(32, int(max_input_tokens * fill_ratio))


def _is_abbrev(prefix: str) -> bool:
    m = re.search(r"([A-Za-z.]+)$", prefix.rstrip())
    if not m:
        return False
    word = m.group(1).rstrip(".").lower()
    if word in _ABBREV:
        return True
    # Initials: "A." / "U.S."
    return len(word.replace(".", "")) == 1


def split_sentences(text: str) -> list[str]:
    """Split on sentence enders. Pieces concatenate back to `text`."""

    if not text:
        return []
    pieces: list[str] = []
    last = 0
    for match in _TERMINATOR.finditer(text):
        if match.group(0).startswith("\n"):
            end = match.end()
        else:
            if _is_abbrev(text[last : match.start()]):
                continue
            end = match.end()
        piece = text[last:end]
        if piece:
            pieces.append(piece)
        last = end
    if last < len(text):
        pieces.append(text[last:])
    return pieces or [text]


def _fits(text: str, count: CountTokens, limit: int) -> bool:
    return count(text) <= limit


def _split_oversize(text: str, count: CountTokens, limit: int) -> list[str]:
    """Last resort when one sentence is itself over the budget."""

    for pattern in (r"(?<=[;；])\s*", r"(?<=[:：])\s*", r"(?<=[,，])\s*"):
        parts = _split_keep_sep(text, pattern)
        if len(parts) > 1 and all(_fits(p, count, limit) for p in parts if p.strip()):
            packed = _pack_parts(parts, count, limit)
            if packed:
                return packed
    words = re.findall(r"\S+\s*", text)
    if len(words) <= 1:
        return _cut_chars(text, count, limit)
    packed = _pack_parts(words, count, limit)
    return packed or _cut_chars(text, count, limit)


def _split_keep_sep(text: str, pattern: str) -> list[str]:
    bits = re.split(f"({pattern})", text)
    out: list[str] = []
    buf = ""
    for bit in bits:
        if re.fullmatch(pattern, bit or ""):
            buf += bit
            if buf:
                out.append(buf)
                buf = ""
        else:
            buf += bit
    if buf:
        out.append(buf)
    return out or [text]


def _pack_parts(parts: list[str], count: CountTokens, limit: int) -> list[str]:
    chunks: list[str] = []
    buf = ""
    for part in parts:
        trial = buf + part
        if not buf or _fits(trial, count, limit):
            buf = trial
            continue
        if buf:
            chunks.append(buf)
        if _fits(part, count, limit):
            buf = part
        else:
            chunks.extend(_cut_chars(part, count, limit))
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def _cut_chars(text: str, count: CountTokens, limit: int) -> list[str]:
    if _fits(text, count, limit):
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        lo, hi = start + 1, n
        best = start + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if _fits(text[start:mid], count, limit):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best <= start:
            best = min(start + 1, n)
        chunks.append(text[start:best])
        start = best
    return chunks


def pack_text(text: str, *, count: CountTokens, limit: int) -> list[TextChunk]:
    """Pack sentences so `count(chunk.text) <= limit`. Never drops characters."""

    if not text:
        return []
    if limit < 1:
        raise ValueError("translator chunk limit must be positive")
    sentences = split_sentences(text)
    out: list[TextChunk] = []
    buf = ""
    para_before = False

    def flush() -> None:
        nonlocal buf, para_before
        if not buf:
            return
        out.append(TextChunk(text=buf, paragraph_before=para_before))
        buf = ""
        para_before = False

    for sent in sentences:
        trial = buf + sent
        if buf and _fits(trial, count, limit):
            buf = trial
            continue
        if buf:
            ended_para = bool(re.search(r"\n\n\s*$", buf))
            flush()
            para_before = ended_para
        if _fits(sent, count, limit):
            buf = sent
            continue
        for piece in _split_oversize(sent, count, limit):
            if buf:
                flush()
            out.append(TextChunk(text=piece, paragraph_before=para_before))
            para_before = False
        para_before = False
    flush()
    return out or [TextChunk(text=text)]


def join_translations(chunks: list[TextChunk], translations: list[str]) -> str:
    parts: list[str] = []
    for chunk, translated in zip(chunks, translations):
        piece = translated.strip()
        if not piece:
            continue
        if not parts:
            parts.append(piece)
            continue
        if chunk.paragraph_before or "\n\n" in chunk.text[:4]:
            parts.append("\n\n")
        else:
            parts.append(" ")
        parts.append(piece)
    return "".join(parts)


def chunk_for_translator(
    text: str,
    *,
    source_lang: str,
    target_lang: str,
    count: CountTokens,
    max_input_tokens: int = TRANSLATE_CONTEXT_TOKENS,
    fill_ratio: float = TRANSLATE_FILL_RATIO,
) -> list[TextChunk]:
    """Pack so the *full* TranslateGemma prompt stays at ≤ fill_ratio of 2K."""

    limit = prompt_token_limit(max_input_tokens, fill_ratio)

    def prompt_count(chunk: str) -> int:
        return count(
            format_translate_prompt(
                chunk, source_lang=source_lang, target_lang=target_lang
            )
        )

    return pack_text(text, count=prompt_count, limit=limit)
