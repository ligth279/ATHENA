"""Byte-level BPE from HuggingFace tokenizer.json. Stdlib only.

OmniVoice ships tokenizer.json (Qwen3). Runtime AMC cannot import the
HuggingFace ``tokenizers`` package (not in requirements-amc.txt).
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_BYTE_ENC = _bytes_to_unicode()


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_number(ch: str) -> bool:
    return unicodedata.category(ch).startswith("N")


def _qwen_split(text: str) -> list[str]:
    """Isolated split matching Qwen3 tokenizer.json pre_tokenizer regex."""

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        # (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if ch == "'":
            low = text[i:].lower()
            hit = next((c for c in _CONTRACTIONS if low.startswith(c)), None)
            if hit:
                out.append(text[i : i + len(hit)])
                i += len(hit)
                continue
        # [^\r\n\p{L}\p{N}]?\p{L}+
        extra = 0
        if ch not in "\r\n" and not _is_letter(ch) and not _is_number(ch) and i + 1 < n and _is_letter(text[i + 1]):
            extra = 1
        if extra or _is_letter(ch):
            j = i + extra
            while j < n and _is_letter(text[j]):
                j += 1
            if j > i + extra:
                out.append(text[i:j])
                i = j
                continue
        # \p{N}
        if _is_number(ch):
            out.append(ch)
            i += 1
            continue
        #  ?[^\s\p{L}\p{N}]+[\r\n]*
        j = i
        if ch == " ":
            j = i + 1
        k = j
        while k < n:
            c = text[k]
            if c.isspace() or _is_letter(c) or _is_number(c):
                break
            k += 1
        if k > j:
            while k < n and text[k] in "\r\n":
                k += 1
            out.append(text[i:k])
            i = k
            continue
        # \s*[\r\n]+
        j = i
        while j < n and text[j] in " \t\f\v":
            j += 1
        if j < n and text[j] in "\r\n":
            while j < n and text[j] in "\r\n":
                j += 1
            out.append(text[i:j])
            i = j
            continue
        # \s+(?!\S) (backtracking leaves one space before a word) or \s+
        if ch.isspace():
            j = i + 1
            while j < n and text[j].isspace() and text[j] not in "\r\n":
                j += 1
            if j < n and not text[j].isspace() and (j - i) >= 2:
                j -= 1
            out.append(text[i:j])
            i = j
            continue
        out.append(ch)
        i += 1
    return out


def _bpe(piece: str, ranks: dict[tuple[str, str], int]) -> tuple[str, ...]:
    word = tuple(piece)
    if len(word) < 2:
        return word
    pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
    while pairs:
        bigram = min(pairs, key=lambda p: ranks.get(p, 1 << 30))
        if bigram not in ranks:
            break
        first, second = bigram
        new: list[str] = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
            except ValueError:
                new.extend(word[i:])
                break
            new.extend(word[i:j])
            if j < len(word) - 1 and word[j + 1] == second:
                new.append(first + second)
                i = j + 2
            else:
                new.append(word[j])
                i = j + 1
        word = tuple(new)
        if len(word) == 1:
            break
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
    return word


class JsonBpeTokenizer:
    """Qwen3 tokenizer.json encode(). No HuggingFace tokenizers import."""

    def __init__(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._vocab: dict[str, int] = dict(data["model"]["vocab"])
        ranks: dict[tuple[str, str], int] = {}
        for i, merge in enumerate(data["model"]["merges"]):
            if isinstance(merge, (list, tuple)) and len(merge) == 2:
                ranks[(str(merge[0]), str(merge[1]))] = i
            else:
                left, right = str(merge).split(" ", 1)
                ranks[(left, right)] = i
        self._ranks = ranks
        special: dict[str, int] = {}
        for tok in data.get("added_tokens") or []:
            content = str(tok["content"])
            special[content] = int(tok["id"])
            self._vocab.setdefault(content, int(tok["id"]))
        self._special = special
        self._special_sorted = sorted(special, key=len, reverse=True)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        text = unicodedata.normalize("NFC", text)
        ids: list[int] = []
        for chunk, is_special in self._split_special(text):
            if is_special:
                ids.append(self._special[chunk])
                continue
            for piece in _qwen_split(chunk):
                mapped = "".join(_BYTE_ENC[b] for b in piece.encode("utf-8"))
                for part in _bpe(mapped, self._ranks):
                    ids.append(self._vocab[part])
        return ids

    def _split_special(self, text: str) -> list[tuple[str, bool]]:
        if not text or not self._special_sorted:
            return [(text, False)] if text else []
        out: list[tuple[str, bool]] = []
        i = 0
        n = len(text)
        while i < n:
            hit = next((s for s in self._special_sorted if text.startswith(s, i)), None)
            if hit is None:
                j = i + 1
                while j < n and not any(text.startswith(s, j) for s in self._special_sorted):
                    j += 1
                out.append((text[i:j], False))
                i = j
            else:
                out.append((hit, True))
                i += len(hit)
        return out
