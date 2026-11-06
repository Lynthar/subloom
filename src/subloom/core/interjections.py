"""Subtitles that hold nothing but interjections: "ああ、うん", "嗯嗯", "Uh, um."."""

import logging
import re
import unicodedata
from dataclasses import replace
from typing import List

from ..models.subtitle import SubtitleSegment

logger = logging.getLogger(__name__)

# Characters that voice a sound but say nothing, one table for every language. Words and
# particles (はい, ね, ほら, 好) are left out, so a subtitle holding one is kept.
_SOUND_CHARS = frozenset(
    "あうえおんはひふへほわ"  # Japanese, compared after katakana and small kana fold to hiragana
    "嗯啊哦噢喔呃唔哎唉呀哈呵诶欸"
    "아어오우으음응흠하허헉와"
)
# Latin-script sounds, compared after a run of one letter collapses ("uhhh" -> "uh").
_SOUND_WORDS = re.compile(r"uh|uhm|um|er|erm|ah|oh|eh|hm|m|mhm|huh|(?:ha)+")
_SMALL_KANA = str.maketrans("ぁぃぅぇぉゎ", "あいうえおわ")
_KATAKANA = re.compile("[ァ-ヶ]")
# Sokuon and long-vowel marks only stretch a sound; dropping them leaves its letters.
_STRETCH = re.compile("[っー]")
_REPEATED_LETTER = re.compile(r"(.)\1+")


def is_interjection(text: str) -> bool:
    """Whether every word of ``text`` is an interjection; punctuation alone is not one."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _KATAKANA.sub(lambda m: chr(ord(m.group()) - 0x60), text)
    text = _STRETCH.sub("", text.translate(_SMALL_KANA))
    words = re.findall(r"[^\W_]+", text)
    return bool(words) and all(
        _SOUND_WORDS.fullmatch(_REPEATED_LETTER.sub(r"\1", word))
        if word.isascii()
        else _SOUND_CHARS.issuperset(word)
        for word in words
    )


def without_interjections(segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
    """The segments that are not interjection-only, renumbered from 1."""
    kept = [seg for seg in segments if not is_interjection(seg.text)]
    logger.info("Dropped %d interjection-only subtitles", len(segments) - len(kept))
    return [replace(seg, index=i) for i, seg in enumerate(kept, 1)]
