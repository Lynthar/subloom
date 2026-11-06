"""Which subtitles count as interjection-only, and what dropping them leaves."""

import pytest

from subloom.core.interjections import is_interjection, without_interjections
from subloom.models.subtitle import SubtitleSegment, WordTiming


@pytest.mark.parametrize(
    "text",
    [
        "あ",
        "ああああ。",
        "うん、うん、うん。",
        "あ、うん、",
        "アッ",
        "ｱｯ",
        "はぁ…",
        "うーん",
        "えっ？",
        "うわあ！",
        "ふふふ",
        "嗯嗯",
        "啊？",
        "ああ、嗯",
        "Uh, um.",
        "Hmm...",
        "UHHH",
        "uh-huh",
        "Hahaha!",
        "음...",
        "아아",
    ],
)
def test_sounds_alone_are_interjections(text):
    assert is_interjection(text)


@pytest.mark.parametrize(
    "text",
    [
        "ああ、そうだね。",
        "はい。",
        "ほら、",
        "ねえ",
        "いや",
        "好啊",
        "Oh no",
        "Ahead",
        "Um, okay.",
        "I'm",
        "아니",
        "あ、3",
        "…",
        "♪",
        "",
    ],
)
def test_a_word_or_no_letters_at_all_is_not(text):
    assert not is_interjection(text)


def test_dropping_keeps_the_rest_whole_and_renumbers_it():
    words = [WordTiming("そうだね", 2.0, 2.8)]
    segments = [
        SubtitleSegment(index=1, start=0.0, end=1.0, text="ああ。"),
        SubtitleSegment(
            index=2, start=2.0, end=2.8, text="そうだね。", words=words, confidence=0.7
        ),
        SubtitleSegment(index=3, start=3.0, end=3.5, text="うん"),
        SubtitleSegment(index=4, start=4.0, end=5.0, text="行こう。"),
    ]

    kept = without_interjections(segments)

    assert [(s.index, s.text) for s in kept] == [(1, "そうだね。"), (2, "行こう。")]
    assert (kept[0].start, kept[0].end, kept[0].words, kept[0].confidence) == (2.0, 2.8, words, 0.7)
