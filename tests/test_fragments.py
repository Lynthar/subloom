"""Which subtitles cut mid-sentence join the next one, and what the joined one carries."""

import pytest

from subloom.core.fragments import rejoin_fragments
from subloom.models.subtitle import SubtitleSegment, WordTiming


def _cues(*spans):
    """(start, end, text) triples as subtitles numbered from 1."""
    return [
        SubtitleSegment(index=i, start=s, end=e, text=t) for i, (s, e, t) in enumerate(spans, 1)
    ]


def _texts(segments):
    return [(s.index, s.text) for s in segments]


def test_a_fragment_joins_the_rest_of_its_sentence_and_the_rest_renumber():
    cues = _cues(
        (8.0, 9.0, "行こう。"),
        (10.0, 11.0, "見せ"),
        (11.2, 12.0, "てみろ。"),
        (14.0, 15.0, "うん。"),
    )

    joined = rejoin_fragments(cues, "ja")

    assert _texts(joined) == [(1, "行こう。"), (2, "見せてみろ。"), (3, "うん。")]
    assert (joined[1].start, joined[1].end) == (10.0, 12.0)


def test_latin_fragments_join_with_a_space_and_an_abbreviation_does_not_end_a_sentence():
    cues = _cues((0.0, 1.0, "I saw Mr."), (1.3, 2.0, "Smith"), (2.4, 3.0, "today."))

    assert _texts(rejoin_fragments(cues, "en")) == [(1, "I saw Mr. Smith today.")]


@pytest.mark.parametrize(
    "cues",
    [
        _cues((0.0, 1.0, "行こう。"), (1.2, 2.0, "うん")),
        _cues((0.0, 1.0, "見せ"), (2.1, 3.0, "てみろ。")),
        _cues((0.0, 4.0, "ずっと待って"), (4.5, 7.5, "たのに。")),
        _cues((0.0, 2.0, "あ" * 30), (2.2, 3.0, "いう。")),
    ],
    ids=["sentence ended", "pause over 1 s", "over 7 s", "over two lines"],
)
def test_it_stays_apart(cues):
    assert _texts(rejoin_fragments(cues, "ja")) == _texts(cues)


def test_latin_lines_hold_more_characters_than_cjk_ones():
    cues = _cues((0.0, 2.0, "a" * 40), (2.2, 3.0, "b" * 40 + "."))

    assert len(rejoin_fragments(cues, "en")) == 1
    assert len(rejoin_fragments(cues, "ja")) == 2


def test_the_joined_subtitle_keeps_words_confidence_and_the_later_end():
    first = SubtitleSegment(
        index=1,
        start=0.0,
        end=3.0,
        text="見せ",
        words=[WordTiming("見せ", 0.0, 0.8)],
        confidence=0.9,
    )
    inside = SubtitleSegment(
        index=2,
        start=0.5,
        end=2.0,
        text="て。",
        words=[WordTiming("て。", 1.0, 2.0)],
        confidence=0.6,
    )
    unaligned = SubtitleSegment(index=2, start=0.5, end=2.0, text="て。")

    (joined,) = rejoin_fragments([first, inside], "ja")
    (partly_aligned,) = rejoin_fragments([first, unaligned], "ja")

    assert (joined.end, joined.confidence) == (3.0, 0.6)
    assert [w.word for w in joined.words] == ["見せ", "て。"]
    assert partly_aligned.words is None
