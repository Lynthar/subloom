"""The SRT structure gate: `check_structure` itself, and that minimal / full timing never trips it.

The property test feeds seeded random ASR-shaped input, degenerate cases included."""

import random

import pytest

from subtitle_forge.core.subtitle import check_structure, check_translation
from subtitle_forge.core.timestamp_processor import TimestampProcessor
from subtitle_forge.models.config import TimestampConfig
from subtitle_forge.models.subtitle import SubtitleSegment, WordTiming


def _seg(index, start, end, text="x", words=None):
    return SubtitleSegment(index=index, start=start, end=end, text=text, words=words)


def test_sound_segments_pass():
    segs = [_seg(1, 0.0, 1.0), _seg(2, 1.0, 2.5), _seg(3, 3.0, 4.0)]
    assert check_structure(segs, audio_duration=4.0) == []
    assert check_translation(segs, segs) == []


@pytest.mark.parametrize(
    "segs, expected",
    [
        ([_seg(1, 0.0, 1.0), _seg(2, 0.9, 2.0)], "#2: overlaps previous by 100 ms"),
        ([_seg(1, 1.0, 1.0)], "#1: non-positive duration"),
        ([_seg(1, 2.0, 1.5)], "#1: non-positive duration"),
        ([_seg(1, -0.5, 1.0)], "#1: starts before 0"),
        ([_seg(1, 0.0, 1.0), _seg(1, 2.0, 3.0)], "#1: duplicate index"),
        ([_seg(1, 9.0, 10.2)], "#1: ends past the audio"),
    ],
)
def test_each_violation_is_reported(segs, expected):
    assert any(v.startswith(expected) for v in check_structure(segs, audio_duration=10.0))


def test_times_are_judged_at_the_millisecond_the_file_stores():
    # Both land on 00:00:01,000 once written, so the file has no overlap.
    segs = [_seg(1, 0.0, 1.0004), _seg(2, 1.0002, 2.0)]
    assert check_structure(segs) == []


def test_a_cue_clamped_to_the_audio_end_is_not_past_it():
    # 9186.666687 s of audio: the clamped end and the audio end both round to 9186667 ms.
    duration = 146986667 / 16000
    assert check_structure([_seg(1, 9186.5, duration)], audio_duration=duration) == []
    assert check_structure([_seg(1, 9186.5, duration + 0.001)], audio_duration=duration) != []


def test_unknown_audio_duration_skips_the_end_check():
    assert check_structure([_seg(1, 0.0, 99.0)], audio_duration=0.0) == []


def test_translation_must_keep_count_and_indices():
    source = [_seg(1, 0.0, 1.0), _seg(2, 2.0, 3.0)]
    assert check_translation(source[:1], source) == ["1 segments out for 2 in"]
    renumbered = [_seg(1, 0.0, 1.0), _seg(3, 2.0, 3.0)]
    assert check_translation(renumbered, source) == ["segment indices differ from the source's"]


_TEXTS = [
    "Hi.",
    "Okay",
    "Go. Run. Hide.",
    "Mr. Smith arrived. He waited.",
    "今日は。いい天気です。",
]


def _random_asr_output(rng, duration):
    """Chronological segments with zero-length, coincident, overlapping and past-the-end cases."""
    segs = []
    t = rng.uniform(0, 1.0)
    while t < duration:
        start = max(segs[-1].start if segs else 0.0, t - rng.choice([0, 0, 0, 0.05, 0.3, 1.0]))
        end = start + rng.choice([0.0, 0.02, 0.1, 0.3, 1.0, 2.5, 9.0, 15.0]) * rng.uniform(0.5, 1.5)
        if rng.random() < 0.05:
            end = duration + rng.uniform(0, 0.5)
        text = rng.choice(_TEXTS)
        words = None
        if rng.random() < 0.7 and end > start:
            tokens = text.split()
            step = (end - start) / len(tokens)
            words = [
                WordTiming(w, start + k * step, start + (k + 1) * step)
                for k, w in enumerate(tokens)
            ]
        segs.append(_seg(len(segs) + 1, start, end, text, words))
        t = end + rng.choice([0.0, 0.01, 0.05, 0.2, 1.0, 5.0])
    return segs


@pytest.mark.parametrize("mode", ["minimal", "full"])
def test_minimal_and_full_output_is_always_structurally_sound(mode):
    for seed in range(400):
        rng = random.Random(seed)
        duration = rng.choice([5.0, 30.0, 120.0])
        config = TimestampConfig(
            mode=mode,
            split_sentences=rng.random() < 0.5,
            min_duration=rng.choice([0.5, 1.0, 2.0]),
            min_gap=rng.choice([0.0, 0.05, 0.2]),
            lead_in_ms=rng.choice([0, 80, 300]),
            linger_ms=rng.choice([0, 300, 800]),
        )
        processor = TimestampProcessor(config, language=rng.choice(["en", "ja"]))
        known = duration if rng.random() < 0.8 else None
        out = processor.process(_random_asr_output(rng, duration), known)
        assert check_structure(out, known) == [], (seed, config)


def test_colliding_segments_merge_instead_of_overlapping():
    # Stretched to min_duration, "Okay" would cover the next line, and with the next line only
    # 0.1s after its start it cannot be cut back to a 0.1s cue either: the two become one.
    words_a = [WordTiming("Okay", 6.0, 6.05)]
    words_b = [WordTiming("Wait!", 6.1, 7.0), WordTiming("Now.", 7.0, 8.44)]
    segs = [_seg(1, 6.0, 6.05, "Okay", words_a), _seg(2, 6.1, 8.44, "Wait! Now.", words_b)]
    config = TimestampConfig(mode="minimal", split_sentences=False, lead_in_ms=0, linger_ms=0)
    out = TimestampProcessor(config).process(segs, audio_duration=60.0)
    assert [(s.index, s.start, s.end, s.text) for s in out] == [(1, 6.0, 8.44, "Okay Wait! Now.")]
    assert out[0].words == words_a + words_b
