"""Regression tests for TimestampProcessor.

These lock in fixes for three timing bugs that shipped in the default paths:
  * minimal mode emitted overlapping subtitles (fast dialogue)
  * "off" mode truncated a segment below its acoustic end via a neighbour's lead-in
  * sentence splitting without word timestamps overflowed the parent span
    (and produced negative-duration segments)

The module is pure/functional and has no heavy dependencies, so these run
without ffmpeg / torch / whisper.
"""

from subloom.core.timestamp_processor import (
    TimestampProcessor,
    display_cells,
    is_sentence_end,
)
from subloom.models.config import TimestampConfig
from subloom.models.subtitle import SubtitleSegment, WordTiming


def _proc(**timestamp_fields):
    return TimestampProcessor(TimestampConfig(**timestamp_fields))


def _seg(index, start, end, text="x", words=True):
    w = [WordTiming(text, start, end)] if words else None
    return SubtitleSegment(index=index, start=start, end=end, text=text, words=w)


def _overlaps(segs):
    return [
        (segs[i - 1].index, segs[i].index)
        for i in range(1, len(segs))
        if segs[i].start < segs[i - 1].end - 1e-9
    ]


def test_minimal_mode_no_overlaps_in_fast_dialogue():
    # Three short utterances ~0.15s apart — min_duration extension used to push
    # each one over the next, and nothing fixed it afterwards.
    segs = [_seg(1, 10.00, 10.15), _seg(2, 10.30, 10.45), _seg(3, 10.60, 10.75)]
    out = _proc(mode="minimal", min_duration=1.0).process(segs, audio_duration=60.0)
    assert _overlaps(out) == []


def test_minimal_mode_no_overlaps_without_audio_duration():
    segs = [_seg(1, 10.00, 10.15), _seg(2, 10.30, 10.45), _seg(3, 10.60, 10.75)]
    out = _proc(mode="minimal", min_duration=1.0).process(segs, audio_duration=None)
    assert _overlaps(out) == []


def test_off_mode_never_truncates_below_acoustic_end():
    # Two adjacent 50ms utterances. In "off" mode the second one's lead-in must
    # not pull the first one's end earlier than where speech actually ended.
    segs = [_seg(1, 9.920, 9.970), _seg(2, 10.000, 10.050)]
    out = _proc(mode="off", min_duration=1.0).process(segs, audio_duration=60.0)
    assert out[0].end >= 9.970 - 1e-9
    assert _overlaps(out) == []


def test_all_modes_produce_ordered_nonnegative_durations():
    segs = [_seg(1, 10.00, 10.15), _seg(2, 10.30, 10.45), _seg(3, 10.60, 10.75)]
    for mode in ("off", "minimal", "full"):
        out = _proc(mode=mode, min_duration=1.0).process(
            [_seg(s.index, s.start, s.end) for s in segs], audio_duration=60.0
        )
        assert all(s.end >= s.start for s in out), mode
        assert _overlaps(out) == [], mode


def test_proportional_split_stays_within_parent_span():
    # No word timestamps -> proportional fallback. Must never exceed seg.end or
    # go negative (the old code inflated each piece to min_duration and blew past
    # the parent, forcing a negative-duration final piece).
    proc = _proc(mode="off", min_duration=1.0)
    parent = SubtitleSegment(1, 5.0, 8.0, "Go. Run. Hide. Wait. Stop. Now.", words=None)
    pieces = proc._split_segment_proportionally(parent)
    assert len(pieces) > 1
    assert all(p.end >= p.start for p in pieces)
    assert all(p.end <= 8.0 + 1e-9 for p in pieces)
    assert pieces[0].start == 5.0
    assert abs(pieces[-1].end - 8.0) < 1e-9


def test_split_sentences_full_pipeline_no_negative_durations():
    proc = _proc(mode="minimal", min_duration=1.0, split_sentences=True)
    parent = SubtitleSegment(1, 5.0, 8.0, "Go. Run. Hide. Wait. Stop. Now.", words=None)
    out = proc.process([parent], audio_duration=60.0)
    assert all(s.end >= s.start for s in out)


def test_word_timestamps_preserved_through_minimal():
    segs = [_seg(1, 1.0, 2.0), _seg(2, 5.0, 6.0)]
    out = _proc(mode="minimal").process(segs, audio_duration=60.0)
    assert all(s.has_word_timestamps() for s in out)


def test_full_mode_does_not_truncate_aligned_overlong_segment():
    # 12s of aligned speech (slow, emphatic delivery). Full mode used to cap it
    # to reading time (~1s for "No") and drop the word timestamps with it —
    # leaving the rest of the utterance with no subtitle at all.
    seg = _seg(1, 0.0, 12.0, text="No")
    out = _proc(mode="full", max_duration=8.0).process([seg], audio_duration=20.0)
    assert out[0].end >= 12.0 - 1e-9
    assert out[0].has_word_timestamps()


def test_full_mode_still_caps_unaligned_overlong_segment():
    # Without word timing the stretched end-time is a transcriber artifact —
    # capping those is the pass's actual job.
    seg = _seg(1, 0.0, 60.0, text="short", words=False)
    out = _proc(mode="full", max_duration=8.0).process([seg], audio_duration=120.0)
    assert out[0].end - out[0].start <= 8.0 + 1e-9


def test_lead_in_linger_stays_inside_audio_at_tail():
    # Overlapping input right at the end of the audio (raw ASR output in "off"
    # mode). The neighbour clamp used to push the second start past the audio
    # end, and the safety net then re-extended its end to 11.05s in a 10s file.
    segs = [_seg(1, 9.4, 10.0), _seg(2, 9.9, 10.0)]
    out = _proc(mode="off", min_duration=1.0).process(segs, audio_duration=10.0)
    assert all(s.start <= 10.0 + 1e-9 for s in out)
    assert all(s.end <= 10.0 + 1e-9 for s in out)
    assert all(s.end > s.start for s in out)


def test_off_mode_sentence_split_does_not_manufacture_overlap():
    # Two sentences in one aligned segment, the first very short: the
    # min-readable extension must not override its own chain clamp. "off" has
    # no downstream repair pass, so an overlap it emits ships as-is.
    words = [WordTiming("Hi.", 0.0, 0.2), WordTiming("Okay", 0.2, 1.2)]
    seg = SubtitleSegment(1, 0.0, 1.2, "Hi. Okay", words=words)
    out = _proc(mode="off", split_sentences=True).process([seg], audio_duration=100.0)
    assert len(out) == 2
    assert _overlaps(out) == []
    # The acoustic ends must survive (bounds only limit extensions).
    assert out[0].end >= 0.2 - 1e-9


def test_off_mode_split_last_sentence_respects_next_segment():
    # A short final sentence ("Hmm") at the end of one segment used to be
    # extended to min_duration straight into the FOLLOWING segment's span.
    words_a = [WordTiming("Sure.", 0.0, 9.8), WordTiming("Hmm", 9.9, 10.0)]
    words_b = [WordTiming("Next", 10.1, 11.0), WordTiming("line", 11.0, 12.0)]
    segs = [
        SubtitleSegment(1, 0.0, 10.0, "Sure. Hmm", words=words_a),
        SubtitleSegment(2, 10.1, 12.0, "Next line", words=words_b),
    ]
    out = _proc(mode="off", split_sentences=True).process(segs, audio_duration=100.0)
    assert len(out) == 3
    assert _overlaps(out) == []


def test_text_split_keeps_abbreviations_with_their_sentence():
    # The text-level splitter runs whenever alignment produced no word timing,
    # and it used to lack the abbreviation guard that the word-level splitter
    # got — real GPU output cut "Mr. Smith arrived..." into a 0.18s "Mr." cue.
    proc = _proc(mode="minimal")
    assert proc._split_into_sentences("Mr. Smith arrived at dawn. He waited.") == [
        "Mr. Smith arrived at dawn.",
        "He waited.",
    ]
    assert proc._split_into_sentences("Dr. Chen opened the door. Three were gone.") == [
        "Dr. Chen opened the door.",
        "Three were gone.",
    ]


def test_abbreviations_are_not_sentence_ends():
    for word in ("Mr.", "Dr.", "MRS.", "vs.", "J."):
        assert is_sentence_end(word) is False, word
    for word in ("stop.", "done!", "really?", "だ。", 'go."', "行くよ。」", "(fine.)"):
        assert is_sentence_end(word) is True, word


def test_display_cells_counts_wide_characters_twice():
    # Netflix counts a full-width character as 1 and a half-width one as 0.5;
    # display_cells keeps that ratio in half-width units.
    assert display_cells("今日はABCDEFGHIJです") == 20
    assert display_cells("ＡＢＣ") == 6
    assert display_cells("ｱｲｳ") == 3
    assert display_cells("plain ascii") == 11


def test_reading_time_weighs_halfwidth_text_in_a_cjk_line_by_half():
    # 5 full-width + 10 half-width characters at 10 full-width chars/s: 1.0s of
    # reading time, not the 1.5s that counting code points gave.
    proc = TimestampProcessor(
        TimestampConfig(
            mode="full", min_duration=0.1, lead_in_ms=0, linger_ms=0, cjk_chars_per_second=10.0
        ),
        language="ja",
    )
    seg = _seg(1, 0.0, 0.2, text="今日はABCDEFGHIJです", words=False)
    out = proc.process([seg], audio_duration=60.0)
    assert abs(out[0].end - 1.0) < 1e-9
