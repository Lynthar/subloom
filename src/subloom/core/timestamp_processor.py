"""Timestamp post-processing module for improving subtitle timing accuracy."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
import re
import unicodedata

from ..models.config import TimestampConfig
from ..models.subtitle import SubtitleSegment, WordTiming

logger = logging.getLogger(__name__)

# Sentence-ending patterns for different languages
SENTENCE_ENDINGS = re.compile(r"([。！？!?\.\n]+)")
# Japanese/Chinese specific sentence endings
CJK_SENTENCE_ENDINGS = re.compile(r"([。！？」』）]+)")
# Pattern to detect sentence-ending punctuation (for word matching)
SENTENCE_END_CHARS = set("。！？!?.")
# Closing quotes and brackets that may follow a sentence's end: 'said "Go."', 「行くよ。」
CLOSING_CHARS = set("\"'”’」』）)]》")


# Words whose trailing period is an abbreviation, not a sentence boundary.
# Compared case-insensitively; without this the word-level sentence split
# cut "Mr. Smith" into two subtitles at "Mr.".
NON_SENTENCE_ABBREVIATIONS = frozenset(
    {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "st.",
        "sr.",
        "jr.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
        "no.",
        "a.m.",
        "p.m.",
        "u.s.",
    }
)


def is_sentence_end(word: str) -> bool:
    """Check if a word ends with sentence-ending punctuation."""
    if not word:
        return False
    token = word.strip().lower()
    if token in NON_SENTENCE_ABBREVIATIONS:
        return False
    # A single-letter initial ("J." in "J. Smith") is part of a name.
    if len(token) == 2 and token.endswith(".") and token[0].isalpha():
        return False
    for char in reversed(word):
        if char in SENTENCE_END_CHARS:
            return True
        if not char.isspace() and char not in CLOSING_CHARS:
            break
    return False


def display_cells(text: str) -> int:
    """Text length in half-width cells: a wide (East Asian Width F/W) character is 2,
    anything else 1 — Netflix's full-width 1 / half-width 0.5 rule, doubled to stay integral.
    Reading-speed and split-threshold arithmetic in this module all works in these cells."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1 for ch in text)


@dataclass
class TimestampIssue:
    """Record of a detected timestamp issue."""

    segment_index: int
    issue_type: str  # 'overlap', 'long_duration', 'large_gap', 'short_duration', 'exceeds_duration'
    severity: str  # 'info', 'warning', 'error'
    details: str


@dataclass
class GapInfo:
    """Information about a detected gap that may indicate missing speech."""

    start: float
    end: float
    duration: float
    before_segment_index: int
    after_segment_index: int


class TimestampProcessor:
    """
    Post-processor for subtitle timestamps.

    Supports three processing modes:
    - "off": No processing, trust WhisperX output
    - "minimal": Only fix overlaps and ensure minimum duration
    - "full": Complete processing (split, extend, fix all issues)
    """

    # CJK language codes
    CJK_LANGUAGES = {"zh", "ja", "ko", "chinese", "japanese", "korean", "yue", "wuu"}

    def __init__(self, config: TimestampConfig, language: Optional[str] = None):
        """
        Initialize timestamp processor.

        Args:
            config: The timestamp section of AppConfig; `enabled` is for the caller to honour.
            language: Detected language code for CJK optimization.
        """
        self.mode = config.mode
        self.language = language
        self.min_duration = config.min_duration
        self.max_duration = config.max_duration
        self.min_gap = config.min_gap
        self.max_gap_warning = config.max_gap_warning
        self.chars_per_second = config.chars_per_second
        self.cjk_chars_per_second = config.cjk_chars_per_second
        self.split_threshold = config.split_threshold
        self.split_sentences = config.split_sentences
        self.lead_in_ms = config.lead_in_ms
        self.linger_ms = config.linger_ms
        self._issues: List[TimestampIssue] = []
        self._gaps: List[GapInfo] = []

        # Effective reading speed in half-width cells per second (see display_cells):
        # the CJK setting counts full-width characters, two cells each.
        if self._is_cjk_language(language):
            self._effective_cps = 2 * config.cjk_chars_per_second
            logger.debug(f"Using CJK reading speed: {config.cjk_chars_per_second} chars/sec")
        else:
            self._effective_cps = config.chars_per_second

    @classmethod
    def _is_cjk_language(cls, language: Optional[str]) -> bool:
        """Check if language is CJK (Chinese, Japanese, Korean)."""
        if not language:
            return False
        return language.lower() in cls.CJK_LANGUAGES

    def _get_split_threshold(self) -> int:
        """Get language-appropriate split threshold, in half-width cells."""
        if self._is_cjk_language(self.language):
            # CJK characters are denser: half the count, but at least 15 full-width characters
            return 2 * max(15, self.split_threshold // 2)
        return self.split_threshold

    def process(
        self,
        segments: List[SubtitleSegment],
        audio_duration: Optional[float] = None,
    ) -> List[SubtitleSegment]:
        """
        Execute post-processing pipeline based on mode.

        Args:
            segments: List of subtitle segments to process.
            audio_duration: Total audio duration for boundary checking.

        Returns:
            Processed subtitle segments with corrected timestamps.
        """
        self._issues = []
        self._gaps = []

        if not segments:
            return segments

        # Sentence splitting runs independently of mode if enabled
        if self.split_sentences:
            original_count = len(segments)
            segments = self._split_by_sentences(segments)
            if len(segments) != original_count:
                logger.info(
                    f"Sentence splitting: {original_count} segments -> {len(segments)} segments"
                )

        # Lead-in / linger applies to ALL modes, "off" included: aligned timestamps are word onset
        # and offset, not on-screen timing. Without it subtitles feel late and vanish too fast
        # whichever mode the user picked.
        segments = self._apply_lead_in_linger(segments, audio_duration)

        # Mode: off - trust WhisperX output completely (apart from lead-in/linger)
        if self.mode == "off":
            logger.debug("Timestamp processing mode: off (lead-in/linger only)")
            return self._reindex(segments)

        # Mode: minimal - only essential fixes
        if self.mode == "minimal":
            logger.debug("Timestamp processing mode: minimal")
            segments = self._cap_overlong_displays(segments)
            segments = self._fix_overlaps(segments)
            segments = self._ensure_minimum_duration(segments)
            # _ensure_minimum_duration extends short segments without looking at the next start,
            # re-introducing the overlaps _fix_overlaps just resolved. Re-run the gap pass after it.
            segments = self._ensure_minimum_gap(segments)
            if audio_duration:
                segments = self._clamp_to_duration(segments, audio_duration)
            return self._reindex(self._merge_collisions(segments))

        # Mode: full - complete processing (original behavior)
        logger.debug("Timestamp processing mode: full")

        # 0. Pre-processing for segments without word-level timestamps
        segments = self._split_multi_sentence_segments(segments)
        segments = self._extend_segment_end_times(segments)

        # 1. Validate and record issues
        self._validate(segments, audio_duration)

        # 2. Apply fixes
        segments = self._fix_overlaps(segments)
        segments = self._fix_long_displays(segments, audio_duration)
        segments = self._ensure_minimum_duration(segments)
        segments = self._ensure_minimum_gap(segments)

        # 3. Clamp to audio duration if known
        if audio_duration:
            segments = self._clamp_to_duration(segments, audio_duration)
        segments = self._merge_collisions(segments)

        # 4. Detect potential gaps (missing speech)
        self._detect_large_gaps(segments, audio_duration)

        # 5. Reindex
        segments = self._reindex(segments)

        # Log summary
        self._log_summary()

        return segments

    def _cap_overlong_displays(
        self,
        segments: List[SubtitleSegment],
    ) -> List[SubtitleSegment]:
        """
        Cap segments whose end-time looks stretched past the actual speech
        offset.

        Critically, only segments that LACK word-level timestamps are
        candidates. When word alignment succeeded, seg.end is set from
        words[-1].end (acoustic-aligned to the moment speech actually
        stopped), so capping based on a hypothetical "reading time" would
        truncate genuinely slow, emphatic, or paused delivery — and leave
        users with subtitles that vanish before the speaker finishes.

        Unaligned segments fall back to the raw Whisper segment end-time,
        which transcribers sometimes stretch up to the next utterance's
        onset (or the end of the file) when there's no following speech to
        bound them. Those are the ones we cap.
        """
        result: List[SubtitleSegment] = []
        linger = self.linger_ms / 1000.0
        # Speech is roughly half as fast as silent reading: take 2x the readable time, then add
        # linger and a generous buffer. Alignment-failed segments are rare, so err on
        # under-clamping.
        soft_buffer = 1.5

        for seg in segments:
            # Trust acoustic-aligned end-times completely.
            if seg.has_word_timestamps():
                result.append(seg)
                continue

            text = seg.text.strip()
            chars = max(1, display_cells(text))
            readable = chars / self._effective_cps
            speech_time = readable * 2  # rough speech-vs-reading speed ratio

            comfortable = speech_time + linger + soft_buffer
            comfortable = max(comfortable, self.min_duration)
            comfortable = min(comfortable, self.max_duration + linger)

            actual = seg.end - seg.start
            if actual > comfortable:
                logger.debug(
                    f"Capping unaligned segment {seg.index}: {actual:.2f}s -> "
                    f"{comfortable:.2f}s (text length {chars} cells)"
                )
                seg = SubtitleSegment(
                    index=seg.index,
                    start=seg.start,
                    end=seg.start + comfortable,
                    text=seg.text,
                    words=seg.words,
                    confidence=seg.confidence,
                )
            result.append(seg)
        return result

    def _apply_lead_in_linger(
        self,
        segments: List[SubtitleSegment],
        audio_duration: Optional[float],
    ) -> List[SubtitleSegment]:
        """
        Shift each segment's start earlier by lead_in_ms and extend end by
        linger_ms, without overlapping neighbours or going outside audio bounds.

        This compensates for acoustic-alignment timestamps being word onset/
        offset times, which are too tight for comfortable subtitle reading.
        """
        if not segments:
            return segments
        if self.lead_in_ms <= 0 and self.linger_ms <= 0:
            return segments

        lead_in = self.lead_in_ms / 1000.0
        linger = self.linger_ms / 1000.0
        result: List[SubtitleSegment] = []

        for i, seg in enumerate(segments):
            new_start = seg.start - lead_in
            new_end = seg.end + linger

            # Don't go before audio start or before previous segment end
            if i == 0:
                new_start = max(new_start, 0.0)
            else:
                prev_end = result[-1].end
                # Keep at least min_gap between segments
                new_start = max(new_start, prev_end + self.min_gap)
                # Lead-in only ever moves a start EARLIER. If the previous segment already ends
                # at/after our acoustic onset, the clamp above manufactures a LATER start — past the
                # audio end that inverts the segment.
                new_start = min(new_start, seg.start)

            # Don't extend past next segment's (already-shifted) start. We
            # don't have it yet, but we can clamp to the original next.start
            # minus lead_in (the start we'd assign next) and keep min_gap.
            if i < len(segments) - 1:
                next_original_start = segments[i + 1].start
                # The next segment will want to start at next_original_start - lead_in,
                # so we cap our end to leave a gap before that.
                next_target_start = max(next_original_start - lead_in, new_start)
                new_end = min(new_end, next_target_start - self.min_gap)

            # Never pull the end in before the acoustic offset: with a neighbour close by the clamp
            # above would drop new_end below seg.end and truncate real speech. The next lead-in
            # yields instead.
            new_end = max(new_end, seg.end)

            # Clamp to audio duration if known
            if audio_duration and new_end > audio_duration:
                new_end = audio_duration

            # Final safety: keep ordering valid even if shifts collided
            if new_end <= new_start:
                new_end = new_start + max(self.min_duration, 0.1)

            result.append(
                SubtitleSegment(
                    index=seg.index,
                    start=new_start,
                    end=new_end,
                    text=seg.text,
                    words=seg.words,
                    confidence=seg.confidence,
                )
            )

        return result

    def get_issues(self) -> List[TimestampIssue]:
        """Get list of detected timestamp issues."""
        return self._issues.copy()

    def get_potential_gaps(self) -> List[GapInfo]:
        """Get list of large gaps that may indicate missed speech."""
        return self._gaps.copy()

    def _validate(
        self,
        segments: List[SubtitleSegment],
        audio_duration: Optional[float],
    ) -> None:
        """Validate segments and record issues."""
        for i, seg in enumerate(segments):
            # Check negative/zero duration
            if seg.end <= seg.start:
                self._issues.append(
                    TimestampIssue(
                        segment_index=seg.index,
                        issue_type="negative_duration",
                        severity="error",
                        details=f"End ({seg.end:.2f}s) <= Start ({seg.start:.2f}s)",
                    )
                )

            # Check excessively long duration
            duration = seg.end - seg.start
            if duration > self.max_duration:
                self._issues.append(
                    TimestampIssue(
                        segment_index=seg.index,
                        issue_type="long_duration",
                        severity="warning",
                        details=f"Duration {duration:.2f}s exceeds max {self.max_duration}s",
                    )
                )

            # Check overlap with previous
            if i > 0:
                prev = segments[i - 1]
                if seg.start < prev.end:
                    overlap = prev.end - seg.start
                    self._issues.append(
                        TimestampIssue(
                            segment_index=seg.index,
                            issue_type="overlap",
                            severity="warning" if overlap < 0.5 else "error",
                            details=f"Overlaps with previous segment by {overlap:.2f}s",
                        )
                    )

            # Check exceeds audio duration
            if audio_duration and seg.end > audio_duration:
                self._issues.append(
                    TimestampIssue(
                        segment_index=seg.index,
                        issue_type="exceeds_duration",
                        severity="error",
                        details=f"End ({seg.end:.2f}s) exceeds audio duration ({audio_duration:.2f}s)",
                    )
                )

    def _fix_overlaps(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Fix overlapping timestamps between segments."""
        if len(segments) <= 1:
            return segments

        result = [segments[0]]
        for seg in segments[1:]:
            prev = result[-1]

            if seg.start < prev.end:
                # Calculate midpoint for fair distribution
                midpoint = (prev.end + seg.start) / 2

                # Adjust previous segment's end
                adjusted_prev_end = midpoint - self.min_gap / 2
                if adjusted_prev_end > prev.start + 0.1:
                    result[-1] = SubtitleSegment(
                        index=prev.index,
                        start=prev.start,
                        end=adjusted_prev_end,
                        text=prev.text,
                        words=prev.words,
                        confidence=prev.confidence,
                    )

                # Adjust current segment's start
                adjusted_start = midpoint + self.min_gap / 2
                seg = SubtitleSegment(
                    index=seg.index,
                    start=adjusted_start,
                    end=seg.end,
                    text=seg.text,
                    words=seg.words,
                    confidence=seg.confidence,
                )

            result.append(seg)
        return result

    def _fix_long_displays(
        self,
        segments: List[SubtitleSegment],
        audio_duration: Optional[float],
    ) -> List[SubtitleSegment]:
        """Fix segments with excessively long display duration."""
        result = []
        for i, seg in enumerate(segments):
            # Trust aligned end-times completely — same contract as _cap_overlong_displays: with
            # word timing seg.end is when speech stopped, and a reading-time estimate cuts it off
            # mid-speech.
            if seg.has_word_timestamps():
                result.append(seg)
                continue

            duration = seg.end - seg.start

            if duration > self.max_duration:
                # Estimate reasonable duration based on text length
                chars = display_cells(seg.text)
                estimated_duration = max(
                    self.min_duration, min(chars / self._effective_cps, self.max_duration)
                )

                new_end = seg.start + estimated_duration

                # Don't exceed next segment's start
                if i < len(segments) - 1:
                    next_start = segments[i + 1].start
                    new_end = min(new_end, next_start - self.min_gap)

                # Don't exceed audio duration
                if audio_duration:
                    new_end = min(new_end, audio_duration)

                # Ensure minimum duration
                new_end = max(new_end, seg.start + self.min_duration)

                seg = SubtitleSegment(
                    index=seg.index,
                    start=seg.start,
                    end=new_end,
                    text=seg.text,
                    words=seg.words,
                    confidence=seg.confidence,
                )

            result.append(seg)
        return result

    def _ensure_minimum_duration(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Ensure all segments have minimum display duration."""
        result = []
        for seg in segments:
            duration = seg.end - seg.start
            if duration < self.min_duration:
                seg = SubtitleSegment(
                    index=seg.index,
                    start=seg.start,
                    end=seg.start + self.min_duration,
                    text=seg.text,
                    words=seg.words,
                    confidence=seg.confidence,
                )
            result.append(seg)
        return result

    def _ensure_minimum_gap(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Ensure minimum gap between consecutive segments."""
        if len(segments) <= 1:
            return segments

        result = [segments[0]]
        for seg in segments[1:]:
            prev = result[-1]
            gap = seg.start - prev.end

            if gap < self.min_gap:
                # Adjust previous segment's end
                new_prev_end = seg.start - self.min_gap
                if new_prev_end > prev.start + 0.1:
                    result[-1] = SubtitleSegment(
                        index=prev.index,
                        start=prev.start,
                        end=new_prev_end,
                        text=prev.text,
                        words=prev.words,
                        confidence=prev.confidence,
                    )

            result.append(seg)
        return result

    def _clamp_to_duration(
        self,
        segments: List[SubtitleSegment],
        audio_duration: float,
    ) -> List[SubtitleSegment]:
        """Ensure all timestamps are within audio duration."""
        result = []
        for seg in segments:
            if seg.end > audio_duration:
                seg = SubtitleSegment(
                    index=seg.index,
                    start=min(seg.start, audio_duration - 0.1),
                    end=audio_duration,
                    text=seg.text,
                    words=seg.words,
                    confidence=seg.confidence,
                )
            result.append(seg)
        return result

    def _merge_collisions(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Merge each segment still overlapping its predecessor into it; runs last, before reindex.

        Any earlier and an overlap ships: the gap pass keeps cues >= 0.1s, the clamp stacks the tail.
        """
        result: List[SubtitleSegment] = []
        for seg in segments:
            if result and seg.start < result[-1].end:
                prev = result[-1]
                result[-1] = SubtitleSegment(
                    index=prev.index,
                    start=prev.start,
                    end=max(prev.end, seg.end),
                    text=self._join_words([prev.text, seg.text]),
                    words=prev.words + seg.words if prev.words and seg.words else None,
                    confidence=min(prev.confidence, seg.confidence),
                )
                continue
            result.append(seg)
        return result

    def _detect_large_gaps(
        self,
        segments: List[SubtitleSegment],
        audio_duration: Optional[float],
    ) -> None:
        """Detect large gaps that may indicate missed speech."""
        # Check gap at beginning
        if segments and segments[0].start > self.max_gap_warning:
            self._gaps.append(
                GapInfo(
                    start=0,
                    end=segments[0].start,
                    duration=segments[0].start,
                    before_segment_index=0,
                    after_segment_index=segments[0].index,
                )
            )
            logger.info(
                f"Large gap at start: 0.00s - {segments[0].start:.2f}s "
                f"(duration: {segments[0].start:.2f}s)"
            )

        # Check gaps between segments
        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]
            gap = curr.start - prev.end

            if gap > self.max_gap_warning:
                self._gaps.append(
                    GapInfo(
                        start=prev.end,
                        end=curr.start,
                        duration=gap,
                        before_segment_index=prev.index,
                        after_segment_index=curr.index,
                    )
                )
                logger.info(
                    f"Large gap detected: {prev.end:.2f}s - {curr.start:.2f}s "
                    f"(duration: {gap:.2f}s) - may indicate missed speech"
                )

        # Check gap at end
        if audio_duration and segments:
            last = segments[-1]
            end_gap = audio_duration - last.end
            if end_gap > self.max_gap_warning:
                self._gaps.append(
                    GapInfo(
                        start=last.end,
                        end=audio_duration,
                        duration=end_gap,
                        before_segment_index=last.index,
                        after_segment_index=-1,  # -1 indicates end of audio
                    )
                )
                logger.info(
                    f"Large gap at end: {last.end:.2f}s - {audio_duration:.2f}s "
                    f"(duration: {end_gap:.2f}s)"
                )

    def _reindex(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Reindex segments starting from 1, preserving all other fields."""
        return [
            SubtitleSegment(
                index=i + 1,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=seg.words,
                confidence=seg.confidence,
            )
            for i, seg in enumerate(segments)
        ]

    def _split_multi_sentence_segments(
        self, segments: List[SubtitleSegment]
    ) -> List[SubtitleSegment]:
        """
        Split segments containing multiple sentences into separate segments.

        This is particularly useful when forced alignment fails and Whisper
        returns long segments with multiple sentences grouped together.
        """
        result = []

        for seg in segments:
            # Skip if segment has word-level timestamps (alignment worked)
            if seg.has_word_timestamps():
                result.append(seg)
                continue

            # Skip short segments (use language-aware threshold)
            threshold = self._get_split_threshold()
            if display_cells(seg.text) < threshold:
                result.append(seg)
                continue

            # Try to split by sentence endings
            sentences = self._split_into_sentences(seg.text)

            if len(sentences) <= 1:
                result.append(seg)
                continue

            # Distribute time proportionally across sentences, strictly within [seg.start, seg.end]:
            # inflating each piece to min_duration overflowed the span and produced negative last
            # pieces.
            total_chars = sum(display_cells(s) for s in sentences)
            total_duration = max(0.0, seg.end - seg.start)
            current_time = seg.start
            n = len(sentences)

            for i, sentence in enumerate(sentences):
                char_ratio = display_cells(sentence) / total_chars if total_chars > 0 else 1.0 / n
                end_time = seg.end if i == n - 1 else current_time + total_duration * char_ratio
                end_time = min(end_time, seg.end)
                if end_time <= current_time:
                    end_time = min(seg.end, current_time + 0.001)

                result.append(
                    SubtitleSegment(
                        index=len(result) + 1,
                        start=current_time,
                        end=end_time,
                        text=sentence.strip(),
                        confidence=seg.confidence,
                    )
                )

                current_time = end_time

            logger.debug(f"Split segment {seg.index} into {len(sentences)} parts")

        return result

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences based on punctuation.

        Handles both Western and CJK punctuation.
        """
        # First try CJK sentence endings
        if any("\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff" for c in text):
            # Contains CJK characters, use CJK pattern
            parts = CJK_SENTENCE_ENDINGS.split(text)
        else:
            # Use standard sentence endings
            parts = SENTENCE_ENDINGS.split(text)

        # Recombine parts (pattern split separates delimiters)
        sentences = []
        current = ""

        for part in parts:
            current += part
            if SENTENCE_ENDINGS.match(part) or CJK_SENTENCE_ENDINGS.match(part):
                # ...unless the period is an abbreviation. With alignment
                # unavailable this is the only splitter that runs, so an
                # unguarded "Mr." becomes a 0.18s cue of its own.
                tokens = current.split()
                if tokens and not is_sentence_end(tokens[-1]):
                    continue
                if current.strip():
                    sentences.append(current.strip())
                current = ""

        # Add remaining text if any
        if current.strip():
            sentences.append(current.strip())

        # Filter out empty sentences and very short ones
        sentences = [s for s in sentences if len(s) >= 2]

        return sentences if sentences else [text]

    def _extend_segment_end_times(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """
        Extend segment end times based on text length.

        When forced alignment fails, segment end times may be too short.
        This method extends end times based on estimated reading time,
        while respecting the next segment's start time.
        """
        if not segments:
            return segments

        result = []

        for i, seg in enumerate(segments):
            # Skip if segment has word-level timestamps (alignment worked)
            if seg.has_word_timestamps():
                result.append(seg)
                continue

            # Calculate minimum required duration based on text length
            # Using language-appropriate reading speed
            chars = display_cells(seg.text)
            min_required_duration = max(self.min_duration, chars / self._effective_cps)

            current_duration = seg.end - seg.start

            # Only extend if current duration is too short
            if current_duration < min_required_duration:
                new_end = seg.start + min_required_duration

                # Don't exceed next segment's start time (with gap)
                if i < len(segments) - 1:
                    next_start = segments[i + 1].start
                    max_end = next_start - self.min_gap
                    new_end = min(new_end, max_end)

                # Ensure we don't make it shorter than original
                new_end = max(new_end, seg.end)

                if new_end != seg.end:
                    seg = SubtitleSegment(
                        index=seg.index,
                        start=seg.start,
                        end=new_end,
                        text=seg.text,
                        words=seg.words,
                        confidence=seg.confidence,
                    )

            result.append(seg)

        return result

    def _split_by_sentences(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """
        Split segments by sentence boundaries using word-level timestamps.

        Detects sentence-ending punctuation (and large word gaps) and uses
        the word-level timestamps to derive precise per-sentence timing.
        Sub-segments retain the corresponding word timestamps so downstream
        passes (cap / lead-in / linger) treat them as fully aligned and do
        not fall back to text-length heuristics.
        """
        result = []

        for pos, seg in enumerate(segments):
            # If no word timestamps, use fallback proportional splitting
            if not seg.has_word_timestamps():
                split_segs = self._split_segment_proportionally(seg)
                result.extend(split_segs)
                continue

            # Find sentence boundaries using word timestamps
            sentences = self._extract_sentences_with_timing(seg)

            if len(sentences) <= 1:
                # No splitting needed
                result.append(seg)
                continue

            # The next segment's start bounds the last sentence's extension:
            # in "off" mode nothing downstream repairs overlaps, so the split
            # must not manufacture one.
            next_segment_start = segments[pos + 1].start if pos + 1 < len(segments) else None
            corrected_sentences = self._apply_sentence_timing_corrections(
                sentences, next_segment_start=next_segment_start
            )

            # Create new segments for each sentence, preserving word timing.
            for sentence_text, start_time, end_time, words in corrected_sentences:
                if not sentence_text.strip():
                    continue

                result.append(
                    SubtitleSegment(
                        index=len(result) + 1,
                        start=start_time,
                        end=end_time,
                        text=sentence_text.strip(),
                        words=words,
                        confidence=seg.confidence,
                    )
                )

        return result

    def _apply_sentence_timing_corrections(
        self,
        sentences: List[Tuple[str, float, float, Optional[List[WordTiming]]]],
        next_segment_start: Optional[float] = None,
    ) -> List[Tuple[str, float, float, Optional[List[WordTiming]]]]:
        """
        Apply timing corrections to split sentences.

        Each sentence's end-time starts from its word-level acoustic boundary
        (i.e. the moment speech actually stopped). The shared lead-in/linger
        pass downstream then adds the configured padding uniformly.

        Earlier versions extended the LAST sentence's end-time to the parent
        segment's end-time "for lingering effect". That left short final
        utterances (like "嗯") hanging on screen for the entire span up to
        the next utterance — the symptom users reported as "subtitle stays
        too long". This implementation trusts the acoustic boundary instead
        and lets `_apply_lead_in_linger` provide a uniform tail.

        Corrections still applied:
        1. Chain-style: every extension is bounded by the next sentence's
           start (or, for the last sentence, by `next_segment_start`) minus
           min_gap — a readability extension must NEVER create an overlap.
           In minimal/full mode _fix_overlaps would repair one downstream,
           but in "off" mode nothing does: the old unconditional
           "at least min_duration" floor turned adjacent short sentences
           into overlapping cues that shipped as-is.
        2. Minimum readable duration: short subtitles get extended (within
           that bound) so they don't flash by faster than text can be read.
        3. The acoustic end itself is never reduced — bounds only limit
           extensions, so overlaps already present in the input pass
           through untouched (mode "off" trusts its input).

        Args:
            sentences: 4-tuples (text, start, end, words) from word timestamps.
            next_segment_start: start of the FOLLOWING segment, if any —
                bounds the last sentence's extension.

        Returns:
            Corrected list of (text, start, end, words) tuples.
        """
        if not sentences:
            return sentences

        n = len(sentences)
        corrected: List[Tuple[str, float, float, Optional[List[WordTiming]]]] = []

        for i, (text, start, original_end, words) in enumerate(sentences):
            is_last = i == n - 1
            # Hard bound for any extension of this sentence's end.
            bound = next_segment_start if is_last else sentences[i + 1][1]

            # Calculate minimum readable duration based on text length
            char_count = display_cells(text)
            min_readable_duration = max(
                self.min_duration,
                char_count / self._effective_cps,
            )

            # Always start from the acoustic word-level end. This is the
            # ground truth for "when speech ended"; downstream lead-in/
            # linger handles padding uniformly.
            end = original_end

            # Ensure minimum readable duration (short utterances need to be
            # held on screen long enough for the eye to catch them), but only
            # as far as the bound allows.
            if end - start < min_readable_duration:
                end = start + min_readable_duration
                if bound is not None:
                    end = min(end, bound - self.min_gap)

            # Extensions may be clamped below the acoustic end when the bound
            # sits close; never truncate actual speech because of it.
            end = max(end, original_end)

            corrected.append((text, start, end, words))

        return corrected

    def _extract_sentences_with_timing(
        self, seg: SubtitleSegment
    ) -> List[Tuple[str, float, float, Optional[List[WordTiming]]]]:
        """
        Extract sentences from segment with their timing using word timestamps.

        Splits on:
        1. Sentence-ending punctuation (。！？!?.)
        2. Large gaps between words (>= 0.5s) indicating natural pauses

        Returns:
            List of (sentence_text, start_time, end_time, words) tuples. The
            words list lets downstream code recognise the resulting sub-segment
            as acoustically aligned and trust its end-time.
        """
        if not seg.words:
            return [(seg.text, seg.start, seg.end, None)]

        # Threshold for considering a gap as sentence boundary (seconds)
        gap_threshold = 0.5

        sentences: List[Tuple[str, float, float, Optional[List[WordTiming]]]] = []
        current_sentence_words: List[WordTiming] = []
        current_text_parts: List[str] = []
        prev_word_end: Optional[float] = None

        def flush() -> None:
            if not current_sentence_words:
                return
            sentence_text = self._join_words(current_text_parts)
            start_time = current_sentence_words[0].start
            end_time = current_sentence_words[-1].end
            sentences.append((sentence_text, start_time, end_time, list(current_sentence_words)))

        for word in seg.words:
            word_text = word.word.strip()
            if not word_text:
                continue

            # Check for large gap before this word (indicates sentence boundary)
            if prev_word_end is not None and current_sentence_words:
                gap = word.start - prev_word_end
                if gap >= gap_threshold:
                    flush()
                    current_sentence_words = []
                    current_text_parts = []

            current_sentence_words.append(word)
            current_text_parts.append(word_text)
            prev_word_end = word.end

            # Check if this word ends with sentence-ending punctuation
            if is_sentence_end(word_text):
                flush()
                current_sentence_words = []
                current_text_parts = []
                prev_word_end = None  # Reset after punctuation split

        # Handle remaining words (sentence without ending punctuation)
        flush()

        return sentences if sentences else [(seg.text, seg.start, seg.end, list(seg.words))]

    def _join_words(self, words: List[str]) -> str:
        """
        Join words into text, handling CJK vs Western spacing.
        """
        if not words:
            return ""

        # Check if primarily CJK
        is_cjk = self._is_cjk_language(self.language)

        if is_cjk:
            # CJK: no space between characters
            return "".join(words)
        else:
            # Western: space between words
            return " ".join(words)

    def _split_segment_proportionally(self, seg: SubtitleSegment) -> List[SubtitleSegment]:
        """
        Fallback: split segment proportionally when no word timestamps available.

        Spans are distributed by character ratio and kept strictly within the
        parent segment's [start, end]. We deliberately do NOT inflate each piece
        to min_duration here: the old code did, which accumulated past seg.end
        and then forced the last piece back to seg.end — producing subtitles that
        ran far past the real segment and even negative-duration spans. Any
        readability floor is applied later by _ensure_minimum_duration /
        _ensure_minimum_gap, which respect neighbouring segments.
        """
        # Try to split by sentence endings
        sentences = [s for s in self._split_into_sentences(seg.text) if s.strip()]

        if len(sentences) <= 1:
            return [seg]

        # Distribute time proportionally across sentences, strictly within [seg.start, seg.end]:
        # inflating each piece to min_duration overflowed the span and produced negative last
        # pieces.
        total_chars = sum(display_cells(s) for s in sentences)
        total_duration = max(0.0, seg.end - seg.start)
        current_time = seg.start
        result: List[SubtitleSegment] = []
        n = len(sentences)

        for i, sentence in enumerate(sentences):
            char_ratio = display_cells(sentence) / total_chars if total_chars > 0 else 1.0 / n
            # Last piece lands exactly on seg.end; others accrue proportionally.
            end_time = seg.end if i == n - 1 else current_time + total_duration * char_ratio
            # Never exceed the parent end or run backwards.
            end_time = min(end_time, seg.end)
            if end_time <= current_time:
                end_time = min(seg.end, current_time + 0.001)

            result.append(
                SubtitleSegment(
                    index=len(result) + 1,
                    start=current_time,
                    end=end_time,
                    text=sentence.strip(),
                    confidence=seg.confidence,
                )
            )
            current_time = end_time

        return result if result else [seg]

    def _log_summary(self) -> None:
        """Log processing summary."""
        error_count = sum(1 for i in self._issues if i.severity == "error")
        warning_count = sum(1 for i in self._issues if i.severity == "warning")
        gap_count = len(self._gaps)

        if error_count or warning_count or gap_count:
            logger.info(
                f"Timestamp processing: {error_count} errors, {warning_count} warnings, "
                f"{gap_count} potential gaps detected"
            )
        else:
            logger.info("Timestamp processing: No issues detected")
