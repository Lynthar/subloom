"""Subtitle processing module for SRT file handling."""

from pathlib import Path
from typing import Iterable, List, Optional
import codecs
import logging
import os
import re
import tempfile

import pysrt

from ..models.subtitle import SubtitleSegment
from ..exceptions import SubtitleError

logger = logging.getLogger(__name__)

# Unicode letters/digits, hyphen, underscore, space (not leading). Covers en,
# zh, zh-TW, yue, pt-BR and loose labels, while excluding every path
# metacharacter (/ \ . : and a leading dot).
_LANGUAGE_TOKEN_RE = re.compile(r"^\w[\w -]{0,31}$")


def validate_language_codes(languages: Iterable[str]) -> None:
    """Reject language tokens that are not safe to embed in a filename.

    Language codes go into output paths verbatim ({stem}.{lang}.srt), so a
    crafted "language" like ``../../x`` from an API caller (or a typo'd CLI
    flag) would write .srt files outside the output directory. Raises
    ValueError on the first offending token.
    """
    for lang in languages:
        if not lang or not _LANGUAGE_TOKEN_RE.fullmatch(lang):
            raise ValueError(
                f"Invalid language code {lang!r}: use letters/digits/hyphen, "
                "e.g. en, zh, zh-TW, yue"
            )


def normalize_target_languages(languages: List[str]) -> List[str]:
    """Validate target languages and drop duplicates, preserving order.

    A repeated ``-t zh`` (or a duplicated API entry) would otherwise run the
    full LLM translation twice and write the same output file twice. Raises
    ValueError on an unsafe token (see validate_language_codes).
    """
    validate_language_codes(languages)
    deduped = list(dict.fromkeys(languages))
    if len(deduped) != len(languages):
        logger.info("Dropping duplicate target languages: %s -> %s", languages, deduped)
    return deduped


class SubtitleProcessor:
    """Subtitle file processor."""

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    @staticmethod
    def seconds_to_time(seconds: float) -> pysrt.SubRipTime:
        """Convert seconds to SRT time, rounded to the nearest millisecond."""
        # Round the total, not the fraction: 59.9996 s must carry into the next second.
        return pysrt.SubRipTime.from_ordinal(round(seconds * 1000))

    @staticmethod
    def time_to_seconds(time: pysrt.SubRipTime) -> float:
        """Convert SRT time format to seconds."""
        return float(
            time.hours * 3600 + time.minutes * 60 + time.seconds + time.milliseconds / 1000
        )

    def segments_to_srt(self, segments: List[SubtitleSegment]) -> pysrt.SubRipFile:
        """Convert subtitle segments to pysrt object."""
        srt_file = pysrt.SubRipFile()

        for seg in segments:
            item = pysrt.SubRipItem(
                index=seg.index,
                start=self.seconds_to_time(seg.start),
                end=self.seconds_to_time(seg.end),
                text=seg.text,
            )
            srt_file.append(item)

        return srt_file

    def srt_to_segments(self, srt_file: pysrt.SubRipFile) -> List[SubtitleSegment]:
        """Convert pysrt object to subtitle segments."""
        segments = []

        for item in srt_file:
            segments.append(
                SubtitleSegment(
                    index=item.index,
                    start=self.time_to_seconds(item.start),
                    end=self.time_to_seconds(item.end),
                    text=item.text,
                )
            )

        return segments

    def save(
        self,
        segments: List[SubtitleSegment],
        output_path: Path,
        encoding: Optional[str] = None,
    ) -> None:
        """
        Save subtitles to SRT file.

        Args:
            segments: Subtitle segments to save.
            output_path: Output file path.
            encoding: File encoding.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        srt_file = self.segments_to_srt(segments)

        # Same-directory temp file, then atomic replace: a crash mid-write must
        # not leave a truncated .srt where a player — or a re-run's collision
        # check — takes it for a finished subtitle.
        fd, tmp_name = tempfile.mkstemp(
            dir=str(output_path.parent), prefix=output_path.name + ".", suffix=".tmp"
        )
        os.close(fd)
        try:
            srt_file.save(tmp_name, encoding=encoding or self.encoding)
            os.replace(tmp_name, str(output_path))
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

        logger.info(f"Subtitles saved: {output_path}")

    def load(self, input_path: Path, encoding: Optional[str] = None) -> List[SubtitleSegment]:
        """
        Load subtitles from SRT file.

        Args:
            input_path: Input file path.
            encoding: File encoding.

        Returns:
            List of subtitle segments.
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise SubtitleError(f"Subtitle file not found: {input_path}")

        # BOM sniff before the trial chain: a UTF-16 file "decodes" without
        # error as ISO-8859-1 into NUL-riddled mojibake, so the fallback chain
        # below must never get to see one.
        with open(input_path, "rb") as fh:
            head = fh.read(2)
        if head in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
            encodings_to_try = ["utf-16"]  # the codec consumes the BOM
        else:
            encodings_to_try = [
                encoding or self.encoding,
                "utf-8-sig",
                "gbk",
                "gb2312",
                "iso-8859-1",
            ]

        for enc in encodings_to_try:
            try:
                srt_file = pysrt.open(str(input_path), encoding=enc)
                if enc == "iso-8859-1":
                    # ISO-8859-1 maps every byte, so this "success" proves
                    # nothing — the text may be mojibake from an encoding we
                    # didn't try. Say so instead of silently corrupting.
                    logger.warning(
                        f"Decoded {input_path.name} as ISO-8859-1 (last-resort fallback); "
                        "if the text looks garbled, re-save the file as UTF-8"
                    )
                elif enc != (encoding or self.encoding):
                    logger.info(f"Loaded file with encoding: {enc}")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise SubtitleError(f"Cannot decode subtitle file: {input_path}")

        segments = self.srt_to_segments(srt_file)
        logger.info(f"Loaded subtitles: {input_path} ({len(segments)} segments)")

        return segments

    def merge_bilingual(
        self,
        original: List[SubtitleSegment],
        translated: List[SubtitleSegment],
        original_on_top: bool = True,
    ) -> List[SubtitleSegment]:
        """
        Merge original and translated subtitles into bilingual format.

        Args:
            original: Original subtitles.
            translated: Translated subtitles.
            original_on_top: Place original text on top.

        Returns:
            Merged bilingual subtitles.
        """
        merged = []
        trans_dict = {seg.index: seg for seg in translated}

        for orig in original:
            trans = trans_dict.get(orig.index)
            if trans:
                if original_on_top:
                    text = f"{orig.text}\n{trans.text}"
                else:
                    text = f"{trans.text}\n{orig.text}"
            else:
                text = orig.text

            merged.append(
                SubtitleSegment(
                    index=orig.index,
                    start=orig.start,
                    end=orig.end,
                    text=text,
                )
            )

        return merged


def check_translation(
    translation: List[SubtitleSegment], source: List[SubtitleSegment]
) -> List[str]:
    """N in, N out: a translation keeps its source's segment count and index sequence."""
    if len(translation) != len(source):
        return [f"{len(translation)} segments out for {len(source)} in"]
    if [s.index for s in translation] != [s.index for s in source]:
        return ["segment indices differ from the source's"]
    return []


def check_structure(
    segments: List[SubtitleSegment], audio_duration: Optional[float] = None
) -> List[str]:
    """Structural violations of the SRT these segments are written as; empty means sound.

    Times, the audio end included, are judged at the millisecond the file stores (a cue clamped
    to the audio end rounds with it); a falsy audio_duration skips the end check.
    """
    violations: List[str] = []
    seen = set()
    audio_end_ms = (
        SubtitleProcessor.seconds_to_time(audio_duration).ordinal if audio_duration else None
    )
    prev_end_ms = None
    for seg in segments:
        start_ms = SubtitleProcessor.seconds_to_time(seg.start).ordinal
        end_ms = SubtitleProcessor.seconds_to_time(seg.end).ordinal
        if seg.index in seen:
            violations.append(f"#{seg.index}: duplicate index")
        seen.add(seg.index)
        if start_ms < 0:
            violations.append(f"#{seg.index}: starts before 0")
        if end_ms <= start_ms:
            violations.append(f"#{seg.index}: non-positive duration ({start_ms}-{end_ms} ms)")
        if prev_end_ms is not None and start_ms < prev_end_ms:
            violations.append(f"#{seg.index}: overlaps previous by {prev_end_ms - start_ms} ms")
        if audio_end_ms is not None and end_ms > audio_end_ms:
            violations.append(f"#{seg.index}: ends past the audio ({end_ms} ms)")
        prev_end_ms = end_ms
    return violations
