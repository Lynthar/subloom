"""Sentences a recogniser cut at a short pause, joined back: translated apart, "見せ" and
"てみろ。" come out as "给" and "我看看。", and "聞こえ" / "ねえ" as "听" / "喂"."""

import logging
from dataclasses import replace
from typing import List, Optional

from ..models.subtitle import SubtitleSegment
from .timestamp_processor import TimestampProcessor, display_cells, is_sentence_end

logger = logging.getLogger(__name__)

# A pause on screen longer than this starts a new utterance rather than a breath inside one.
MAX_GAP_SECONDS = 1.0
# Netflix's limits for one subtitle: 7 s on screen and two lines of 16 full-width characters
# (Chinese, Japanese, Korean) or of 42 characters, in display_cells.
MAX_SECONDS = 7.0
MAX_CELLS_CJK = 64
MAX_CELLS = 84


def rejoin_fragments(
    segments: List[SubtitleSegment], language: Optional[str]
) -> List[SubtitleSegment]:
    """Join each subtitle that stops mid-sentence to the next one, renumbered from 1: only
    across a short pause and while the joined one stays within Netflix's limits, so a chain of
    fragments joins as far as those allow."""
    cjk = (language or "").lower() in TimestampProcessor.CJK_LANGUAGES
    max_cells, separator = (MAX_CELLS_CJK, "") if cjk else (MAX_CELLS, " ")
    joined: List[SubtitleSegment] = []
    for seg in segments:
        prev = joined[-1] if joined else None
        text = prev.text.rstrip() + separator + seg.text.lstrip() if prev else ""
        if (
            prev is not None
            and not _ends_sentence(prev.text)
            and seg.start - prev.end <= MAX_GAP_SECONDS
            and max(prev.end, seg.end) - prev.start <= MAX_SECONDS
            and display_cells(text) <= max_cells
        ):
            joined[-1] = replace(
                prev,
                end=max(prev.end, seg.end),
                text=text,
                words=prev.words + seg.words if prev.words and seg.words else None,
                confidence=min(prev.confidence, seg.confidence),
            )
        else:
            joined.append(seg)
    logger.info("Joined %d sentence fragments", len(segments) - len(joined))
    return [replace(seg, index=i) for i, seg in enumerate(joined, 1)]


def _ends_sentence(text: str) -> bool:
    tokens = text.split()
    return bool(tokens) and is_sentence_end(tokens[-1])
