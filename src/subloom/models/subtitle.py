"""Subtitle data model."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WordTiming:
    """Word-level timing information."""

    word: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    probability: float = 1.0  # Confidence score

    @property
    def duration(self) -> float:
        """Get word duration in seconds."""
        return self.end - self.start

    def __str__(self) -> str:
        return f"[{self.start:.2f}-{self.end:.2f}] {self.word}"


@dataclass
class SubtitleSegment:
    """A single subtitle segment with timing information."""

    index: int
    start: float  # Start time in seconds
    end: float  # End time in seconds
    text: str
    words: Optional[List[WordTiming]] = field(default=None)  # Word-level timestamps
    confidence: float = 1.0  # Segment confidence score

    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start

    def has_word_timestamps(self) -> bool:
        """Check if word-level timestamps are available."""
        return self.words is not None and len(self.words) > 0

    def get_refined_timing(self) -> tuple:
        """
        Get refined start/end times using word-level timestamps if available.

        Returns:
            Tuple of (start, end) times. Uses word timestamps if available,
            otherwise returns original segment times.
        """
        if self.words:
            return self.words[0].start, self.words[-1].end
        return self.start, self.end

    def __str__(self) -> str:
        return f"[{self.start:.2f}-{self.end:.2f}] {self.text}"
