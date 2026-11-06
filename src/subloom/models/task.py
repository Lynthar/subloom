"""Task data model."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

from .degradation import Degradation


class TaskStatus(Enum):
    """Task status enum."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class VideoTask:
    """Video processing task."""

    video_path: Path
    target_langs: List[str]
    output_dir: Path
    options: Dict[str, Any] = field(default_factory=dict)

    status: TaskStatus = TaskStatus.PENDING
    source_lang: Optional[str] = None
    error: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Output file paths
    original_srt: Optional[Path] = None
    translated_srts: Dict[str, Path] = field(default_factory=dict)
    degradations: List[Degradation] = field(default_factory=list)

    @property
    def elapsed_time(self) -> Optional[float]:
        """Get elapsed time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
