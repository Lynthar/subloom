"""The speech recognition backend contract, and the factory every entry path builds one through.

Kept free of model imports: backends load lazily, so importing this pulls in no torch."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Tuple

from ..models.config import AppConfig
from ..models.degradation import Degradation
from ..models.subtitle import SubtitleSegment


@dataclass
class TranscriptionInfo:
    """Transcription metadata."""

    language: str  # Code of the transcribed language, e.g. "en"
    language_probability: Optional[float]  # None when the backend reports no confidence
    duration: float  # Audio length in seconds; 0.0 when unknown
    # Every fallback this run took, standing ones (a missing package, no GPU) included.
    degradations: List[Degradation] = field(default_factory=list)


class AsrBackend(Protocol):
    """What the pipeline and the entry paths need from a speech recognition backend."""

    model_name: str  # The configured model, for messages

    def transcribe(
        self, audio_path: Path, *, language: Optional[str] = None
    ) -> Tuple[List[SubtitleSegment], TranscriptionInfo]:
        """Transcribe a 16 kHz mono WAV.

        Args:
            audio_path: The extracted audio file.
            language: Source language code; None asks the backend to detect it.

        Returns:
            Chronological segments numbered from 1, carrying ``words`` when the backend
            has word timing, and the run's metadata. Timing is raw: the caller runs the
            timestamp post-processing, so a backend that applies it too applies it twice.
            Every fallback goes in ``degradations``; one only logged reaches no result.

        Raises:
            TranscriptionError: Recognition failed.

        One instance may be shared by threads (batch does this); a backend serializes
        its own inference.
        """
        ...

    def is_model_cached(self) -> bool:
        """True when the model is on disk, so transcribe() will not download."""
        ...

    def get_model_size(self) -> int:
        """Approximate download size in bytes, for the progress bar."""
        ...

    def download_model(
        self, progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Download the model, reporting (downloaded_bytes, total_bytes); total 0 is unknown."""
        ...

    def unload_model(self) -> None:
        """Release the model and its memory; the next transcribe() loads it again.

        Waits for a transcription in progress on another thread, so a shared instance
        is never unloaded mid-run.
        """
        ...


def create_backend(config: AppConfig, **overrides) -> AsrBackend:
    """Build the backend ``config.asr.backend`` names, from its config section.

    Args:
        config: The application config.
        **overrides: Constructor keywords for the selected backend that replace their
            configured values (e.g. model_name from --model).

    Raises:
        ValueError: asr.backend names no known backend.
        TypeError: An override the selected backend does not take.
    """
    if config.asr.backend == "whisper":
        from .transcriber import Transcriber

        return Transcriber.from_config(config.whisper, **overrides)
    if config.asr.backend == "qwen3_asr":
        from .qwen3_asr import Qwen3AsrBackend

        return Qwen3AsrBackend.from_config(config.qwen3_asr, **overrides)
    raise ValueError(f"Unknown ASR backend {config.asr.backend!r}")
