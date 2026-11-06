"""Custom exceptions for subloom."""


class SubloomError(Exception):
    """Base exception class."""

    pass


class AudioExtractionError(SubloomError):
    """Audio extraction error."""

    pass


class TranscriptionError(SubloomError):
    """Speech recognition error."""

    pass


class TranslationError(SubloomError):
    """Translation error."""

    pass


class SubtitleError(SubloomError):
    """Subtitle processing error."""

    pass


class ConfigError(SubloomError):
    """Configuration error."""

    pass


class ModelNotFoundError(SubloomError):
    """Model not found error."""

    pass
