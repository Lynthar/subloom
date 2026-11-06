"""Logging utilities."""

import logging
from pathlib import Path
from typing import List, Optional

from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_level: Optional[str] = None,
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Default level applied to the root logger and (when not
            overridden) to both the console and file handlers.
        log_file: Optional log file path. The file handler always captures
            DEBUG so the file is useful for post-mortem analysis even when
            console is quieter.
        console_level: Optional override for the console handler. Pass
            "INFO" together with level="DEBUG" to keep DEBUG out of the
            user's terminal (where Rich would render every third-party
            stack trace) while still writing it all to the log file.
    """
    handlers: List[logging.Handler] = []

    # Console handler mirrors `level` unless the caller passes a higher console_level: that keeps
    # the terminal quiet while the file still gets DEBUG (what --save-debug-log wants, since
    # torio's extension-load fallbacks render as full tracebacks).
    effective_console_level = console_level if console_level is not None else level
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False,
    )
    console_handler.setLevel(effective_console_level)
    handlers.append(console_handler)

    # File handler — always DEBUG so saved logs are diagnostic-ready.
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    # Root logger must be at the most permissive level so DEBUG actually
    # reaches the file handler when requested. Per-handler levels above
    # then filter what each destination actually shows.
    root_level = "DEBUG" if log_file else level
    logging.basicConfig(
        level=root_level,
        handlers=handlers,
        force=True,
    )

    # Silence DEBUG/INFO from third-party libraries that would otherwise dump native-extension
    # probing, alignment-model migration and HTTP transport detail into run.log with no value
    # for subtitle debugging. **Add a line here when a new noisy dependency shows up.**
    NOISY_THIRD_PARTY = (
        "faster_whisper",
        "httpx",
        "httpcore",
        "torio",  # FFmpeg extension probing fallbacks
        "torch",
        "torchaudio",
        "torio._extension",
        "speechbrain",
        "pyannote",
        "pytorch_lightning",
        "huggingface_hub",
        "urllib3",
        "matplotlib",
        "matplotlib.font_manager",
    )
    for name in NOISY_THIRD_PARTY:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
