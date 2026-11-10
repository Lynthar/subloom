"""Configuration data model."""

import logging
import os
import sys
import tempfile
from pathlib import Path
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import List, Optional, Tuple

import yaml

# Backends core.asr.create_backend can build; each one reads the config section of the same name.
ASR_BACKENDS = ("whisper", "qwen3_asr")
# Likewise for core.translator.create_translator: LLMs take a prompt, MT_BACKENDS take texts.
TRANSLATION_BACKENDS = ("ollama", "openai", "deepl", "google")
MT_BACKENDS = ("deepl", "google")

# Each is read before the section's api_key, so a key never has to be written into the config
# file. Not OPENAI_API_KEY: that one is meant for api.openai.com and would go to any base_url.
OPENAI_API_KEY_ENV = "SUBLOOM_OPENAI_API_KEY"
DEEPL_API_KEY_ENV = "SUBLOOM_DEEPL_API_KEY"
GOOGLE_API_KEY_ENV = "SUBLOOM_GOOGLE_API_KEY"

# Keys the ollama section held before translation settings became shared by every backend.
_MOVED_TO_TRANSLATION = (
    "temperature",
    "max_batch_size",
    "max_retries",
    "request_timeout",
    "prompt_template_id",
)

logger = logging.getLogger(__name__)


@dataclass
class AsrConfig:
    """Speech recognition backend selection."""

    backend: str = "whisper"  # One of ASR_BACKENDS


@dataclass
class TimestampConfig:
    """Timestamp post-processing configuration."""

    enabled: bool = True  # Enable timestamp post-processing
    mode: str = "minimal"  # Processing mode: "off", "minimal", "full"
    min_duration: float = 1.0  # Minimum subtitle duration (seconds)
    max_duration: float = 8.0  # Maximum subtitle duration (seconds)
    min_gap: float = 0.05  # Minimum gap between subtitles (seconds)
    max_gap_warning: float = 10.0  # Gap threshold for missed speech warning (seconds)
    chars_per_second: float = 15.0  # Reading speed for Western languages
    cjk_chars_per_second: float = 10.0  # Reading speed for CJK languages
    split_threshold: int = 30  # Minimum characters before attempting split
    split_sentences: bool = True  # Split segments by sentence using word timestamps
    # Lead-in and linger to compensate for acoustic-alignment timestamps not being
    # the same thing as on-screen subtitle timing. Without these, subtitles tend
    # to feel like they "chase" the audio and disappear too fast.
    lead_in_ms: int = 80  # Show subtitle this many ms BEFORE the first word's onset
    linger_ms: int = 300  # Keep subtitle this many ms AFTER the last word's offset


@dataclass
class WhisperConfig:
    """Whisper transcription configuration."""

    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True
    batch_size: Optional[int] = None
    download_root: Optional[str] = None
    # VAD tuned for subtitle timing: 250ms padding (silero default 400) keeps subtitles tight
    # without eating word onsets; 700ms min silence avoids merging adjacent utterances in fast
    # dialogue while still bridging natural breath pauses.
    speech_pad_ms: int = 250
    min_silence_duration_ms: int = 700
    # WhisperX options
    use_whisperx: bool = True  # Use WhisperX for better timestamp accuracy
    whisperx_align: bool = True  # Enable forced alignment with wav2vec2
    hf_token: Optional[str] = None  # HuggingFace token for gated/private model downloads
    # HuggingFace mirror for users in regions with limited access
    hf_endpoint: Optional[str] = None  # e.g., "https://hf-mirror.com"


@dataclass
class Qwen3AsrConfig:
    """Qwen3-ASR transcription configuration (the [qwen3-asr] extra)."""

    model: str = "Qwen/Qwen3-ASR-1.7B"
    # Supplies the word timing; without it Qwen3-ASR has no timestamps at all.
    aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    device: str = "cuda"
    batch_size: int = 4  # ~15 s chunks (~8 s for Japanese) per inference batch; lower on small GPUs
    download_root: Optional[str] = None
    hf_token: Optional[str] = None
    hf_endpoint: Optional[str] = None


@dataclass
class TranslationConfig:
    """Translation settings shared across backends; the backend's own section holds the rest."""

    backend: str = "ollama"  # One of TRANSLATION_BACKENDS
    max_retries: int = 3
    request_timeout: float = 180.0  # Per-request timeout (seconds)
    # The rest apply to the LLM backends (ollama, openai) only.
    # Translation is a deterministic task — temperature=0 dramatically reduces
    # the rate of skipped/renumbered segments compared to 0.3.
    temperature: float = 0.0
    max_batch_size: int = 10
    prompt_template_id: Optional[str] = None  # Prompt library template (None = built-in prompt)
    # Character names as the translation must write them, e.g. ["光希", "琴音"]
    names: List[str] = field(default_factory=list)
    think: bool = False  # Let the model reason before each translation batch (ollama only)
    find_names: bool = True  # Read the whole transcript first for the names and their spellings


@dataclass
class OllamaConfig:
    """Ollama server (translation.backend: ollama)."""

    model: str = "qwen2.5:14b"
    host: str = "http://localhost:11434"


@dataclass
class OpenAIConfig:
    """OpenAI-compatible chat API (translation.backend: openai), local or cloud."""

    # No default: an unset URL must fail validation, never reach a cloud service.
    base_url: str = ""
    model: str = ""
    api_key: Optional[str] = None  # OPENAI_API_KEY_ENV overrides it


@dataclass
class DeepLConfig:
    """DeepL API (translation.backend: deepl); a key ending in ":fx" uses the Free endpoint."""

    api_key: Optional[str] = None  # DEEPL_API_KEY_ENV overrides it


@dataclass
class GoogleConfig:
    """Google Cloud Translation Basic, v2 (translation.backend: google)."""

    api_key: Optional[str] = None  # GOOGLE_API_KEY_ENV overrides it


@dataclass
class OutputConfig:
    """Output configuration."""

    encoding: str = "utf-8"
    keep_original: bool = True
    bilingual: bool = False
    original_on_top: bool = True
    drop_interjections: bool = False  # Drop interjection-only subtitles, before translating
    merge_fragments: bool = True  # Join subtitles cut mid-sentence, before translating

    def resolve_flags(
        self, *, keep_original: Optional[bool], bilingual: Optional[bool]
    ) -> Tuple[bool, bool]:
        """(keep_original, bilingual) with None meaning "as configured". Every entry path
        resolves through here, so the output section applies to all of them."""
        return (
            self.keep_original if keep_original is None else keep_original,
            self.bilingual if bilingual is None else bilingual,
        )


def _migrate_ollama_keys(data: dict) -> List[str]:
    """Move a pre-translation-section file's ollama keys into ``data``'s translation section.

    Returns the keys moved. A key already set under translation wins. A custom prompt body
    becomes the user library template ``custom``, as `config set-prompt` now stores one.

    Raises:
        ValueError: The body cannot be stored without replacing a different ``custom``.
    """
    ollama = data.get("ollama") or {}
    moved = [k for k in (*_MOVED_TO_TRANSLATION, "prompt_template") if k in ollama]
    if not moved:
        return []
    translation = dict(data.get("translation") or {})
    explicit = set(translation)
    for key in _MOVED_TO_TRANSLATION:
        value = ollama.pop(key, None)
        if key in moved and key not in explicit:
            translation[key] = value
    body = ollama.pop("prompt_template", None)
    if body and "prompt_template_id" not in explicit:
        from ..core.prompt_library import store_custom_template

        # The body took precedence over a library id, so it still does.
        translation["prompt_template_id"] = store_custom_template(body, replace=False)
    data["ollama"], data["translation"] = ollama, translation
    return moved


@dataclass
class AppConfig:
    """Application configuration."""

    asr: AsrConfig = field(default_factory=AsrConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    qwen3_asr: Qwen3AsrConfig = field(default_factory=Qwen3AsrConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    deepl: DeepLConfig = field(default_factory=DeepLConfig)
    google: GoogleConfig = field(default_factory=GoogleConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    timestamp: TimestampConfig = field(default_factory=TimestampConfig)
    max_workers: int = 2
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def get_config_path(cls) -> Path:
        """Get user config file path.

        Windows uses the %APPDATA% location the user guide documents;
        everything else uses ~/.config. A directory under the project's old name stays in use
        until a new one exists, so an upgrade keeps its settings and prompt templates.
        """
        appdata = os.environ.get("APPDATA") if sys.platform == "win32" else None
        root = Path(appdata) if appdata else Path.home() / ".config"
        directory, old = root / "subloom", root / "subtitle-forge"
        return (old if old.is_dir() and not directory.exists() else directory) / "config.yaml"

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        """Load configuration from YAML file."""
        if path is None:
            path = cls.get_config_path()

        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        moved = _migrate_ollama_keys(data)
        if moved:
            logger.warning(
                f"{path}: moved {', '.join(f'ollama.{k}' for k in moved)} to the translation "
                f"section; save the new layout with: subloom config import {path}"
            )

        kwargs: dict = {}
        for fld in fields(cls):
            section = fld.type
            if isinstance(section, type) and is_dataclass(section):
                kwargs[fld.name] = section(**data.get(fld.name, {}))
            elif fld.name in data:
                kwargs[fld.name] = data[fld.name]
        config = cls(**kwargs)
        config.validate()
        return config

    def validate(self) -> None:
        """Raise ValueError listing every invalid field value.

        Called from load() (and by `config set` before saving) so a bad value
        fails before any work starts — chars_per_second=0 used to save fine
        and surface as a ZeroDivisionError mid-transcription.
        """

        def _num(value) -> bool:
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        ts, wh, tr = self.timestamp, self.whisper, self.translation
        errors = []

        if self.asr.backend not in ASR_BACKENDS:
            errors.append(
                f"asr.backend must be one of {'/'.join(ASR_BACKENDS)}, got {self.asr.backend!r}"
            )
        if tr.backend not in TRANSLATION_BACKENDS:
            errors.append(
                f"translation.backend must be one of {'/'.join(TRANSLATION_BACKENDS)}, "
                f"got {tr.backend!r}"
            )
        if tr.backend == "openai":
            if not self.openai.base_url.startswith(("http://", "https://")):
                errors.append(
                    f"openai.base_url must be an http(s) URL such as http://localhost:1234/v1, "
                    f"got {self.openai.base_url!r}"
                )
            if not self.openai.model:
                errors.append("openai.model must name the model to translate with")
        if not isinstance(tr.names, list) or not all(isinstance(n, str) for n in tr.names):
            errors.append(f"translation.names must be a list of names, got {tr.names!r}")
        elif tr.names and tr.backend in MT_BACKENDS:
            errors.append(
                f"translation.names needs an LLM backend (ollama, openai); {tr.backend} takes no "
                "name list"
            )
        if ts.mode not in ("off", "minimal", "full"):
            errors.append(f"timestamp.mode must be one of off/minimal/full, got {ts.mode!r}")
        if wh.device not in ("cuda", "cpu", "auto"):
            errors.append(f"whisper.device must be one of cuda/cpu/auto, got {wh.device!r}")
        if self.qwen3_asr.device not in ("cuda", "cpu"):
            errors.append(
                f"qwen3_asr.device must be one of cuda/cpu, got {self.qwen3_asr.device!r}"
            )
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append(
                f"log_level must be one of DEBUG/INFO/WARNING/ERROR/CRITICAL, got {self.log_level!r}"
            )

        positive = [
            ("timestamp.min_duration", ts.min_duration),
            ("timestamp.max_duration", ts.max_duration),
            ("timestamp.max_gap_warning", ts.max_gap_warning),
            ("timestamp.chars_per_second", ts.chars_per_second),
            ("timestamp.cjk_chars_per_second", ts.cjk_chars_per_second),
            ("translation.request_timeout", tr.request_timeout),
        ]
        non_negative = [
            ("timestamp.min_gap", ts.min_gap),
            ("timestamp.lead_in_ms", ts.lead_in_ms),
            ("timestamp.linger_ms", ts.linger_ms),
            ("whisper.speech_pad_ms", wh.speech_pad_ms),
            ("whisper.min_silence_duration_ms", wh.min_silence_duration_ms),
            ("translation.temperature", tr.temperature),
        ]
        # max_retries >= 1 matters: the translator's attempt loop is
        # `range(max_retries)`, so 0 would silently skip translation entirely.
        at_least_one = [
            ("timestamp.split_threshold", ts.split_threshold),
            ("whisper.beam_size", wh.beam_size),
            ("qwen3_asr.batch_size", self.qwen3_asr.batch_size),
            ("translation.max_batch_size", tr.max_batch_size),
            ("translation.max_retries", tr.max_retries),
            ("max_workers", self.max_workers),
        ]
        for name, value in positive:
            if not _num(value) or value <= 0:
                errors.append(f"{name} must be a number > 0, got {value!r}")
        for name, value in non_negative:
            if not _num(value) or value < 0:
                errors.append(f"{name} must be a number >= 0, got {value!r}")
        for name, value in at_least_one:
            if not _num(value) or value < 1:
                errors.append(f"{name} must be an integer >= 1, got {value!r}")
        if wh.batch_size is not None and (not _num(wh.batch_size) or wh.batch_size < 1):
            errors.append(
                f"whisper.batch_size must be an integer >= 1 or null, got {wh.batch_size!r}"
            )
        if _num(ts.min_duration) and _num(ts.max_duration) and ts.max_duration < ts.min_duration:
            errors.append(
                f"timestamp.max_duration ({ts.max_duration}) must be >= "
                f"timestamp.min_duration ({ts.min_duration})"
            )

        if errors:
            raise ValueError("Invalid configuration:\n  " + "\n  ".join(errors))

    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to YAML file.

        Serializes EVERY dataclass field via ``asdict`` so hand-edited values
        survive a rewrite. The previous version used a hand-maintained whitelist
        that silently omitted several fields (e.g. ``ollama.request_timeout`` /
        ``max_retries``, ``whisper.batch_size`` / ``download_root``), so running
        any ``config set`` erased them and reverted them to defaults — a
        data-loss bug. Field order matches the dataclass definitions.
        """
        if path is None:
            path = self.get_config_path()

        path.parent.mkdir(parents=True, exist_ok=True)

        content = yaml.safe_dump(
            asdict(self), default_flow_style=False, allow_unicode=True, sort_keys=False
        )

        # Same-directory temp file, then replace: a crash mid-write must not
        # leave a truncated config. 0600 because the file can hold hf_token in
        # plaintext, and the umask default 0644 is world-readable on POSIX.
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, str(path))
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
