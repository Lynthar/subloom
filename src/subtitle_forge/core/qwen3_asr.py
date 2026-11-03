"""The Qwen3-ASR backend: Qwen3-ASR transcription, Qwen3-ForcedAligner word timing.

Model code loads lazily from the [qwen3-asr] extra, so importing this pulls in no torch."""

import gc
import importlib.util
import json
import logging
import os
import threading
import unicodedata
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..exceptions import TranscriptionError
from ..models.config import Qwen3AsrConfig
from ..models.degradation import (
    ALIGNMENT_UNSUPPORTED_LANGUAGE,
    CPU_FALLBACK,
    FLASH_ATTENTION_MISSING,
    Degradation,
)
from ..models.subtitle import SubtitleSegment, WordTiming
from .asr import TranscriptionInfo
from .hf_hub import download_repos
from .timestamp_processor import is_sentence_end

logger = logging.getLogger(__name__)

# Qwen3-ASR's language names, keyed by the codes the rest of the tool uses.
LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}
_LANGUAGE_CODES = {name: code for code, name in LANGUAGE_NAMES.items()}

# The languages Qwen3-ForcedAligner was trained to align (its model card).
ALIGNER_LANGUAGES = frozenset(
    {
        "Chinese",
        "English",
        "Cantonese",
        "French",
        "German",
        "Italian",
        "Japanese",
        "Korean",
        "Portuguese",
        "Russian",
        "Spanish",
    }
)

# Approximate download sizes in bytes, for the progress bar.
MODEL_SIZES = {
    "Qwen/Qwen3-ASR-1.7B": 4_700_000_000,
    "Qwen/Qwen3-ASR-0.6B": 1_880_000_000,
    "Qwen/Qwen3-ForcedAligner-0.6B": 1_840_000_000,
}

SAMPLE_RATE = 16000
# Chunk length, with or without FlashAttention 2. On the aligner's 180 s limit the model
# dropped whole passages (60-67% of the lines on two films went missing; 15 s: 13-23%). The
# splitter moves each cut up to 5 s to land on a quiet point, so a chunk runs to about 20 s.
CHUNK_SECONDS = 15.0
# Generation cap per chunk. A ~20 s chunk of real speech stays far below it (155 characters at
# most on two films); a chunk the model loops on stops here instead of running on for minutes.
MAX_NEW_TOKENS = 512
# A segment ends at sentence punctuation, at a pause this long, or before it outgrows
# the longest a subtitle should stay up (Netflix's 7 s).
PAUSE_SECONDS = 0.5
MAX_SEGMENT_SECONDS = 7.0
# Where a segment over the cap prefers to be cut.
CLAUSE_END_CHARS = set(",，、;；:：")

FLASH_ATTENTION_WARNING = (
    "FlashAttention 2 is unavailable (flash-attn is not installed, or the device is the CPU): "
    "Qwen3-ASR encodes audio longer than ~8 s without its windowed attention, which lowers "
    "accuracy. Install flash-attn on a CUDA machine for full accuracy."
)


def qwen_language(code: str) -> str:
    """Qwen3-ASR's name for a language code; a region suffix is ignored (zh-TW is Chinese).

    Raises:
        TranscriptionError: Qwen3-ASR does not recognise the language.
    """
    name = LANGUAGE_NAMES.get(code) or LANGUAGE_NAMES.get(code.split("-")[0].lower())
    if name is None:
        raise TranscriptionError(
            f"Qwen3-ASR does not support language {code!r}; "
            f"supported: {', '.join(sorted(LANGUAGE_NAMES))}"
        )
    return name


def _kept_by_aligner(ch: str) -> bool:
    # The aligner keeps only letters, digits and apostrophes of the transcript.
    return ch == "'" or unicodedata.category(ch)[0] in ("L", "N")


def words_from_alignment(text: str, items: List[Any], offset: float) -> List[WordTiming]:
    """Give each aligned unit its span of the transcript, punctuation and spacing included.

    The aligner strips everything but letters, digits and apostrophes, so the transcript is
    walked alongside its units; a unit that cannot be found there keeps its bare text.
    """
    words: List[WordTiming] = []
    pos = 0
    for item in items:
        i = pos
        found = True
        for ch in item.text:
            while i < len(text) and text[i] != ch and not _kept_by_aligner(text[i]):
                i += 1
            if i < len(text) and text[i] == ch:
                i += 1
            else:
                found = False
                break
        if found:
            # Trailing punctuation belongs to this unit; spacing belongs to the next one.
            while i < len(text) and not _kept_by_aligner(text[i]) and not text[i].isspace():
                i += 1
            span, pos = text[pos:i], i
        else:
            span = item.text
        words.append(
            WordTiming(
                word=span,
                start=float(item.start_time) + offset,
                end=float(item.end_time) + offset,
            )
        )
    return words


def segments_from_words(words: List[WordTiming]) -> List[SubtitleSegment]:
    """Group words into subtitle-sized segments, numbered from 1."""
    segments: List[SubtitleSegment] = []
    current: List[WordTiming] = []

    def emit(group: List[WordTiming]) -> None:
        text = "".join(w.word for w in group).strip()
        if text:
            segments.append(
                SubtitleSegment(
                    index=len(segments) + 1,
                    start=min(w.start for w in group),
                    end=max(w.end for w in group),
                    text=text,
                    words=group,
                )
            )

    for word in words:
        if current and word.start - current[-1].end >= PAUSE_SECONDS:
            emit(current)
            current = []
        elif current and word.end - current[0].start > MAX_SEGMENT_SECONDS:
            # Over the cap: cut after the last clause punctuation rather than mid-phrase.
            cut = next(
                (i + 1 for i in range(len(current) - 1, -1, -1) if _ends_clause(current[i].word)),
                len(current),
            )
            emit(current[:cut])
            current = current[cut:]
        current.append(word)
        if is_sentence_end(word.word):
            emit(current)
            current = []
    emit(current)
    return segments


def _ends_clause(word: str) -> bool:
    stripped = word.rstrip()
    return bool(stripped) and stripped[-1] in CLAUSE_END_CHARS


def _read_wav(path: Path) -> np.ndarray:
    """The extracted 16 kHz mono 16-bit WAV as float32 samples in [-1, 1]."""
    with wave.open(str(path), "rb") as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, SAMPLE_RATE):
            raise TranscriptionError(f"Expected 16 kHz mono 16-bit WAV: {path}")
        frames = f.readframes(f.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _split_into_chunks(wav: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    """(chunk, offset seconds) pieces of about CHUNK_SECONDS, cut at quiet points."""
    from qwen_asr.inference.utils import split_audio_into_chunks

    return list(split_audio_into_chunks(wav, sr=SAMPLE_RATE, max_chunk_sec=CHUNK_SECONDS))


def _snapshot_complete(repo_id: str, cache_dir: Optional[str]) -> bool:
    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(repo_id, cache_dir=cache_dir, local_files_only=True))
    except Exception:  # noqa: BLE001 - any failure here means "not usable offline"
        return False
    index = path / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        return all((path / shard).exists() for shard in shards)
    return (path / "model.safetensors").exists()


class Qwen3AsrBackend:
    """The Qwen3-ASR ASR backend; see core.asr.AsrBackend for the contract."""

    def __init__(
        self,
        *,
        model_name: str,
        aligner_model: str,
        device: str,
        batch_size: int,
        download_root: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_endpoint: Optional[str] = None,
    ):
        """Build through from_config(): the defaults live in Qwen3AsrConfig."""
        self.model_name = model_name
        self.aligner_model = aligner_model
        self.device = device
        self.batch_size = batch_size
        self.download_root = download_root
        self.hf_token = hf_token
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
            logger.info(f"Using HuggingFace mirror: {hf_endpoint}")
        self._model: Any = None
        # Fallbacks taken by the last load, reported with every transcription.
        self._load_degradations: List[Degradation] = []
        # Serializes inference, the lazy load and unloading when threads share this instance.
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: Qwen3AsrConfig, **overrides) -> "Qwen3AsrBackend":
        """Build from the `qwen3_asr` config section.

        Raises:
            TypeError: An override that is not an __init__ keyword.
        """
        kwargs: Dict[str, Any] = {
            "model_name": cfg.model,
            "aligner_model": cfg.aligner_model,
            "device": cfg.device,
            "batch_size": cfg.batch_size,
            "download_root": cfg.download_root,
            "hf_token": cfg.hf_token,
            "hf_endpoint": cfg.hf_endpoint,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def _repos(self) -> List[str]:
        return [self.model_name, self.aligner_model]

    def is_model_cached(self) -> bool:
        return all(_snapshot_complete(repo, self.download_root) for repo in self._repos)

    def get_model_size(self) -> int:
        return sum(MODEL_SIZES.get(repo, 2_000_000_000) for repo in self._repos)

    def download_model(
        self, progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        logger.info(f"Downloading Qwen3-ASR models: {', '.join(self._repos)}")
        try:
            download_repos(
                self._repos,
                cache_dir=self.download_root,
                token=self.hf_token,
                progress_callback=progress_callback,
            )
        except Exception as e:
            raise TranscriptionError(f"Failed to download Qwen3-ASR models: {e}") from e

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from huggingface_hub import snapshot_download
            from qwen_asr import Qwen3ASRModel
        except ImportError as e:
            raise TranscriptionError(
                "Qwen3-ASR needs the [qwen3-asr] extra: pip install -e '.[qwen3-asr]'"
            ) from e

        device = self.device
        degradations: List[Degradation] = []
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, running Qwen3-ASR on the CPU")
            device = "cpu"
            degradations.append(
                Degradation(CPU_FALLBACK, "CUDA is unavailable; Qwen3-ASR ran on the CPU")
            )
        kwargs: Dict[str, Any] = {
            "dtype": torch.bfloat16 if device == "cuda" else torch.float32,
            "device_map": "cuda:0" if device == "cuda" else "cpu",
        }
        if device == "cuda" and importlib.util.find_spec("flash_attn") is not None:
            kwargs["attn_implementation"] = "flash_attention_2"
        else:
            logger.warning(FLASH_ATTENTION_WARNING)
            degradations.append(
                Degradation(
                    FLASH_ATTENTION_MISSING,
                    "FlashAttention 2 is unavailable; accuracy drops on audio over ~8 s",
                )
            )

        try:
            # Local snapshot paths: qwen-asr loads its processors without our cache_dir.
            model_path, aligner_path = (
                snapshot_download(repo, cache_dir=self.download_root, token=self.hf_token)
                for repo in self._repos
            )
            logger.info(f"Loading Qwen3-ASR: {self.model_name} + {self.aligner_model} ({device})")
            self._model = Qwen3ASRModel.from_pretrained(
                model_path,
                forced_aligner=aligner_path,
                forced_aligner_kwargs=kwargs,
                max_inference_batch_size=self.batch_size,
                max_new_tokens=MAX_NEW_TOKENS,
                **kwargs,
            )
        except Exception as e:
            raise TranscriptionError(f"Failed to load Qwen3-ASR: {e}") from e
        self._load_degradations = degradations
        return self._model

    def transcribe(
        self, audio_path: Path, *, language: Optional[str] = None
    ) -> Tuple[List[SubtitleSegment], TranscriptionInfo]:
        """Transcribe audio; the AsrBackend contract in core.asr says what comes back.

        Raises:
            TranscriptionError: The audio is missing or unreadable, the language is
                unsupported, or recognition failed.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")
        forced = qwen_language(language) if language else None
        wav = _read_wav(audio_path)

        logger.info(f"Starting transcription: {audio_path.name}")
        with self._lock:
            model = self._load()
            try:
                chunks = _split_into_chunks(wav)
                results = model.transcribe(
                    audio=[(chunk, SAMPLE_RATE) for chunk, _ in chunks],
                    language=forced,
                    return_time_stamps=True,
                )
            except Exception as e:
                raise TranscriptionError(f"Qwen3-ASR transcription failed: {e}") from e
            degradations = list(self._load_degradations)

        # Segment chunk by chunk: a chunk's text joins the next one's without a space.
        segments: List[SubtitleSegment] = []
        spoken: Counter = Counter()
        for (_, offset), result in zip(chunks, results):
            if result.language:
                spoken[result.language] += len(result.text)
            if result.time_stamps is not None:
                words = words_from_alignment(result.text, result.time_stamps.items, offset)
                for seg in segments_from_words(words):
                    seg.index = len(segments) + 1
                    segments.append(seg)

        name = forced or (spoken.most_common(1)[0][0] if spoken else "")
        if name and name not in ALIGNER_LANGUAGES:
            logger.warning(
                f"Qwen3-ForcedAligner is not trained on {name}; subtitle timing may be off"
            )
            degradations.append(
                Degradation(
                    ALIGNMENT_UNSUPPORTED_LANGUAGE,
                    f"Qwen3-ForcedAligner is not trained on {name}; timing may be off",
                )
            )
        # "und" (undetermined) when nothing was said: the code still has to be a filename.
        detected = language if language else _LANGUAGE_CODES.get(name, "und")
        logger.info(f"Transcription complete: {len(segments)} segments, language: {detected}")
        info = TranscriptionInfo(
            language=detected,
            language_probability=None,
            duration=len(wav) / SAMPLE_RATE,
            degradations=degradations,
        )
        return segments, info

    def unload_model(self) -> None:
        with self._lock:
            self._model = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
