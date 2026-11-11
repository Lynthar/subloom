"""Speech recognition module using faster-whisper and WhisperX."""

# CRITICAL: set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD before any import that may load torch
# (faster_whisper included). PyTorch 2.6 flipped torch.load()'s weights_only default, which
# breaks pyannote-audio models carrying omegaconf objects. **Never reorder these imports.**
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
# Without this transformers starts a thread on every online load of a repo without safetensors:
# it may ask a third-party Space to convert the repo, downloads weights it never loads, and
# keeps the process from exiting until done.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")
# pyannote, which WhisperX loads for VAD, reports usage to otel.pyannote.ai unless told not to,
# and ignores DO_NOT_TRACK. A value the user already set wins, so it can still be turned on.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Callable
import logging
import math
import threading

import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline, decode_audio
from faster_whisper.tokenizer import Tokenizer

from ..models.config import WhisperConfig
from ..models.degradation import (
    ALIGNMENT_FAILED,
    CPU_FALLBACK,
    WHISPERX_MISSING,
    Degradation,
)
from ..models.subtitle import SubtitleSegment, WordTiming
from ..utils.gpu import get_optimal_compute_type, get_available_vram
from ..exceptions import TranscriptionError
from .asr import TranscriptionInfo
from .hallucinations import is_hallucination
from .hf_hub import complete_snapshot, download_repos
from .timestamp_processor import TimestampProcessor

logger = logging.getLogger(__name__)

# WhisperX makes each VAD chunk one segment, and its sentence splitter has no rules for Chinese,
# Japanese or Korean, so there a chunk stays one subtitle. 7 s keeps it in Netflix's limit.
WHISPERX_CHUNK_SECONDS = 20
WHISPERX_CJK_CHUNK_SECONDS = 7

# The [whisperx] extra is optional. This binding is unbound when the import fails,
# so every use of `whisperx` below must stay behind `self.use_whisperx`.
WHISPERX_AVAILABLE = False
try:
    import whisperx

    WHISPERX_AVAILABLE = True
except ImportError:
    pass


def _should_use_whisperx(use_whisperx: bool) -> bool:
    """
    Determine if WhisperX should be used based on availability.

    Args:
        use_whisperx: User's preference for using WhisperX.

    Returns:
        True if WhisperX should be used.
    """
    if not use_whisperx:
        return False

    if not WHISPERX_AVAILABLE:
        return False

    return True


def _log_hallucinations(count: int) -> None:
    if count:
        logger.info(f"Dropped {count} lines Whisper made up over audio without speech")


def _align_model_cached(language: str) -> bool:
    """Whether WhisperX's HuggingFace alignment model for the language is on disk; never online.
    WhisperX loads it from the default HF cache, so that is where this looks."""
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF

    repo = DEFAULT_ALIGN_MODELS_HF.get(language)
    return repo is not None and complete_snapshot(repo, None) is not None


# Model name to HuggingFace repo: must be the repo faster-whisper loads for that name, or a
# model we report as cached is not where it looks with local_files_only.
WHISPER_HF_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    # Japanese only, so it stays out of MODEL_VRAM_REQUIREMENTS and is never picked for you.
    "kotoba-whisper-v2.0": "kotoba-tech/kotoba-whisper-v2.0-faster",
}

# Repos whose word timing points at decoder layers they lack, which kills the process in plain
# faster-whisper. WhisperX times words with its own aligner, so they run only under it.
WHISPERX_ONLY_REPOS = {"kotoba-tech/kotoba-whisper-v2.0-faster"}

# Approximate model sizes in bytes for progress display
WHISPER_MODEL_SIZES = {
    "tiny": 75_000_000,  # ~75MB
    "tiny.en": 75_000_000,
    "base": 145_000_000,  # ~145MB
    "base.en": 145_000_000,
    "small": 488_000_000,  # ~488MB
    "small.en": 488_000_000,
    "medium": 1_530_000_000,  # ~1.5GB
    "medium.en": 1_530_000_000,
    "large-v1": 3_100_000_000,  # ~3.1GB
    "large-v2": 3_100_000_000,
    "large-v3": 3_100_000_000,
    "large-v3-turbo": 1_600_000_000,  # ~1.6GB
    "distil-large-v2": 1_500_000_000,
    "distil-large-v3": 1_500_000_000,
    "kotoba-whisper-v2.0": 1_510_000_000,
}


class Transcriber:
    """The Whisper ASR backend: faster-whisper, or WhisperX with forced alignment."""

    # VRAM in MB. Ties are broken by insertion order (stable sort in
    # select_optimal_model), so large-v3 must precede large-v2 — reversed,
    # every GPU with the VRAM for either gets recommended the older model.
    MODEL_VRAM_REQUIREMENTS = {
        "tiny": 1000,
        "base": 1500,
        "small": 2500,
        "medium": 5000,
        "large-v3": 6000,
        "large-v2": 6000,
        "large-v3-turbo": 3000,
        "distil-large-v2": 4000,
        "distil-large-v3": 4000,
    }

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        use_whisperx: bool,
        whisperx_align: bool,
        compute_type: Optional[str] = None,
        download_root: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_endpoint: Optional[str] = None,
        beam_size: int = WhisperConfig.beam_size,
        vad_filter: bool = WhisperConfig.vad_filter,
        batch_size: Optional[int] = WhisperConfig.batch_size,
        vad_parameters: Optional[dict] = None,
    ):
        """
        Initialize transcriber. Build through from_config(): the defaults live in WhisperConfig.

        Args:
            model_name: Whisper model name.
            device: Device type (cuda/cpu).
            use_whisperx: Use WhisperX for better timestamp accuracy.
            whisperx_align: Enable forced alignment with wav2vec2.
            compute_type: Compute precision (float16/int8_float16/int8); None picks per device.
            download_root: Model download directory.
            hf_token: HuggingFace token, passed to model downloads (needed
                only for gated/private repos).
            hf_endpoint: HuggingFace mirror endpoint (e.g., "https://hf-mirror.com").
            beam_size: Beam search size.
            vad_filter: Enable faster-whisper's VAD filter.
            batch_size: >1 runs faster-whisper's BatchedInferencePipeline; WhisperX defaults to 8.
            vad_parameters: silero VAD tuning; None uses DEFAULT_VAD_PARAMETERS.

        Raises:
            ValueError: The model runs only under WhisperX, which is off or not installed.
        """
        self.model_name = model_name
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.batch_size = batch_size
        self.vad_parameters = vad_parameters
        self.device = device
        self.compute_type = compute_type or get_optimal_compute_type(device)
        self.download_root = download_root
        self.whisperx_align = whisperx_align
        self.hf_token = hf_token
        self.hf_endpoint = hf_endpoint

        # Set HuggingFace mirror endpoint if provided
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
            logger.info(f"Using HuggingFace mirror: {hf_endpoint}")

        # Determine if WhisperX should be used
        self.use_whisperx = _should_use_whisperx(use_whisperx)
        if self.repo_id in WHISPERX_ONLY_REPOS and not self.use_whisperx:
            raise ValueError(
                f"{model_name} runs only under WhisperX: install the [whisperx] extra "
                "and leave use_whisperx on (no --no-whisperx)"
            )

        # Fallbacks fixed at construction, reported with every transcription.
        self._standing_degradations: List[Degradation] = []
        if use_whisperx and not WHISPERX_AVAILABLE:
            logger.warning(
                "WhisperX not available. Install with: pip install whisperx. "
                "Falling back to faster-whisper."
            )
            self._standing_degradations.append(
                Degradation(
                    WHISPERX_MISSING,
                    "WhisperX is not installed; timing comes from plain faster-whisper",
                )
            )

        self._model: Optional[WhisperModel] = None
        self._batched_pipeline: Optional[BatchedInferencePipeline] = None
        self._whisperx_model: Any = None
        self._whisperx_align_model = None
        self._whisperx_metadata = None
        # The align model is language-specific; remember which language the
        # cached one was loaded for so we can reload when it changes.
        self._whisperx_align_language: Optional[str] = None
        # Warn only once per instance that VAD tuning doesn't apply under WhisperX.
        self._whisperx_vad_warned = False
        # Serializes transcription and unloading when one Transcriber is shared across threads
        # (batch does this): inference and the lazy model loads are not concurrency-safe on one
        # instance, and an unload mid-run would pull the models out from under it.
        self._transcribe_lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: WhisperConfig, **overrides) -> "Transcriber":
        """Build a Transcriber from the `whisper` config section — every one, not
        just some: a partial build looks in the wrong cache directory and ignores
        the configured mirror, so the model gets downloaded a second time.

        Args:
            cfg: The whisper section of AppConfig.
            **overrides: Any __init__ keyword whose configured value the
                caller replaces (e.g. model_name from --model).

        Raises:
            TypeError: An override that is not an __init__ keyword.
            ValueError: The model runs only under WhisperX, which is off or not installed.
        """
        kwargs: Dict[str, Any] = {
            "model_name": cfg.model,
            "device": cfg.device,
            "compute_type": cfg.compute_type,
            "download_root": cfg.download_root,
            "use_whisperx": cfg.use_whisperx,
            "whisperx_align": cfg.whisperx_align,
            "hf_token": cfg.hf_token,
            "hf_endpoint": cfg.hf_endpoint,
            "beam_size": cfg.beam_size,
            "vad_filter": cfg.vad_filter,
            "batch_size": cfg.batch_size,
            "vad_parameters": build_vad_parameters(cfg),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def select_optimal_model(cls, prefer_large: bool = True) -> Optional[str]:
        """
        Select optimal model based on available VRAM.

        Args:
            prefer_large: Prefer larger models if VRAM allows.

        Returns:
            Recommended model name, or None when free VRAM cannot be read.
        """
        available_vram = get_available_vram()

        if available_vram <= 0:
            logger.warning("Cannot detect free GPU VRAM, no model recommendation")
            return None

        # Sort models by VRAM requirement
        sorted_models = sorted(
            cls.MODEL_VRAM_REQUIREMENTS.items(),
            key=lambda x: x[1],
            reverse=prefer_large,
        )

        for model_name, required_vram in sorted_models:
            if available_vram >= required_vram * 1.2:  # 20% headroom
                logger.info(
                    f"Selected model: {model_name} "
                    f"(requires {required_vram}MB, available {available_vram}MB)"
                )
                return model_name

        return "tiny"

    @property
    def repo_id(self) -> str:
        """The HuggingFace repo the model loads from: a known name's, else the name itself."""
        return WHISPER_HF_REPOS.get(self.model_name, self.model_name)

    def is_model_cached(self) -> bool:
        """
        Check if Whisper model is already downloaded.

        Returns:
            True if model files exist locally.
        """
        try:
            from huggingface_hub import try_to_load_from_cache

            # Import _CACHED_NO_EXIST separately as it might not exist in older versions
            try:
                from huggingface_hub import _CACHED_NO_EXIST
            except ImportError:
                _CACHED_NO_EXIST = None

            repo_id = self.repo_id

            # Check for the main model file
            result = try_to_load_from_cache(
                repo_id,
                "model.bin",
                cache_dir=self.download_root,
            )

            # If result is a string path, the file is cached
            if isinstance(result, str):
                return True

            # If _CACHED_NO_EXIST exists and result equals it, file definitely doesn't exist
            if _CACHED_NO_EXIST is not None and result is _CACHED_NO_EXIST:
                return False

            # Otherwise (result is None), file is not cached
            return False

        except Exception as e:
            logger.debug(f"Could not check model cache: {e}")
            # If we can't check, assume not cached
            return False

    def get_model_size(self) -> int:
        """Get approximate model size in bytes."""
        return WHISPER_MODEL_SIZES.get(self.model_name, 1_000_000_000)

    def download_model(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Download Whisper model with progress tracking.

        Args:
            progress_callback: Callback(downloaded_bytes, total_bytes) for progress updates.
        """
        repo_id = self.repo_id

        logger.info(f"Downloading Whisper model: {self.model_name} from {repo_id}")

        try:
            download_repos(
                [repo_id],
                cache_dir=self.download_root,
                token=self.hf_token,
                progress_callback=progress_callback,
            )
            logger.info(f"Model {self.model_name} downloaded successfully")

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise TranscriptionError(f"Failed to download Whisper model: {e}") from e

    def load_model(self) -> WhisperModel:
        """Load the Whisper model on first use and return it."""
        if self._model is not None:
            return self._model

        logger.info(f"Loading Whisper model: {self.model_name} ({self.compute_type})...")

        try:
            # A cached model must not ask the Hub: an SSL or proxy error there fails the load
            # with every file on disk, where other network errors fall back to the cache.
            self._model = WhisperModel(
                self.repo_id,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
                local_files_only=self.is_model_cached(),
            )
        except Exception as e:
            raise TranscriptionError(f"Failed to load model: {e}") from e

        logger.info("Model loaded successfully")
        return self._model

    # Values derive from WhisperConfig (rationale there). VAD sets where speech is cut; on-screen
    # feel comes from lead-in/linger — keep the two apart.
    DEFAULT_VAD_PARAMETERS = {
        "speech_pad_ms": WhisperConfig.speech_pad_ms,
        "min_silence_duration_ms": WhisperConfig.min_silence_duration_ms,
    }

    # Without VAD, Whisper decodes music and breathing too. Not conditioning on its own previous
    # text, and skipping hallucinations around silence, stop one invented line repeating for
    # minutes.
    NO_VAD_OPTIONS = {"condition_on_previous_text": False, "hallucination_silence_threshold": 2.0}

    # Preset VAD modes for different use cases
    VAD_PRESETS = {
        "default": DEFAULT_VAD_PARAMETERS,
        "aggressive": {
            # Snappier, may clip word edges — useful for very fast dialogue
            "speech_pad_ms": 150,
            "min_silence_duration_ms": 400,
        },
        "relaxed": {
            # More forgiving, longer segments — good for narration / docs
            "speech_pad_ms": 400,
            "min_silence_duration_ms": 1000,
        },
        "precise": {
            # Maximum boundary tightness, risks clipping
            "speech_pad_ms": 100,
            "min_silence_duration_ms": 300,
        },
    }

    @classmethod
    def get_vad_parameters(
        cls,
        mode: Optional[str] = None,
        speech_pad_ms: Optional[int] = None,
        min_silence_duration_ms: Optional[int] = None,
    ) -> dict:
        """
        Get VAD parameters based on mode or custom values.

        Args:
            mode: Preset mode name ("default", "aggressive", "relaxed", "precise").
            speech_pad_ms: Custom speech padding (overrides mode).
            min_silence_duration_ms: Custom min silence duration (overrides mode).

        Returns:
            VAD parameters dictionary.
        """
        # Start with default or mode preset
        if mode and mode in cls.VAD_PRESETS:
            params = cls.VAD_PRESETS[mode].copy()
        else:
            params = cls.DEFAULT_VAD_PARAMETERS.copy()

        # Override with custom values if provided
        if speech_pad_ms is not None:
            params["speech_pad_ms"] = speech_pad_ms
        if min_silence_duration_ms is not None:
            params["min_silence_duration_ms"] = min_silence_duration_ms

        return params

    def transcribe(
        self, audio_path: Path, *, language: Optional[str] = None
    ) -> Tuple[List[SubtitleSegment], TranscriptionInfo]:
        """Transcribe audio; the AsrBackend contract in core.asr says what comes back.

        Raises:
            TranscriptionError: The audio file is missing or recognition failed.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        logger.info(f"Starting transcription: {audio_path.name}")

        # Serialize inference + lazy model loads across threads sharing this
        # instance (see _transcribe_lock).
        with self._transcribe_lock:
            if self.use_whisperx:
                segments, info = self._transcribe_whisperx(audio_path, language=language)
            else:
                segments, info = self._transcribe_faster_whisper(audio_path, language=language)
        info.degradations = [*self._standing_degradations, *info.degradations]
        return segments, info

    def _transcribe_whisperx(
        self, audio_path: Path, *, language: Optional[str]
    ) -> Tuple[List[SubtitleSegment], TranscriptionInfo]:
        """Transcribe using WhisperX with forced alignment."""
        import torch

        logger.debug("Using WhisperX for transcription with forced alignment")

        # WhisperX uses pyannote-based VAD (vad_onset/vad_offset), so the
        # silero-style speech_pad_ms / min_silence_duration_ms tuning does not
        # translate. Warn once instead of silently discarding it.
        if self.vad_parameters and not self._whisperx_vad_warned:
            logger.warning(
                "VAD tuning (speech_pad_ms / min_silence_duration_ms) is ignored "
                "under WhisperX, which uses pyannote VAD. Run with --no-whisperx "
                "to use faster-whisper if you need those parameters."
            )
            self._whisperx_vad_warned = True

        try:
            # When falling back to CPU the compute type must follow: CTranslate2 rejects float16 on
            # CPU ("float16 computation is not supported"), so keeping the configured GPU compute
            # type would make the fallback fail anyway.
            device = self.device
            compute_type = self.compute_type
            degradations: List[Degradation] = []
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                compute_type = get_optimal_compute_type("cpu")
                logger.warning(
                    f"CUDA not available, using CPU for WhisperX (compute_type={compute_type})"
                )
                degradations.append(
                    Degradation(CPU_FALLBACK, "CUDA is unavailable; WhisperX ran on the CPU")
                )

            if self._whisperx_model is None:
                logger.debug(f"Loading WhisperX model: {self.model_name}")
                # beam_size is a WhisperX ASR option, not a transcribe() kwarg: left out of
                # asr_options, the configured beam size is silently dropped. A cached model must
                # not ask the Hub, as in load_model.
                self._whisperx_model = whisperx.load_model(
                    self.repo_id,
                    device=device,
                    compute_type=compute_type,
                    download_root=self.download_root,
                    asr_options={"beam_size": self.beam_size},
                    local_files_only=self.is_model_cached(),
                )

            # Load audio
            audio = whisperx.load_audio(str(audio_path))

            # WhisperX would judge the language inside transcribe, too late for the chunk size, so
            # judge it here the same way (the first 30 s) and hand it over.
            if not language:
                language = self._whisperx_model.detect_language(audio)

            # batch_size honours config; 8 is only a default.
            cjk = language.lower() in TimestampProcessor.CJK_LANGUAGES
            result = self._whisperx_model.transcribe(
                audio,
                language=language,
                batch_size=self.batch_size or 8,
                chunk_size=WHISPERX_CJK_CHUNK_SECONDS if cjk else WHISPERX_CHUNK_SECONDS,
            )

            detected_language = result.get("language", language)
            audio_duration = len(audio) / 16000  # WhisperX uses 16kHz
            result["segments"] = self._drop_hallucinated_chunks(
                result.get("segments", []), audio, detected_language
            )

            # Forced alignment for word-level timestamps
            if self.whisperx_align and result.get("segments"):
                logger.debug("Performing forced alignment with wav2vec2")
                try:
                    # Reload the alignment model when the detected language changes: these models
                    # are language-specific, so a long-lived Transcriber aligns later jobs with the
                    # first one's model.
                    if (
                        self._whisperx_align_model is None
                        or self._whisperx_metadata is None
                        or self._whisperx_align_language != detected_language
                    ):
                        self._whisperx_align_model, self._whisperx_metadata = (
                            whisperx.load_align_model(
                                language_code=detected_language,
                                device=device,
                                model_cache_only=_align_model_cached(detected_language),
                            )
                        )
                        self._whisperx_align_language = detected_language

                    # Align
                    result = whisperx.align(
                        result["segments"],
                        self._whisperx_align_model,
                        self._whisperx_metadata,
                        audio,
                        device,
                        return_char_alignments=False,
                    )
                except Exception as e:
                    logger.warning(f"Forced alignment failed: {e}. Using original timestamps.")
                    degradations.append(
                        Degradation(
                            ALIGNMENT_FAILED,
                            f"Forced alignment failed ({e}); timestamps are segment-level",
                        )
                    )

            # Convert to SubtitleSegments
            segments = []
            for i, seg in enumerate(result.get("segments", [])):
                # Extract word-level timestamps if available
                words = None
                if "words" in seg:
                    words = [
                        WordTiming(
                            word=w.get("word", ""),
                            start=w.get("start", 0),
                            end=w.get("end", 0),
                            probability=w.get("score", 1.0),
                        )
                        for w in seg["words"]
                        if "start" in w and "end" in w
                    ]

                # Use word timestamps for more accurate boundaries if available
                if words:
                    start = words[0].start
                    end = words[-1].end
                else:
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)

                segments.append(
                    SubtitleSegment(
                        index=i + 1,
                        start=start,
                        end=end,
                        text=seg.get("text", "").strip(),
                        words=words,
                        confidence=seg.get("score", 1.0) if "score" in seg else 1.0,
                    )
                )

            transcription_info = TranscriptionInfo(
                language=detected_language,
                language_probability=result.get("language_probability", 1.0)
                if "language_probability" in result
                else 1.0,
                duration=audio_duration,
                degradations=degradations,
            )

            logger.info(
                f"WhisperX transcription complete: {len(segments)} segments, "
                f"language: {transcription_info.language}"
            )

            return segments, transcription_info

        except Exception as e:
            logger.error(f"WhisperX transcription failed: {e}")
            raise TranscriptionError(f"WhisperX transcription failed: {e}") from e

    def _drop_hallucinated_chunks(
        self, chunks: List[dict], audio: Any, language: str
    ) -> List[dict]:
        """The WhisperX chunks that are not wholly made up; see core.hallucinations."""
        kept = []
        for chunk in chunks:
            if is_hallucination(
                chunk.get("text", ""),
                lambda: self._whisperx_no_speech_prob(chunk, audio, language),
            ):
                logger.debug(f"Dropped made-up chunk at {chunk['start']:.1f} s: {chunk['text']}")
            else:
                kept.append(chunk)
        _log_hallucinations(len(chunks) - len(kept))
        return kept

    def _whisperx_no_speech_prob(self, chunk: dict, audio: Any, language: str) -> float:
        """Whisper's no-speech probability for one chunk, which WhisperX's batched decoding never
        asks for: the features WhisperX decoded, through the encoder and one decoder step again."""
        pipeline = self._whisperx_model
        model = pipeline.model
        clip = audio[int(chunk["start"] * 16000) : int(chunk["end"] * 16000)]
        features = np.asarray(pipeline.preprocess({"inputs": clip})["inputs"])
        tokenizer = Tokenizer(
            model.hf_tokenizer, model.model.is_multilingual, task="transcribe", language=language
        )
        result = model.model.generate(
            model.encode(features),
            [tokenizer.sot_sequence],
            max_length=1,
            return_no_speech_prob=True,
        )
        return float(result[0].no_speech_prob)

    def _transcribe_faster_whisper(
        self, audio_path: Path, *, language: Optional[str]
    ) -> Tuple[List[SubtitleSegment], TranscriptionInfo]:
        """Transcribe using faster-whisper."""
        model = self.load_model()

        logger.debug("Using faster-whisper for transcription")

        # Use optimized VAD parameters for better subtitle timing
        vad_params = (
            self.vad_parameters if self.vad_parameters is not None else self.DEFAULT_VAD_PARAMETERS
        )
        beam_size, vad_filter, batch_size = self.beam_size, self.vad_filter, self.batch_size

        try:
            # Select inference method
            if batch_size and batch_size > 1:
                if self._batched_pipeline is None:
                    self._batched_pipeline = BatchedInferencePipeline(model=model)

                # Without VAD the pipeline refuses audio over 30 s unless given clips, and each
                # blind 30 s clip needs Whisper's own timestamps to come out as lines, not a block.
                # It never conditions on previous text and ignores hallucination_silence_threshold.
                audio: Any = str(audio_path)
                no_vad: Dict[str, Any] = {}
                if not vad_filter:
                    audio = decode_audio(str(audio_path))
                    duration = len(audio) / model.feature_extractor.sampling_rate
                    no_vad = {
                        "clip_timestamps": whole_audio_clips(duration),
                        "without_timestamps": False,
                    }
                segments_iter, info = self._batched_pipeline.transcribe(
                    audio,
                    language=language,
                    beam_size=beam_size,
                    batch_size=batch_size,
                    vad_filter=vad_filter,
                    word_timestamps=True,
                    vad_parameters=vad_params if vad_filter else None,
                    **no_vad,
                )
            else:
                segments_iter, info = model.transcribe(
                    str(audio_path),
                    language=language,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    word_timestamps=True,
                    vad_parameters=vad_params if vad_filter else None,
                    **({} if vad_filter else self.NO_VAD_OPTIONS),
                )

            # Collect all segments with word-level timestamps
            segments: List[SubtitleSegment] = []
            hallucinated = 0
            for segment in segments_iter:
                if is_hallucination(segment.text, lambda: segment.no_speech_prob):
                    logger.debug(
                        f"Dropped made-up segment at {segment.start:.1f} s: {segment.text}"
                    )
                    hallucinated += 1
                    continue
                # Extract word-level timestamps if available
                words = None
                if hasattr(segment, "words") and segment.words:
                    words = [
                        WordTiming(
                            word=w.word,
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                        )
                        for w in segment.words
                    ]

                # Use word timestamps for more accurate segment boundaries
                if words:
                    start = words[0].start
                    end = words[-1].end
                else:
                    start = segment.start
                    end = segment.end

                segments.append(
                    SubtitleSegment(
                        index=len(segments) + 1,
                        start=start,
                        end=end,
                        text=segment.text.strip(),
                        words=words,
                    )
                )

            _log_hallucinations(hallucinated)
            audio_duration = info.duration if hasattr(info, "duration") else 0.0

            transcription_info = TranscriptionInfo(
                language=info.language,
                language_probability=info.language_probability,
                duration=audio_duration,
            )

            logger.info(
                f"Transcription complete: {len(segments)} segments, "
                f"language: {info.language} (confidence: {info.language_probability:.2%})"
            )

            return segments, transcription_info

        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

    def unload_model(self) -> None:
        """Unload model to free VRAM; waits for a transcription in progress on another thread."""
        import gc

        with self._transcribe_lock:
            self._model = None
            self._batched_pipeline = None
            self._whisperx_model = None
            self._whisperx_align_model = None
            self._whisperx_metadata = None
            self._whisperx_align_language = None

            gc.collect()

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


def build_vad_parameters(
    cfg: WhisperConfig,
    *,
    mode: Optional[str] = None,
    speech_pad_ms: Optional[int] = None,
    min_silence_duration_ms: Optional[int] = None,
) -> dict:
    """Build the silero VAD parameters for a Transcriber.

    Precedence, highest first: the explicit arguments, the `mode` preset (a VAD_PRESETS
    key), then cfg.speech_pad_ms / cfg.min_silence_duration_ms. A preset ignores cfg.
    """
    if mode is not None:
        return Transcriber.get_vad_parameters(
            mode=mode,
            speech_pad_ms=speech_pad_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )
    return Transcriber.get_vad_parameters(
        speech_pad_ms=speech_pad_ms if speech_pad_ms is not None else cfg.speech_pad_ms,
        min_silence_duration_ms=(
            min_silence_duration_ms
            if min_silence_duration_ms is not None
            else cfg.min_silence_duration_ms
        ),
    )


def whole_audio_clips(duration: float) -> List[Dict[str, float]]:
    """Equal back-to-back clips, in seconds, covering the whole audio; batched decoding without VAD.

    Each is at most 30 s, as the batched pipeline transcribes only a clip's first 30 s, and none is
    a sliver: a few samples left over at the end would not survive feature extraction.
    """
    count = max(1, math.ceil(duration / 30.0))
    edges = [duration * k / count for k in range(count)] + [duration]
    return [{"start": a, "end": b} for a, b in zip(edges, edges[1:])]
