"""Core video → subtitles pipeline shared by the CLI and the HTTP server.

A single audio→transcribe→translate→save flow that both entry points call.
Optional hooks (`PipelineHooks`) let the CLI plug in its Rich progress bar
without the pipeline knowing or caring about UI; the server passes no hooks
and gets a quiet, side-effect-free run.

Caller responsibilities:
- Build the ASR backend with core.asr.create_backend and ensure its model is cached locally.
- Build the translator with create_translator and check its service before any work,
  unless target_languages is empty (transcribe-only runs pass none).
- Unload the backend's model when done, if needed (unload_model()).

The pipeline cleans up its own audio scratch file in a finally block.
"""

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, ContextManager, Dict, List, Optional

from ..exceptions import SubtitleError
from ..models.config import MT_BACKENDS, AppConfig, TimestampConfig
from ..models.degradation import Degradation
from ..models.subtitle import SubtitleSegment
from .asr import AsrBackend
from .audio import AudioExtractor
from .subtitle import (
    SubtitleProcessor,
    check_structure,
    check_translation,
    normalize_target_languages,
    validate_language_codes,
)
from .timestamp_processor import TimestampProcessor
from .translator import Translator

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    """One produced subtitle file."""

    language: str  # e.g. "en", "ja", "en-zh" (bilingual label)
    path: Path


@dataclass
class PipelineResult:
    """End state of a pipeline run."""

    detected_language: str
    language_probability: Optional[float]
    segment_count: int
    outputs: List[PipelineOutput] = field(default_factory=list)
    # The backend's fallbacks, then one per target language left partly untranslated.
    degradations: List[Degradation] = field(default_factory=list)

    def describe_language(self) -> str:
        """The detected language, with the backend's confidence when it reports one."""
        if self.language_probability is None:
            return self.detected_language
        return f"{self.detected_language} ({self.language_probability:.1%})"


class StructureCheckError(SubtitleError):
    """Written subtitle files failed the structure check; the files are kept on disk."""

    def __init__(self, failures: Dict[Path, List[str]], outputs: List[PipelineOutput]):
        self.failures = failures
        self.outputs = outputs
        lines = ["Subtitle structure check failed (files were written):"]
        for path, found in failures.items():
            more = f"; and {len(found) - 5} more" if len(found) > 5 else ""
            lines.append(f"  {path}: {'; '.join(found[:5])}{more}")
        super().__init__("\n".join(lines))


@dataclass
class PipelineHooks:
    """Optional UI hooks. Each defaults to None (no-op).

    Server callers pass nothing; the CLI wires these to its progress bar.
    """

    # Called once after the audio scratch file has been written.
    on_audio_extracted: Optional[Callable[[Path], None]] = None

    # Called once after transcription is complete.
    # Args: (segment_count, detected_language)
    on_transcribe_complete: Optional[Callable[[int, str], None]] = None

    # Called once after the original-language SRT has been saved.
    on_original_saved: Optional[Callable[[Path], None]] = None

    # Called once per target language that was skipped (same as source).
    on_translation_skipped: Optional[Callable[[str], None]] = None

    # Context manager factory for per-language translation progress: called with
    # (target_lang, total_segments), yields a (completed, total) callback. None = no progress.
    translation_progress_ctx: Optional[
        Callable[[str, int], ContextManager[Callable[[int, int], None]]]
    ] = None

    # Called once per saved translation file.
    # Args: (output_path, label) — label is e.g. "zh" or "en-zh" (bilingual)
    on_translation_saved: Optional[Callable[[Path, str], None]] = None


def build_timestamp_config(
    config: AppConfig,
    *,
    mode_override: Optional[str] = None,
    split_sentences_override: Optional[bool] = None,
) -> Optional[TimestampConfig]:
    """The timestamp post-processing settings, with the CLI overrides applied.

    Returns None when post-processing is disabled at the config level —
    callers should treat None as "skip the timestamp processor entirely".
    """
    ts = config.timestamp
    if not ts.enabled:
        return None
    ts = replace(
        ts,
        mode=mode_override or ts.mode,
        split_sentences=(
            ts.split_sentences if split_sentences_override is None else split_sentences_override
        ),
    )
    valid_modes = {"off", "minimal", "full"}
    if ts.mode not in valid_modes:
        # Without this, an unknown mode (e.g. a typo'd `--timestamp-mode min`)
        # silently fell through to the "full" branch in TimestampProcessor.
        raise ValueError(f"Invalid timestamp mode {ts.mode!r}; choose one of {sorted(valid_modes)}")
    return ts


def run_pipeline(
    video_path: Path,
    config: AppConfig,
    *,
    transcriber: AsrBackend,
    target_languages: List[str],
    output_dir: Path,
    translator: Optional[Translator] = None,
    source_language: Optional[str] = None,
    keep_original: bool = True,
    original_output_path: Optional[Path] = None,
    bilingual: bool = False,
    timestamp_mode: Optional[str] = None,
    split_sentences: Optional[bool] = None,
    post_process: bool = True,
    hooks: Optional[PipelineHooks] = None,
    free_gpu_for_translation: bool = False,
) -> PipelineResult:
    """Core video → subtitles pipeline.

    See module docstring for caller responsibilities. Hooks are optional
    callbacks for progress UI; with hooks=None this runs silently (server
    mode) and produces no console output of its own.

    Args:
        translator: Required unless target_languages is empty.
        original_output_path: Exact path for the original-language subtitle
            file, for callers that let the user name it. Default None derives
            it from output_dir; translation outputs always do.
        free_gpu_for_translation: Unload the ASR model before an LLM backend translates.
            Only for the transcriber's sole user: another thread about to transcribe with
            it would wait for the unload, then load the model again.

    Raises:
        ValueError: target_languages is non-empty and translator is None, or a
            language code is not filename-safe.
        StructureCheckError: in minimal / full timestamp mode, a written file
            overlaps, has a non-positive duration, runs past the audio, or (for a
            translation) lost or renumbered segments. Raised after every file is written.
    """
    hooks = hooks or PipelineHooks()
    extractor = AudioExtractor()
    subtitle_processor = SubtitleProcessor(encoding=config.output.encoding)
    stem = video_path.stem

    # Language codes become output filenames verbatim ({stem}.{lang}.srt), so
    # path metacharacters must be rejected here — the single choke point for
    # every entry path, server requests included.
    target_languages = normalize_target_languages(target_languages)
    if source_language:
        validate_language_codes([source_language])

    if target_languages and translator is None:
        raise ValueError("run_pipeline needs a translator when target_languages is non-empty")

    audio_path = extractor.extract(video_path)
    try:
        if hooks.on_audio_extracted is not None:
            hooks.on_audio_extracted(audio_path)

        timestamp_config = build_timestamp_config(
            config,
            mode_override=timestamp_mode,
            split_sentences_override=split_sentences,
        )

        segments, info = transcriber.transcribe(audio_path, language=source_language)
        detected_language = info.language

        # None when timestamp.enabled is off at the config level.
        post_processed = post_process and timestamp_config is not None
        if post_processed:
            assert timestamp_config is not None  # restated for the type checker
            processor = TimestampProcessor(timestamp_config, language=detected_language)
            segments = processor.process(segments, info.duration)

        # mode=off and disabled post-processing promise raw timing, so only minimal / full
        # output is held to the structure check.
        checked = post_processed and timestamp_config is not None and timestamp_config.mode != "off"
        failures: Dict[Path, List[str]] = {}

        def save(
            subtitles: List[SubtitleSegment],
            path: Path,
            translation: Optional[List[SubtitleSegment]] = None,
        ) -> None:
            subtitle_processor.save(subtitles, path)
            if checked:
                found = check_structure(subtitles, info.duration)
                # Checked before any bilingual merge, which hides a missing line behind the original.
                if translation is not None:
                    found += check_translation(translation, segments)
                if found:
                    failures[path] = found

        if hooks.on_transcribe_complete is not None:
            hooks.on_transcribe_complete(len(segments), detected_language)

        outputs: List[PipelineOutput] = []
        degradations = list(info.degradations)

        # A local LLM shares the GPU, and with the ASR model still resident the two can overflow
        # its memory and translate several times slower. Machine translation never uses the GPU.
        if (
            free_gpu_for_translation
            and config.translation.backend not in MT_BACKENDS
            and any(lang != detected_language for lang in target_languages)
        ):
            logger.info("Unloading the ASR model to free GPU memory for translation")
            transcriber.unload_model()

        if keep_original:
            original_srt = original_output_path or output_dir / f"{stem}.{detected_language}.srt"
            save(segments, original_srt)
            outputs.append(PipelineOutput(language=detected_language, path=original_srt))
            if hooks.on_original_saved is not None:
                hooks.on_original_saved(original_srt)

        for lang in target_languages:
            assert translator is not None  # guarded above; restated for the type checker
            if lang == detected_language:
                if hooks.on_translation_skipped is not None:
                    hooks.on_translation_skipped(lang)
                else:
                    logger.info("Skipping translation to %s (same as source)", lang)
                continue

            if hooks.translation_progress_ctx is not None:
                with hooks.translation_progress_ctx(lang, len(segments)) as progress_cb:
                    translated = translator.translate(
                        segments,
                        detected_language,
                        lang,
                        progress_callback=progress_cb,
                    )
            else:
                translated = translator.translate(segments, detected_language, lang)
            untranslated = translator.untranslated_degradation(lang, len(segments))
            if untranslated is not None:
                degradations.append(untranslated)

            if bilingual:
                merged = subtitle_processor.merge_bilingual(
                    segments, translated, original_on_top=config.output.original_on_top
                )
                out_path = output_dir / f"{stem}.{detected_language}-{lang}.srt"
                save(merged, out_path, translation=translated)
                label = f"{detected_language}-{lang}"
            else:
                out_path = output_dir / f"{stem}.{lang}.srt"
                save(translated, out_path, translation=translated)
                label = lang

            outputs.append(PipelineOutput(language=label, path=out_path))

            if hooks.on_translation_saved is not None:
                hooks.on_translation_saved(out_path, label)

        if failures:
            raise StructureCheckError(failures, outputs)

        return PipelineResult(
            detected_language=detected_language,
            language_probability=info.language_probability,
            segment_count=len(segments),
            outputs=outputs,
            degradations=degradations,
        )

    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to clean up audio scratch file %s: %s", audio_path, e)
