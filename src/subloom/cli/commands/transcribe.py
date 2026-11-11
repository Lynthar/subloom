"""Transcribe command."""

from pathlib import Path
from typing import Dict, List, Optional

import typer

from . import ASR_BACKEND_HELP, DROP_INTERJECTIONS_HELP, MERGE_FRAGMENTS_HELP


def transcribe_video(
    video: Path = typer.Argument(..., help="Video file path", exists=True),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output SRT file path",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Source language (auto-detect if not specified)",
    ),
    asr_backend: Optional[str] = typer.Option(None, "--asr-backend", help=ASR_BACKEND_HELP),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Speech recognition model name (for the configured asr.backend)",
    ),
    vad_filter: Optional[bool] = typer.Option(
        None,
        "--vad-filter/--no-vad-filter",
        help="Enable VAD filtering (default: config whisper.vad_filter)",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Batch size (uses BatchedInferencePipeline)",
    ),
    auto_model: bool = typer.Option(
        False,
        "--auto-model",
        help="Auto-select the Whisper model based on GPU VRAM",
    ),
    use_whisperx: Optional[bool] = typer.Option(
        None,
        "--whisperx/--no-whisperx",
        help="Use WhisperX for better timestamp accuracy",
    ),
    post_process: bool = typer.Option(
        True,
        "--post-process/--no-post-process",
        help="Enable timestamp post-processing to fix timing issues",
    ),
    timestamp_mode: Optional[str] = typer.Option(
        None,
        "--timestamp-mode",
        help="Timestamp processing mode: off, minimal (default), full",
    ),
    split_sentences: Optional[bool] = typer.Option(
        None,
        "--split-sentences/--no-split-sentences",
        help="Split multi-sentence segments using word timestamps for better timing",
    ),
    drop_interjections: Optional[bool] = typer.Option(
        None,
        "--drop-interjections/--keep-interjections",
        help=DROP_INTERJECTIONS_HELP,
    ),
    merge_fragments: Optional[bool] = typer.Option(
        None,
        "--merge-fragments/--no-merge-fragments",
        help=MERGE_FRAGMENTS_HELP,
    ),
    save_debug_log: bool = typer.Option(
        False,
        "--save-debug-log",
        help="Save detailed debug logs (creates {video}_debug/ beside the output)",
    ),
):
    """
    Transcribe video audio to subtitles (no translation).

    Example:
        subloom transcribe video.mp4
        subloom transcribe video.mp4 --language en --model large-v3
    """
    from ...core.asr import create_backend
    from ...core.pipeline import PipelineHooks, run_pipeline
    from ...core.transcriber import Transcriber
    from ...utils.progress import (
        SubtitleProgress,
        print_success,
        print_error,
        print_info,
    )
    from ...utils.logger import setup_logging
    from ...models.degradation import AUTO_MODEL_FALLBACK, Degradation
    from ..app import (
        EXIT_DEGRADED,
        apply_asr_flag,
        get_config,
        prepare_asr_backend,
        whisper_flags_apply,
    )

    # get_config() (not a bare AppConfig.load()) so the root --config flag
    # actually reaches this command.
    config = get_config()
    apply_asr_flag(config, asr_backend)
    progress = SubtitleProgress()
    output_dir = output.parent if output else video.parent

    # Handle --save-debug-log option
    if save_debug_log:
        debug_dir = output_dir / f"{video.stem}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_log_path = str(debug_dir / "run.log")
        # console_level="INFO" keeps third-party DEBUG stack traces out of the
        # terminal while the file still captures everything (same as process).
        setup_logging("DEBUG", debug_log_path, console_level="INFO")

    whisper_flags = {
        "--auto-model": auto_model,
        "--batch-size": batch_size,
        "--vad-filter/--no-vad-filter": vad_filter,
        "--whisperx/--no-whisperx": use_whisperx,
    }
    backend_overrides: Dict[str, object] = {}
    degradations: List[Degradation] = []
    if model:
        backend_overrides["model_name"] = model
    if whisper_flags_apply(config, whisper_flags):
        if auto_model:
            selected = Transcriber.select_optimal_model()
            if selected is None:
                selected = "small"
                degradations.append(
                    Degradation(
                        AUTO_MODEL_FALLBACK,
                        "Free GPU memory could not be read; --auto-model fell back to small",
                    )
                )
            backend_overrides["model_name"] = selected
            print_info(f"Auto-selected model: {selected}")
        if vad_filter is not None:
            backend_overrides["vad_filter"] = vad_filter
        if use_whisperx is not None:
            backend_overrides["use_whisperx"] = use_whisperx
        if batch_size is not None:
            backend_overrides["batch_size"] = batch_size

    try:
        # ========== Phase 1: Prepare model (outside main progress bar) ==========
        transcriber = create_backend(config, **backend_overrides)
        prepare_asr_backend(transcriber)

        # ========== Phase 2: Main processing (single progress bar) ==========

        with progress.track_video(video.name, total_steps=3) as tracker:
            tracker.set_description("Extracting audio...")

            def hook_audio_extracted(_audio_path):
                tracker.update("Audio extraction complete")
                tracker.set_description("Transcribing...")

            def hook_transcribe_complete(_segment_count, _detected_lang):
                tracker.update("Transcription complete")
                tracker.set_description("Saving subtitles...")

            def hook_original_saved(_path):
                tracker.update("Save complete")

            # No target languages, so no translator: this command never
            # translates. --output names the one file it does write.
            result = run_pipeline(
                video,
                config,
                transcriber=transcriber,
                target_languages=[],
                output_dir=output_dir,
                source_language=language,
                keep_original=True,
                original_output_path=output,
                timestamp_mode=timestamp_mode,
                split_sentences=split_sentences,
                post_process=post_process,
                drop_interjections=drop_interjections,
                merge_fragments=merge_fragments,
                hooks=PipelineHooks(
                    on_audio_extracted=hook_audio_extracted,
                    on_transcribe_complete=hook_transcribe_complete,
                    on_original_saved=hook_original_saved,
                ),
            )

            transcriber.unload_model()

        degradations += result.degradations
        print_success(
            f"Transcription complete!\n"
            f"  Detected language: {result.describe_language()}\n"
            f"  Segments: {result.segment_count}\n"
            f"  Output: {result.outputs[0].path}",
            degradations,
        )

    except Exception as e:
        print_error(str(e))
        raise typer.Exit(1)

    if degradations:
        raise typer.Exit(EXIT_DEGRADED)
