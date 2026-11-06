"""CLI main application."""

import io
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import typer
from rich.console import Console

from .commands import (
    ASR_BACKEND_HELP,
    DROP_INTERJECTIONS_HELP,
    FIND_NAMES_HELP,
    MERGE_FRAGMENTS_HELP,
    NAMES_HELP,
    TRANSLATION_BACKEND_HELP,
)
from .commands import transcribe, translate, batch, config, serve
from ..models.config import MT_BACKENDS, AppConfig
from ..utils.logger import setup_logging

app = typer.Typer(
    name="subtitle-forge",
    help="Local video subtitle generation and translation tool",
    add_completion=True,
    no_args_is_help=True,
)

console = Console()

# Exit code of a run that finished with degradations; 1 is failure and 2 click's usage error.
EXIT_DEGRADED = 3

# transcribe/translate/batch/serve expose a single action each, so they register as plain root
# commands. **Never attach them with add_typer()** — that makes a Click *group*, which never
# auto-invokes its lone command, so `subtitle-forge transcribe <video>` fails with No such command.
app.command("transcribe", help="Transcribe video to subtitles")(transcribe.transcribe_video)
app.command("translate", help="Translate existing subtitles")(translate.translate_subtitle)
app.command("batch", help="Batch process multiple videos")(batch.batch_process)
app.command("serve", help="Run as an HTTP server (job-based REST API)")(serve.serve)
app.add_typer(config.app, name="config", help="Configuration management")

# Global config
_config: Optional[AppConfig] = None
# The config subcommands need the PATH, not just the loaded object, so that
# `--config custom.yaml config set ...` reads AND writes custom.yaml instead of reading it
# and saving to the default location. None = default location.
_config_path: Optional[Path] = None


def get_config() -> AppConfig:
    """Get current configuration."""
    global _config
    if _config is None:
        _config = AppConfig.load(_config_path)
    return _config


def apply_asr_flag(cfg: AppConfig, backend: Optional[str]) -> None:
    """Put --asr-backend into cfg; exits 1 on an unknown backend, before any work. Runs before
    whisper_flags_apply, which reads the backend."""
    if not backend:
        return
    cfg.asr.backend = backend
    try:
        cfg.validate()
    except ValueError as e:
        from ..utils.progress import print_error

        print_error(str(e))
        raise typer.Exit(1)


def whisper_flags_apply(cfg: AppConfig, flags: Mapping[str, object]) -> bool:
    """True under the Whisper backend; otherwise warn about each Whisper-only flag that was set."""
    if cfg.asr.backend == "whisper":
        return True
    given = [name for name, value in flags.items() if value not in (None, False)]
    if given:
        from ..utils.progress import print_warning

        print_warning(
            f"{', '.join(given)}: Whisper-only, ignored under asr.backend {cfg.asr.backend}"
        )
    return False


def apply_translation_flags(
    cfg: AppConfig,
    backend: Optional[str],
    model: Optional[str],
    names: Optional[str] = None,
    find_names: Optional[bool] = None,
) -> None:
    """Put --translation-backend / --translation-model / --names / --find-names into cfg, then
    build its translator once; exits 1 when any fails, before any work."""
    from ..core.translator import create_translator
    from ..utils.progress import print_error, print_warning

    if backend:
        cfg.translation.backend = backend
    if model:
        # Each backend reads the config section of its own name; an MT one has no model.
        section = getattr(cfg, cfg.translation.backend, None)
        if section is not None and hasattr(section, "model"):
            section.model = model
        else:
            print_warning(
                f"Model {model!r} ignored under translation.backend {cfg.translation.backend}"
            )
    if names is not None:
        cfg.translation.names = [n.strip() for n in re.split(r"[,，、]", names) if n.strip()]
    if find_names is not None:
        cfg.translation.find_names = find_names
    try:
        cfg.validate()
        create_translator(cfg)  # An unset API key fails here, not once per batch task
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)


def prepare_asr_backend(transcriber) -> None:
    """Say how the backend will run, and download its model behind a progress bar if missing."""
    from ..core.transcriber import Transcriber
    from ..utils.progress import download_asr_model_with_progress, print_info

    if isinstance(transcriber, Transcriber) and transcriber.use_whisperx:
        print_info("Using WhisperX for improved timestamp accuracy")
    if not transcriber.is_model_cached():
        model_size_mb = transcriber.get_model_size() / (1024 * 1024)
        console.print(
            f"\n[cyan]Downloading speech recognition model: {transcriber.model_name}[/cyan]"
        )
        console.print(f"[dim]Model size: ~{model_size_mb:.0f}MB (one-time download)[/dim]\n")
        download_asr_model_with_progress(transcriber)
        print_info("Speech recognition model downloaded.\n")


def get_config_path() -> Optional[Path]:
    """The --config override from the root callback (None = default path)."""
    return _config_path


@app.callback()
def main(
    ctx: typer.Context,
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file path",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output mode",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Quiet mode, only show errors",
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        help="Log file path",
    ),
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="Disable progress bar",
    ),
):
    """subtitle-forge - Local video subtitle generation and translation tool"""
    global _config, _config_path

    # Load configuration. load() validates field ranges and raises ValueError
    # with the full list of problems — surface that as a clean error instead
    # of a traceback.
    try:
        _config_path = config_file
        _config = AppConfig.load(config_file)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # Setup logging
    log_level = "DEBUG" if verbose else ("ERROR" if quiet else _config.log_level)
    setup_logging(log_level, str(log_file) if log_file else _config.log_file)

    # Wire --quiet / --no-progress into the Rich helpers — before this they
    # were parsed but had no effect on panels or progress bars.
    from ..utils.progress import set_ui_options

    set_ui_options(quiet=quiet, no_progress=no_progress)

    # Store in context
    ctx.ensure_object(dict)
    ctx.obj["config"] = _config
    ctx.obj["no_progress"] = no_progress


@app.command()
def process(
    video: Path = typer.Argument(..., help="Video file path", exists=True),
    target_lang: List[str] = typer.Option(
        ...,
        "--target-lang",
        "-t",
        help="Target language(s) (can be specified multiple times)",
    ),
    source_lang: Optional[str] = typer.Option(
        None,
        "--source-lang",
        "-s",
        help="Source language (auto-detect if not specified)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory",
    ),
    asr_backend: Optional[str] = typer.Option(None, "--asr-backend", help=ASR_BACKEND_HELP),
    whisper_model: Optional[str] = typer.Option(
        None,
        "--whisper-model",
        help="Whisper model name",
    ),
    translation_backend: Optional[str] = typer.Option(
        None, "--translation-backend", help=TRANSLATION_BACKEND_HELP
    ),
    translation_model: Optional[str] = typer.Option(
        None,
        "--translation-model",
        "--ollama-model",
        help="Translation model (default: the backend's configured model)",
    ),
    keep_original: Optional[bool] = typer.Option(
        None,
        "--keep-original/--no-keep-original",
        help="Keep original language subtitles (default: config output.keep_original)",
    ),
    bilingual: Optional[bool] = typer.Option(
        None,
        "--bilingual/--no-bilingual",
        help="Generate bilingual subtitles (default: config output.bilingual)",
    ),
    drop_interjections: Optional[bool] = typer.Option(
        None,
        "--drop-interjections/--keep-interjections",
        help=DROP_INTERJECTIONS_HELP,
    ),
    names: Optional[str] = typer.Option(None, "--names", help=NAMES_HELP),
    find_names: Optional[bool] = typer.Option(
        None, "--find-names/--no-find-names", help=FIND_NAMES_HELP
    ),
    merge_fragments: Optional[bool] = typer.Option(
        None,
        "--merge-fragments/--no-merge-fragments",
        help=MERGE_FRAGMENTS_HELP,
    ),
    # VAD options for subtitle timing
    vad_mode: Optional[str] = typer.Option(
        None,
        "--vad-mode",
        help="VAD preset mode: default, aggressive, relaxed, precise",
    ),
    speech_pad: Optional[int] = typer.Option(
        None,
        "--speech-pad",
        help="Speech padding in milliseconds (overrides vad-mode)",
    ),
    min_silence: Optional[int] = typer.Option(
        None,
        "--min-silence",
        help="Minimum silence duration in ms for segment breaks (overrides vad-mode)",
    ),
    # Prompt template option
    prompt_template: Optional[str] = typer.Option(
        None,
        "--prompt-template",
        "-p",
        help="Prompt template ID from library (use 'config list-prompts' to see available)",
    ),
    # WhisperX options
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
    hf_mirror: Optional[str] = typer.Option(
        None,
        "--hf-mirror",
        help="HuggingFace mirror URL (e.g., https://hf-mirror.com)",
    ),
    save_failed_log: bool = typer.Option(
        False,
        "--save-failed-log",
        help="Save failed translations to a JSON log file for debugging",
    ),
    save_debug_log: bool = typer.Option(
        False,
        "--save-debug-log",
        help="Save detailed debug logs and failure reports (creates {video}_debug/ folder)",
    ),
):
    """
    Process a video: extract audio -> transcribe -> translate -> save subtitles

    Example:
        subtitle-forge process video.mp4 --target-lang zh
        subtitle-forge process video.mp4 -t zh -t ja --bilingual
    """
    from ..core.asr import create_backend
    from ..core.chat import OllamaChat
    from ..core.service import ServiceError
    from ..core.transcriber import build_vad_parameters
    from ..core.translator import create_translator
    from ..utils.progress import (
        SubtitleProgress,
        TranslationProgressTracker,
        print_success,
        print_error,
        print_info,
        print_warning,
        print_translation_explainer,
        pull_ollama_with_progress,
    )

    cfg = get_config()

    # Override config if specified
    apply_asr_flag(cfg, asr_backend)
    if whisper_model:
        cfg.whisper.model = whisper_model
    apply_translation_flags(cfg, translation_backend, translation_model, names, find_names)
    if prompt_template and cfg.translation.backend in MT_BACKENDS:
        print_warning(
            f"--prompt-template: LLM-only, ignored under translation.backend {cfg.translation.backend}"
        )
    elif prompt_template:
        # Reject an unknown id here, not after the GPU run: it leaves JSON mode
        # off (id is set) while resolving to the default prompt, so the whole
        # translation runs on the least reliable parse path.
        from ..core.prompt_library import get_prompt_library

        if get_prompt_library().get_template(prompt_template) is None:
            print_error(f"Prompt template not found: {prompt_template}")
            console.print("\n[dim]Use 'config list-prompts' to see available templates[/dim]")
            raise typer.Exit(1)
        cfg.translation.prompt_template_id = prompt_template

    output_dir = output_dir or video.parent
    keep_original, bilingual = cfg.output.resolve_flags(
        keep_original=keep_original, bilingual=bilingual
    )
    progress = SubtitleProgress()

    # Handle --save-debug-log option
    debug_dir = None
    debug_log_path = None
    debug_failed_log_path = None
    if save_debug_log:
        debug_dir = output_dir / f"{video.stem}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_log_path = str(debug_dir / "run.log")
        debug_failed_log_path = str(debug_dir / "translation_failures.json")
        # File handler captures DEBUG; console stays at INFO so the terminal is not flooded with
        # third-party stack traces (torio's FFmpeg-extension probing, which Rich renders in full).
        setup_logging(level="DEBUG", log_file=debug_log_path, console_level="INFO")

    try:
        # ========== Phase 1: Prepare models (outside main progress bar) ==========

        backend_overrides: Dict[str, object] = {}
        if hf_mirror:
            backend_overrides["hf_endpoint"] = hf_mirror
        whisper_flags = {
            "--whisper-model": whisper_model,
            "--whisperx/--no-whisperx": use_whisperx,
            "--vad-mode": vad_mode,
            "--speech-pad": speech_pad,
            "--min-silence": min_silence,
        }
        if whisper_flags_apply(cfg, whisper_flags):
            if use_whisperx is not None:
                backend_overrides["use_whisperx"] = use_whisperx
            # Full precedence: CLI flag > --vad-mode preset > config. The bare
            # Transcriber.get_vad_parameters would ignore the configured tuning.
            backend_overrides["vad_parameters"] = build_vad_parameters(
                cfg.whisper,
                mode=vad_mode,
                speech_pad_ms=speech_pad,
                min_silence_duration_ms=min_silence,
            )
        transcriber = create_backend(cfg, **backend_overrides)
        prepare_asr_backend(transcriber)

        # Initialize translator
        # --save-debug-log implies saving failed log to debug directory
        effective_save_failed_log = save_failed_log or save_debug_log
        if debug_failed_log_path:
            failed_log_path = debug_failed_log_path
        elif save_failed_log:
            failed_log_path = str(output_dir / f"{video.stem}_translation_failures.json")
        else:
            failed_log_path = None

        translator = create_translator(
            cfg,
            save_failed_log=effective_save_failed_log,
            failed_log_path=failed_log_path,
        )
        chat = translator.backend

        # Same order as `config pull-model`: an unreachable server must not read as a
        # missing model, or the user is offered a download that cannot work.
        if not translator.check_connection():
            hint = " Run: ollama serve" if isinstance(chat, OllamaChat) else ""
            print_error(f"Cannot connect to {chat.label} at {chat.endpoint}.{hint}")
            raise typer.Exit(1)
        try:
            model_available = translator.check_model_available()
            offered = [] if model_available else chat.list_models()
        except ServiceError as e:
            print_error(f"{chat.label} at {chat.endpoint} refused the check: {e}")
            raise typer.Exit(1)

        # Fail or download here, not after transcription: a GPU run that ends with no
        # translation model is wasted.
        if not model_available:
            if not isinstance(chat, OllamaChat):
                print_error(
                    f"Translation model '{chat.model}' is not offered by {chat.endpoint}; "
                    f"it lists: {', '.join(offered) or 'nothing'}"
                )
                raise typer.Exit(1)
            print_warning(f"Translation model '{chat.model}' not found")
            if typer.confirm("Download model now?", default=True):
                console.print(f"\n[cyan]Downloading model: {chat.model}[/cyan]")
                console.print("[dim]This may take a while for large models...[/dim]\n")
                try:
                    pull_ollama_with_progress(chat.model_manager.pull_model(chat.model))
                except RuntimeError as e:
                    print_error(f"{e}\nRun: subtitle-forge config pull-model")
                    raise typer.Exit(1)
                print_info("Model downloaded successfully!\n")
            else:
                print_error(
                    "Translation requires the configured model. Run: subtitle-forge config pull-model"
                )
                raise typer.Exit(1)

        # ========== Phase 2: Main processing (single progress bar) ==========

        from contextlib import contextmanager
        from ..core.pipeline import PipelineHooks, run_pipeline

        with progress.track_video(video.name) as tracker:
            tracker.set_description(f"[1/4] Extracting audio: {video.name}")
            phase_3_started = [False]  # mutable so nested closures can flip it

            def _start_phase_3():
                """Run once on the first translate-or-skip event to keep the
                visual sequence (audio→transcribe→original→translation) the
                same as the pre-refactor flow."""
                if phase_3_started[0]:
                    return
                phase_3_started[0] = True
                tracker.set_description(f"[3/4] Translating: {video.name}")
                # Pause main progress bar for translation (avoid two bars)
                tracker.pause()
                print_translation_explainer()

            def hook_audio_extracted(_path):
                tracker.update("[1/4] Audio extraction complete")
                tracker.set_description(f"[2/4] Transcribing: {video.name}")

            def hook_transcribe_complete(_seg_count, _detected_lang):
                tracker.update("[2/4] Transcription complete")

            def hook_original_saved(path):
                print_info(f"Original subtitles saved: {path}")

            def hook_translation_skipped(lang):
                _start_phase_3()
                print_info(f"Skipping translation to {lang} (same as source)")

            @contextmanager
            def trans_progress_ctx(lang, total):
                _start_phase_3()
                lang_name = translator.LANGUAGE_NAMES.get(lang, lang)
                print_info(f"Translating to {lang_name}...")
                with TranslationProgressTracker(
                    total_segments=total,
                    # Use the model-clamped size the translator will actually use
                    # (7B/14B clamp below max_batch_size), so "batch x/y" is right.
                    batch_size=translator.effective_batch_size(),
                    target_lang=lang_name,
                ) as trans_progress:
                    yield trans_progress.update

            def hook_translation_saved(path, _label):
                print_info(f"Translated subtitles saved: {path}")

            result = run_pipeline(
                video,
                cfg,
                transcriber=transcriber,
                translator=translator,
                target_languages=target_lang,
                output_dir=output_dir,
                source_language=source_lang,
                keep_original=keep_original,
                bilingual=bilingual,
                timestamp_mode=timestamp_mode,
                split_sentences=split_sentences,
                post_process=post_process,
                drop_interjections=drop_interjections,
                merge_fragments=merge_fragments,
                hooks=PipelineHooks(
                    on_audio_extracted=hook_audio_extracted,
                    on_transcribe_complete=hook_transcribe_complete,
                    on_original_saved=hook_original_saved,
                    on_translation_skipped=hook_translation_skipped,
                    translation_progress_ctx=trans_progress_ctx,
                    on_translation_saved=hook_translation_saved,
                ),
                free_gpu_for_translation=True,
            )

            # Only resume if we actually paused (in case nothing translatable)
            if phase_3_started[0]:
                tracker.resume()
            tracker.update("[3/4] Translation complete")

            # Cleanup — pipeline handles the audio scratch file itself.
            tracker.set_description("[4/4] Cleaning up")
            transcriber.unload_model()
            tracker.update("[4/4] Complete")

        print_success(
            f"Processing complete!\n"
            f"  Video: {video.name}\n"
            f"  Detected language: {result.describe_language()}\n"
            f"  Segments: {result.segment_count}\n"
            f"  Output directory: {output_dir}",
            result.degradations,
        )

    except typer.Exit:
        # typer.Exit subclasses RuntimeError — without this re-raise the
        # handler below would print an empty error panel for clean exits.
        raise
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(1)

    if result.degradations:
        raise typer.Exit(EXIT_DEGRADED)


@app.command()
def quickstart():
    """
    Interactive first-time setup wizard.

    Guides you through:
    - Checking system requirements (ffmpeg, GPU)
    - Choosing the speech recognition and translation backends
    - Downloading their models and checking the translation service

    Example:
        subtitle-forge quickstart
    """
    from ..utils.setup_wizard import run_setup_wizard

    run_setup_wizard(get_config(), get_config_path())


@app.command()
def version():
    """Show version information."""
    from .. import __version__

    console.print(f"subtitle-forge version {__version__}")


def cli():
    """Entry point."""
    # A pipe or file on Windows gets the ANSI code page, which cannot encode CJK text or Rich's
    # spinner, and Rich then raises mid-run with exit 1; UTF-8 encodes everything.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="replace")
    app()


if __name__ == "__main__":
    cli()
