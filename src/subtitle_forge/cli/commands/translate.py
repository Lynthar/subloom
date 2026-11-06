"""Translate command."""

import re
from pathlib import Path
from typing import Optional

import typer

from . import (
    DROP_INTERJECTIONS_HELP,
    FIND_NAMES_HELP,
    MERGE_FRAGMENTS_HELP,
    NAMES_HELP,
    TRANSLATION_BACKEND_HELP,
)


def translate_subtitle(
    subtitle: Path = typer.Argument(..., help="Subtitle file path (SRT)", exists=True),
    target_lang: str = typer.Option(
        ...,
        "--target-lang",
        "-t",
        help="Target language code",
    ),
    source_lang: Optional[str] = typer.Option(
        None,
        "--source-lang",
        "-s",
        help="Source language code",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output SRT file path",
    ),
    translation_backend: Optional[str] = typer.Option(
        None, "--translation-backend", help=TRANSLATION_BACKEND_HELP
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Translation model (default: the backend's configured model)",
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
):
    """
    Translate existing subtitle file.

    Example:
        subtitle-forge translate video.en.srt --target-lang zh
        subtitle-forge translate video.srt -s en -t zh --bilingual
    """
    from ...core.translator import create_translator
    from ...core.fragments import rejoin_fragments
    from ...core.interjections import without_interjections
    from ...core.subtitle import SubtitleProcessor, validate_language_codes
    from ...utils.progress import SubtitleProgress, print_success, print_error, print_info
    from ..app import EXIT_DEGRADED, apply_translation_flags, get_config

    # get_config() (not a bare AppConfig.load()) so the root --config flag
    # actually reaches this command.
    config = get_config()
    apply_translation_flags(config, translation_backend, model, names, find_names)

    try:
        validate_language_codes([target_lang])
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    # Try to detect source language from filename if not specified
    if source_lang is None:
        # Try pattern: name.lang.srt — validate the *shape* of the last segment
        # ("en", "zh", "yue", "zh-TW") so a resolution/codec tag like the "1080p"
        # in "video.1080p.srt" isn't mistaken for a source language.
        parts = subtitle.stem.split(".")
        if len(parts) >= 2 and re.fullmatch(r"[a-z]{2,3}(-[A-Z]{2})?", parts[-1]):
            source_lang = parts[-1]
            print_info(f"Detected source language from filename: {source_lang}")
        else:
            print_error("Cannot detect source language. Please specify with --source-lang")
            raise typer.Exit(1)

    # Translating a file into its own language is a no-op — and with the
    # default output naming it would overwrite the input file with itself.
    if target_lang == source_lang:
        print_error(
            f"Target language '{target_lang}' is the same as the source — nothing to translate."
        )
        raise typer.Exit(1)

    _, bilingual = config.output.resolve_flags(keep_original=None, bilingual=bilingual)

    # Output path
    if output is None:
        stem = subtitle.stem
        if stem.endswith(f".{source_lang}"):
            stem = stem[: -len(f".{source_lang}")]

        if bilingual:
            output = subtitle.parent / f"{stem}.{source_lang}-{target_lang}.srt"
        else:
            output = subtitle.parent / f"{stem}.{target_lang}.srt"

    if output.resolve() == subtitle.resolve():
        print_error(f"Output path equals the input file ({subtitle}) — refusing to overwrite it.")
        raise typer.Exit(1)

    progress = SubtitleProgress()

    try:
        with progress.track_video(subtitle.name, total_steps=2) as tracker:
            # 1. Load subtitles
            tracker.set_description("Loading subtitles...")
            processor = SubtitleProcessor(encoding=config.output.encoding)
            segments = processor.load(subtitle)
            if drop_interjections is None:
                drop_interjections = config.output.drop_interjections
            if merge_fragments is None:
                merge_fragments = config.output.merge_fragments
            if drop_interjections:
                segments = without_interjections(segments)
            if merge_fragments:
                segments = rejoin_fragments(segments, source_lang)
            tracker.update("Subtitles loaded")

            # 2. Translate
            tracker.set_description("Translating...")
            translator = create_translator(config)

            translated = translator.translate(segments, source_lang, target_lang)
            fell_short = [
                d
                for d in (
                    translator.names_degradation(),
                    translator.untranslated_degradation(target_lang, len(segments)),
                )
                if d is not None
            ]

            # Save
            if bilingual:
                merged = processor.merge_bilingual(
                    segments, translated, original_on_top=config.output.original_on_top
                )
                processor.save(merged, output)
            else:
                processor.save(translated, output)

            tracker.update("Translation complete")

        print_success(
            f"Translation complete!\n"
            f"  Source: {subtitle.name} ({source_lang})\n"
            f"  Target: {target_lang}\n"
            f"  Segments: {len(translated)}\n"
            f"  Output: {output}",
            fell_short,
        )

    except Exception as e:
        print_error(str(e))
        raise typer.Exit(1)

    if fell_short:
        raise typer.Exit(EXIT_DEGRADED)
