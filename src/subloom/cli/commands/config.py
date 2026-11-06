"""Configuration management command."""

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ...models.config import (
    DEEPL_API_KEY_ENV,
    GOOGLE_API_KEY_ENV,
    OPENAI_API_KEY_ENV,
    AppConfig,
)
from ...utils.gpu import get_gpu_info, check_cuda_available
from ...utils.progress import (
    print_success,
    print_error,
    print_info,
    pull_ollama_with_progress,
)

app = typer.Typer(no_args_is_help=True)
console = Console()


def _active_config_path() -> Optional[Path]:
    """The --config override from the root callback, if any (None = default).

    Deferred import: cli.app imports this module at startup, so a top-level
    import would be circular.
    """
    from ..app import get_config_path

    return get_config_path()


def _load_config() -> AppConfig:
    """The config the root callback loaded, from the root --config flag's file."""
    from ..app import get_config

    return get_config()


def _config_file_display() -> Path:
    return _active_config_path() or AppConfig.get_config_path()


@app.command()
def show():
    """Show current configuration (every field, including nested sections)."""
    import dataclasses

    config = _load_config()

    # Secrets never go to the terminal (or the scrollback / screen share it
    # ends up in) — only whether one is set.
    sensitive_fields = {"hf_token", "api_key"}

    def _display(value, field_name: str = "") -> str:
        if value is None:
            return "[dim]null[/dim]"
        if field_name in sensitive_fields:
            return "[dim]<set, hidden>[/dim]"
        text = str(value)
        if len(text) > 60:
            text = text[:57] + "..."
        return text

    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Reflect over the dataclasses so every field shows up, new ones included: the hand-maintained
    # table silently omitted whole sections (timestamp.*) and fields (ollama.request_timeout,
    # whisper.batch_size) — exactly the values `config set` writes.
    for section_field in dataclasses.fields(config):
        value = getattr(config, section_field.name)
        if dataclasses.is_dataclass(value):
            table.add_row(f"[bold]{section_field.name}[/bold]", "")
            for f in dataclasses.fields(value):
                table.add_row(f"  {f.name}", _display(getattr(value, f.name), f.name))
        else:
            table.add_row(section_field.name, _display(value, section_field.name))

    console.print(table)
    for env, section in (
        (OPENAI_API_KEY_ENV, "openai"),
        (DEEPL_API_KEY_ENV, "deepl"),
        (GOOGLE_API_KEY_ENV, "google"),
    ):
        if os.environ.get(env):
            console.print(f"\n{env} is set and overrides {section}.api_key")
    console.print(f"\nConfig file: {_config_file_display()}")


@app.command()
def set(
    key: str = typer.Argument(..., help="Setting key (e.g., whisper.model)"),
    value: str = typer.Argument(..., help="Setting value"),
):
    """
    Set a configuration value.

    Accepts any field in the config, resolved by reflection over the dataclass
    (so new fields work automatically and nothing silently no-ops). Use a dotted
    key for nested sections; the value is coerced to the field's declared type.

    Example:
        subloom config set whisper.model large-v3
        subloom config set whisper.use_whisperx false
        subloom config set timestamp.linger_ms 250
        subloom config set translation.request_timeout 600
        subloom config set max_workers 4
    """
    import dataclasses
    import typing

    def _coerce(raw: str, annotation):
        """Coerce a CLI string to the field's annotated type."""
        low = raw.strip().lower()
        base = annotation
        args = typing.get_args(annotation)
        if args:  # e.g. Optional[int] -> (int, NoneType)
            if low in ("null", "none", ""):
                return None
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                base = non_none[0]
        if base is bool:
            if low in ("true", "1", "yes", "on"):
                return True
            if low in ("false", "0", "no", "off"):
                return False
            raise ValueError(f"expected true/false, got {raw!r}")
        if base is int:
            return int(raw)
        if base is float:
            return float(raw)
        return raw  # str, or unknown type -> leave as-is

    config = _load_config()
    parts = key.split(".")

    try:
        if len(parts) == 1:
            fields = {f.name: f for f in dataclasses.fields(config)}
            fld = fields.get(key)
            # Reject the nested-section names (whisper/ollama/...) — they need a
            # dotted key like "whisper.model".
            if fld is None or dataclasses.is_dataclass(getattr(config, key)):
                print_error(f"Unknown setting: {key}")
                raise typer.Exit(1)
            setattr(config, key, _coerce(value, fld.type))
        elif len(parts) == 2:
            section_name, field_name = parts
            section = getattr(config, section_name, None)
            if section is None or not dataclasses.is_dataclass(section):
                print_error(f"Unknown section: {section_name}")
                raise typer.Exit(1)
            fields = {f.name: f for f in dataclasses.fields(section)}
            fld = fields.get(field_name)
            if fld is None:
                print_error(f"Unknown {section_name} setting: {field_name}")
                raise typer.Exit(1)
            setattr(section, field_name, _coerce(value, fld.type))
        else:
            print_error(f"Invalid key format: {key}")
            raise typer.Exit(1)
    except (ValueError, TypeError) as e:
        print_error(f"Invalid value for {key}: {e}")
        raise typer.Exit(1)

    # Domain validation before saving — otherwise an out-of-range value
    # (chars_per_second 0, max_retries 0, ...) persists fine and only blows
    # up mid-job, possibly hours later.
    try:
        config.validate()
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    config.save(_active_config_path())
    print_success(f"Set {key} = {value}")


@app.command()
def reset():
    """Reset configuration to defaults."""
    config = AppConfig()
    config.save(_active_config_path())
    print_success("Configuration reset to defaults")


@app.command()
def check(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed diagnostic information",
    ),
):
    """
    Check system status and dependencies.

    Example:
        subloom config check
        subloom config check --verbose
    """
    import shutil

    console.print(Panel("System Diagnostics", style="bold blue"))

    issues = []
    config = _load_config()

    # Check CUDA
    console.print("\n[bold]GPU Status:[/bold]")
    if check_cuda_available():
        gpu_info = get_gpu_info()
        console.print("  [green]CUDA:[/green] Available")
        console.print(f"    GPU: {gpu_info.get('device_name', 'Unknown')}")
        console.print(f"    VRAM Total: {gpu_info.get('total_vram_mb', 0)}MB")
        console.print(f"    VRAM Available: {gpu_info.get('available_vram_mb', 0)}MB")

        if verbose:
            try:
                import torch

                console.print(f"    PyTorch Version: {torch.__version__}")
                console.print(f"    CUDA Version: {torch.version.cuda}")
            except ImportError:
                pass
    else:
        console.print("  [yellow]CUDA:[/yellow] Not available")
        if config.whisper.device == "cuda":
            # Don't just say "will use CPU" — with plain faster-whisper a cuda
            # device setting fails to load the model outright (only the
            # WhisperX path falls back). Make the mismatch actionable.
            console.print(
                "    [red]Config mismatch:[/red] whisper.device is 'cuda' but CUDA is not available"
            )
            console.print("    Fix with: subloom config set whisper.device cpu")
            issues.append("whisper.device=cuda but CUDA is not available")
        else:
            console.print("    Transcription will use CPU (slower)")

        if verbose:
            console.print(
                "    [dim]To enable GPU acceleration, install CUDA and PyTorch with CUDA support[/dim]"
            )

    # Check ffmpeg
    console.print("\n[bold]FFmpeg Status:[/bold]")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        console.print("  [green]ffmpeg:[/green] Found")
        if verbose:
            console.print(f"    Path: {ffmpeg_path}")
            import subprocess

            try:
                result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
                version_line = result.stdout.split("\n")[0] if result.stdout else "Unknown"
                console.print(f"    Version: {version_line}")
            except Exception:
                pass
    else:
        console.print("  [red]ffmpeg:[/red] Not found")
        console.print("    [yellow]Install ffmpeg:[/yellow]")
        console.print("      macOS:   brew install ffmpeg")
        console.print("      Ubuntu:  sudo apt install ffmpeg")
        console.print("      Windows: choco install ffmpeg")
        issues.append("ffmpeg not installed")

    # Check the configured translation backend
    console.print(f"\n[bold]Translation ({config.translation.backend}):[/bold]")
    try:
        from ...core.chat import OllamaChat
        from ...core.translator import create_translator

        chat = create_translator(config).backend
        is_ollama = isinstance(chat, OllamaChat)

        if chat.check_connection():
            console.print("  [green]Connection:[/green] OK")
            console.print(f"    Endpoint: {chat.endpoint}")

            if not chat.model:
                # A machine translation service has no model to pick; what can fail is the key.
                chat.check_model_available()
                console.print("  [green]API key:[/green] accepted")
            else:
                available_models = chat.list_models()
                if verbose:
                    console.print(f"    Available models: {', '.join(available_models) or 'None'}")
                if chat.check_model_available():
                    console.print(f"  [green]Model:[/green] {chat.model} (ready)")
                elif is_ollama:
                    console.print(f"  [yellow]Model:[/yellow] {chat.model} (not downloaded)")
                    console.print("    [yellow]Download with:[/yellow] subloom config pull-model")
                    issues.append(f"Translation model '{chat.model}' not downloaded")
                else:
                    console.print(f"  [yellow]Model:[/yellow] {chat.model} (not offered)")
                    console.print(f"    Offered: {', '.join(available_models) or 'nothing'}")
                    issues.append(
                        f"Translation model '{chat.model}' not offered by {chat.endpoint}"
                    )
        else:
            console.print("  [red]Connection:[/red] Failed")
            console.print(f"    Cannot connect to {chat.endpoint}")
            if is_ollama:
                console.print("    [yellow]Start Ollama with:[/yellow] ollama serve")
            issues.append(f"Cannot connect to {chat.label}")

    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        if verbose:
            import traceback

            console.print(f"    [dim]{traceback.format_exc()}[/dim]")
        issues.append(f"Translation backend error: {e}")

    # Configuration summary
    if verbose:
        console.print("\n[bold]Current Configuration:[/bold]")
        console.print(f"  Whisper Model: {config.whisper.model}")
        console.print(f"  Whisper Device: {config.whisper.device}")
        console.print(f"  Translation Backend: {config.translation.backend}")
        console.print(f"  Max Workers: {config.max_workers}")
        console.print(f"  Config Path: {_config_file_display()}")

    # Check the configured backend's speech recognition model
    console.print(f"\n[bold]Speech Recognition ({config.asr.backend}):[/bold]")
    try:
        from ...core.asr import create_backend

        transcriber = create_backend(config)
        model = transcriber.model_name
        if transcriber.is_model_cached():
            console.print(f"  [green]Model:[/green] {model} (ready)")
        else:
            model_size_mb = transcriber.get_model_size() / (1024 * 1024)
            console.print(
                f"  [yellow]Model:[/yellow] {model} (not downloaded, ~{model_size_mb:.0f}MB)"
            )
            console.print("    [dim]Will be downloaded automatically on first use[/dim]")
            issues.append(f"Speech recognition model '{model}' not downloaded")

    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        if verbose:
            import traceback

            console.print(f"    [dim]{traceback.format_exc()}[/dim]")

    # Recommended model based on VRAM
    if check_cuda_available():
        from ...core.transcriber import Transcriber

        recommended = Transcriber.select_optimal_model()
        if recommended is not None:
            console.print("\n[bold]Recommendation:[/bold]")
            console.print(f"  Whisper model: [cyan]{recommended}[/cyan] (based on available VRAM)")

    # Summary
    console.print()
    if issues:
        console.print(
            Panel(
                "[yellow]Issues Found:[/yellow]\n\n"
                + "\n".join(f"  - {issue}" for issue in issues)
                + "\n\n[dim]Run 'subloom quickstart' for guided setup[/dim]",
                title="Status",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                "[green]All systems operational![/green]\n\nReady to process videos.",
                title="Status",
                border_style="green",
            )
        )


@app.command()
def export(
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output file path",
    ),
):
    """Export configuration to file."""
    config = _load_config()
    config.save(output)
    print_success(f"Configuration exported to: {output}")


@app.command("import")
def import_config(
    input_file: Path = typer.Argument(..., help="Configuration file to import", exists=True),
):
    """Import configuration from file."""
    config = AppConfig.load(input_file)
    config.save(_active_config_path())
    print_success(f"Configuration imported from: {input_file}")


@app.command("pull-model")
def pull_model(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to download (default: configured model)",
    ),
):
    """
    Download Ollama translation model with progress display.

    Supports automatic resume - if download is interrupted,
    simply run this command again to continue from where it left off.

    Example:
        subloom config pull-model
        subloom config pull-model --model qwen2.5:32b
    """

    from ...core.model_manager import OllamaModelManager

    config = _load_config()
    if config.translation.backend != "ollama":
        print_error(
            f"pull-model downloads Ollama models; translation.backend is "
            f"{config.translation.backend}, whose server manages its own models"
        )
        raise typer.Exit(1)
    target_model = model or config.ollama.model

    console.print(
        Panel(
            f"[cyan]Model:[/cyan] {target_model}\n[cyan]Host:[/cyan] {config.ollama.host}",
            title="Download Configuration",
            border_style="blue",
        )
    )

    manager = OllamaModelManager(host=config.ollama.host)

    # Check connection first
    if not manager.check_connection():
        print_error("Cannot connect to Ollama service")
        console.print("\n[yellow]Please ensure Ollama is running:[/yellow]")
        console.print("  ollama serve")
        raise typer.Exit(1)

    # Check if already available
    if manager.is_model_available(target_model):
        print_success(f"Model {target_model} is already downloaded and ready to use")
        return

    console.print(f"\n[cyan]Downloading model: {target_model}[/cyan]")
    console.print("[dim]Tip: If interrupted, run this command again to resume download[/dim]\n")

    try:
        pull_ollama_with_progress(manager.pull_model(target_model))

        print_success(f"Model {target_model} downloaded successfully!")
        console.print("\n[green]You can now use subloom for translation.[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Download paused. Run this command again to resume.[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        print_error(f"Download failed: {e}")
        console.print(
            "\n[yellow]You can try again - download will resume from where it stopped.[/yellow]"
        )
        raise typer.Exit(1)


@app.command("show-prompt")
def show_prompt():
    """
    Show current translation prompt template.

    Displays the prompt used for subtitle translation. If a custom prompt
    is configured, it will be shown; otherwise, the default prompt is displayed.

    Example:
        subloom config show-prompt
    """
    from ...core.translator import SubtitleTranslator, effective_prompt_template

    config = _load_config()
    template_id = config.translation.prompt_template_id
    prompt = effective_prompt_template(template_id)

    # A dangling template id resolves to the default, so label what is
    # actually shown rather than what the config asked for.
    if template_id is None or prompt == SubtitleTranslator.DEFAULT_PROMPT_TEMPLATE:
        header, border = "[dim](Using default prompt)[/dim]", "blue"
    else:
        header, border = f"[cyan]library:{template_id}[/cyan]", "cyan"

    console.print(
        Panel(
            f"{header}\n\n{prompt}",
            title="Translation Prompt",
            border_style=border,
        )
    )

    console.print("\n[bold]Available placeholders:[/bold]")
    console.print("  {source_lang}    - Source language name")
    console.print("  {target_lang}    - Target language name")
    console.print("  {context_before} - Previous dialogue (for context)")
    console.print("  {segments}       - Lines to translate")
    console.print("  {context_after}  - Following dialogue (for context)")


@app.command("set-prompt")
def set_prompt(
    file: Path = typer.Option(
        ...,
        "--file",
        "-f",
        help="Path to prompt template file",
        exists=True,
    ),
):
    """
    Set custom translation prompt from a file.

    Saves it as the user library template "custom" (replacing an earlier one) and
    selects it. The prompt file should contain the template with placeholders.

    Available placeholders:
      {source_lang}    - Source language name
      {target_lang}    - Target language name
      {context_before} - Previous dialogue (for context)
      {segments}       - Lines to translate
      {context_after}  - Following dialogue (for context)

    Example:
        subloom config set-prompt --file my_prompt.txt
    """
    # Read prompt from file
    try:
        prompt_content = file.read_text(encoding="utf-8")
    except Exception as e:
        print_error(f"Failed to read file: {e}")
        raise typer.Exit(1)

    # Validate placeholders
    required_placeholders = ["{source_lang}", "{target_lang}", "{segments}"]
    missing = [p for p in required_placeholders if p not in prompt_content]
    if missing:
        print_error(f"Missing required placeholders: {', '.join(missing)}")
        console.print("\n[yellow]Required placeholders:[/yellow]")
        console.print("  {source_lang} - Source language name")
        console.print("  {target_lang} - Target language name")
        console.print("  {segments}    - Lines to translate")
        raise typer.Exit(1)

    from ...core.prompt_library import store_custom_template

    config = _load_config()
    config.translation.prompt_template_id = store_custom_template(prompt_content, replace=True)
    config.save(_active_config_path())

    print_success(f"Custom prompt loaded from: {file}")
    console.print(f"\n[dim]Prompt length: {len(prompt_content)} characters[/dim]")


@app.command("reset-prompt")
def reset_prompt():
    """
    Reset translation prompt to default.

    Deselects any library or custom template and reverts to the built-in default.

    Example:
        subloom config reset-prompt
    """
    config = _load_config()

    if config.translation.prompt_template_id:
        config.translation.prompt_template_id = None
        config.save(_active_config_path())
        print_success("Translation prompt reset to default")
    else:
        print_info("Already using default prompt")


@app.command("export-prompt")
def export_prompt(
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output file path",
    ),
):
    """
    Export current prompt template to a file.

    Exports the current prompt (custom or default) for editing.

    Example:
        subloom config export-prompt -o my_prompt.txt
    """
    from ...core.translator import effective_prompt_template

    config = _load_config()
    prompt = effective_prompt_template(config.translation.prompt_template_id)

    try:
        output.write_text(prompt, encoding="utf-8")
        print_success(f"Prompt exported to: {output}")
        console.print(
            "\n[dim]Edit the file and use 'config set-prompt -f <file>' to apply changes[/dim]"
        )
    except Exception as e:
        print_error(f"Failed to write file: {e}")
        raise typer.Exit(1)


# ========== Prompt Library Commands ==========


@app.command("list-prompts")
def list_prompts(
    genre: Optional[str] = typer.Option(
        None,
        "--genre",
        "-g",
        help="Filter by genre (movie, documentary, anime, etc.)",
    ),
):
    """
    List all available prompt templates.

    Shows built-in and user-defined templates from the prompt library.

    Example:
        subloom config list-prompts
        subloom config list-prompts --genre movie
    """
    from ...core.prompt_library import get_prompt_library

    library = get_prompt_library()
    templates = library.list_templates(genre=genre)

    if not templates:
        if genre:
            print_info(f"No templates found for genre: {genre}")
        else:
            print_info("No templates found")
        return

    # Create table
    table = Table(title="Available Prompt Templates")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description")
    table.add_column("Genre", style="dim")
    table.add_column("Source", style="dim")

    for t in templates:
        source = "[dim]builtin[/dim]" if library.is_builtin(t.id) else "[cyan]user[/cyan]"
        table.add_row(t.id, t.name, t.description, t.genre, source)

    console.print(table)

    # Show available genres
    genres = library.get_genres()
    console.print(f"\n[dim]Available genres: {', '.join(genres)}[/dim]")
    console.print("[dim]Use: subloom config use-prompt <template-id>[/dim]")


@app.command("show-prompt-template")
def show_prompt_template(
    template_id: str = typer.Argument(..., help="Template ID to display"),
):
    """
    Show details of a specific prompt template.

    Example:
        subloom config show-prompt-template movie-scifi
    """
    from ...core.prompt_library import get_prompt_library

    library = get_prompt_library()
    template = library.get_template(template_id)

    if not template:
        print_error(f"Template not found: {template_id}")
        console.print("\n[dim]Use 'config list-prompts' to see available templates[/dim]")
        raise typer.Exit(1)

    # Determine source
    source = "Built-in" if library.is_builtin(template_id) else "User-defined"

    console.print(
        Panel(
            f"[bold]ID:[/bold] {template.id}\n"
            f"[bold]Name:[/bold] {template.name}\n"
            f"[bold]Genre:[/bold] {template.genre}\n"
            f"[bold]Tags:[/bold] {', '.join(template.tags) if template.tags else 'None'}\n"
            f"[bold]Source:[/bold] {source}\n\n"
            f"[bold]Description:[/bold]\n{template.description}\n\n"
            f"[bold]Template:[/bold]\n[dim]{template.template}[/dim]",
            title=f"Prompt Template: {template.name}",
            border_style="cyan",
        )
    )


@app.command("use-prompt")
def use_prompt(
    template_id: str = typer.Argument(..., help="Template ID to use"),
):
    """
    Set the prompt template from library.

    Example:
        subloom config use-prompt movie-scifi
        subloom config use-prompt documentary
    """
    from ...core.prompt_library import get_prompt_library

    library = get_prompt_library()
    template = library.get_template(template_id)

    if not template:
        print_error(f"Template not found: {template_id}")
        console.print("\n[dim]Use 'config list-prompts' to see available templates[/dim]")
        raise typer.Exit(1)

    config = _load_config()
    config.translation.prompt_template_id = template_id
    config.save(_active_config_path())

    print_success(f"Now using prompt template: {template.name}")
    console.print(f"[dim]Description: {template.description}[/dim]")


@app.command("save-prompt")
def save_prompt(
    file: Path = typer.Option(
        ...,
        "--file",
        "-f",
        help="Path to prompt template file",
        exists=True,
    ),
    id: str = typer.Option(
        ...,
        "--id",
        help="Unique identifier for this template",
    ),
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Display name for this template",
    ),
    description: str = typer.Option(
        "",
        "--description",
        "-d",
        help="Description of when to use this template",
    ),
    genre: str = typer.Option(
        "custom",
        "--genre",
        "-g",
        help="Genre category (movie, documentary, anime, etc.)",
    ),
):
    """
    Save a custom prompt template to user library.

    Creates a new template from a file that can be reused later.

    Example:
        subloom config save-prompt -f my_prompt.txt --id my-scifi --name "My Sci-Fi"
    """
    from ...core.prompt_library import get_prompt_library
    from ...models.prompt import PromptTemplate

    # Read prompt from file
    try:
        prompt_content = file.read_text(encoding="utf-8")
    except Exception as e:
        print_error(f"Failed to read file: {e}")
        raise typer.Exit(1)

    # Create template
    template = PromptTemplate(
        id=id,
        name=name,
        description=description,
        template=prompt_content,
        genre=genre,
        tags=[],
        author="user",
    )

    # Validate
    if not template.is_valid():
        missing = template.validate()
        print_error(f"Invalid template: missing placeholders {', '.join(missing)}")
        console.print("\n[yellow]Required placeholders:[/yellow]")
        console.print("  {source_lang} - Source language name")
        console.print("  {target_lang} - Target language name")
        console.print("  {segments}    - Lines to translate")
        raise typer.Exit(1)

    # Save to library
    library = get_prompt_library()
    try:
        file_path = library.save_user_template(template)
        print_success(f"Template saved: {template.name}")
        console.print(f"[dim]ID: {template.id}[/dim]")
        console.print(f"[dim]File: {file_path}[/dim]")
        console.print(f"\n[dim]Use with: subloom config use-prompt {template.id}[/dim]")
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)


@app.command("delete-prompt")
def delete_prompt(
    template_id: str = typer.Argument(..., help="Template ID to delete"),
):
    """
    Delete a user-defined prompt template.

    Only user-created templates can be deleted. Built-in templates cannot be removed.

    Example:
        subloom config delete-prompt my-custom-template
    """
    from ...core.prompt_library import get_prompt_library

    library = get_prompt_library()

    # Check if built-in
    if library.is_builtin(template_id):
        print_error(f"Cannot delete built-in template: {template_id}")
        raise typer.Exit(1)

    # Check if exists
    if not library.is_user_defined(template_id):
        print_error(f"Template not found: {template_id}")
        raise typer.Exit(1)

    # Confirm deletion
    if not typer.confirm(f"Delete template '{template_id}'?", default=False):
        print_info("Cancelled")
        raise typer.Exit(0)

    # Delete
    if library.delete_user_template(template_id):
        print_success(f"Template deleted: {template_id}")

        # Clear from config if it was the active template
        config = _load_config()
        if config.translation.prompt_template_id == template_id:
            config.translation.prompt_template_id = None
            config.save(_active_config_path())
            console.print("[dim]Switched to default prompt template[/dim]")
    else:
        print_error("Failed to delete template")
        raise typer.Exit(1)


@app.command("export-prompt-template")
def export_prompt_template(
    template_id: str = typer.Argument(..., help="Template ID to export"),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output file path",
    ),
):
    """
    Export a library template to a file for customization.

    Example:
        subloom config export-prompt-template movie-scifi -o my_scifi.txt
    """
    from ...core.prompt_library import get_prompt_library

    library = get_prompt_library()
    template = library.get_template(template_id)

    if not template:
        print_error(f"Template not found: {template_id}")
        console.print("\n[dim]Use 'config list-prompts' to see available templates[/dim]")
        raise typer.Exit(1)

    try:
        output.write_text(template.template, encoding="utf-8")
        print_success(f"Template '{template.name}' exported to: {output}")
        console.print(
            f'\n[dim]Edit and save with: subloom config save-prompt -f {output} --id my-{template_id} --name "My {template.name}"[/dim]'
        )
    except Exception as e:
        print_error(f"Failed to write file: {e}")
        raise typer.Exit(1)
