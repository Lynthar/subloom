"""Interactive setup wizard for first-time users."""

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from .progress import (
    download_asr_model_with_progress,
    pull_ollama_with_progress,
)
from ..models.config import (
    ASR_BACKENDS,
    DEEPL_API_KEY_ENV,
    GOOGLE_API_KEY_ENV,
    OPENAI_API_KEY_ENV,
    TRANSLATION_BACKENDS,
    AppConfig,
)

console = Console()

ASR_NOTES = {
    "whisper": "faster-whisper, aligned by WhisperX when installed; about 100 languages",
    "qwen3_asr": "Qwen3-ASR with its forced aligner; recommended for Chinese, Japanese, Korean",
}
TRANSLATION_NOTES = {
    "ollama": "a local LLM served by Ollama",
    "openai": "an OpenAI-compatible server: LM Studio, llama.cpp, vLLM or a cloud API",
    "deepl": "DeepL API (API key)",
    "google": "Google Cloud Translation Basic (API key)",
}
KEY_ENVS = {"openai": OPENAI_API_KEY_ENV, "deepl": DEEPL_API_KEY_ENV, "google": GOOGLE_API_KEY_ENV}
STEPS = 6


def check_ffmpeg() -> bool:
    """Check if ffmpeg is installed."""
    return shutil.which("ffmpeg") is not None


def qwen3_asr_installed() -> bool:
    """Whether the [qwen3-asr] extra is installed."""
    return importlib.util.find_spec("qwen_asr") is not None


def whisperx_installed() -> bool:
    """Whether WhisperX imports, as the Whisper backend will find it."""
    from ..core.transcriber import WHISPERX_AVAILABLE

    return WHISPERX_AVAILABLE


def check_gpu() -> dict:
    """Check GPU availability and info."""
    from .gpu import check_cuda_available, get_gpu_info

    if check_cuda_available():
        return get_gpu_info()
    return {"cuda_available": False}


def recommend_ollama_model(gpu_info: dict) -> str:
    """The Ollama translation model that fits the GPU in ``gpu_info``."""
    total_vram = gpu_info.get("total_vram_mb", 0) if gpu_info.get("cuda_available") else 0
    if total_vram >= 30000:
        return "qwen2.5:32b"
    if total_vram >= 16000:
        return "qwen2.5:14b"
    return "qwen2.5:7b"


def _step(number: int, title: str) -> None:
    console.print()
    console.print(f"[bold]Step {number}/{STEPS}: {title}[/bold]")


def _pick(
    question: str,
    options: Sequence[str],
    default: Optional[str],
    notes: Optional[Mapping[str, str]] = None,
    free: bool = False,
) -> str:
    """Show ``options`` numbered and return the one chosen by number or name; ``free`` also
    takes any other text."""
    for i, name in enumerate(options, 1):
        note = f"  [dim]{notes[name]}[/dim]" if notes else ""
        console.print(f"    {i}. [cyan]{escape(name)}[/cyan]{note}")
    while True:
        answer = str(typer.prompt(f"  {question}", default=default or None)).strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        if answer in options or (free and answer):
            return answer
        console.print(f"  [red]Choose a number from 1 to {len(options)} or a name[/red]")


def _save(config: AppConfig, path: Optional[Path]) -> None:
    config.validate()
    config.save(path)


def _key_source(config: AppConfig, backend: str) -> Optional[str]:
    """Where the backend's API key comes from, or None when it has none."""
    if os.environ.get(KEY_ENVS[backend]):
        return KEY_ENVS[backend]
    if getattr(config, backend).api_key:
        return f"{backend}.api_key in the config file"
    return None


def _print_key_hint(env: str) -> None:
    if sys.platform == "win32":
        console.print(f"    setx {env} <your key>    [dim](then open a new terminal)[/dim]")
    else:
        console.print(f"    export {env}=<your key>    [dim](add it to your shell profile)[/dim]")


def _missing_extra(extra: str, what: str, issues: List[str]) -> None:
    console.print(f"  [yellow]MISSING[/yellow] {what} needs the {escape(f'[{extra}]')} extra:")
    console.print(escape(f"    pip install -e '.[{extra}]'"))
    issues.append(f"{extra} extra")


def _setup_asr(config: AppConfig, has_gpu: bool, issues: List[str]) -> None:
    """Choose the speech recognition backend and fit its device and model to this machine."""
    backend = _pick("Backend", ASR_BACKENDS, config.asr.backend, ASR_NOTES)
    config.asr.backend = backend

    if backend == "qwen3_asr" and not qwen3_asr_installed():
        _missing_extra("qwen3-asr", "Qwen3-ASR", issues)
    if backend == "whisper" and config.whisper.use_whisperx and not whisperx_installed():
        _missing_extra("whisperx", "WhisperX word alignment", issues)
        console.print(
            "  [dim]Without it, timing comes from plain faster-whisper "
            "and every run exits with code 3[/dim]"
        )

    # Without CUDA, write cpu now: faster-whisper does not fall back from cuda by itself, and
    # Qwen3-ASR's fallback would report every run as degraded.
    section = getattr(config, backend)
    if not has_gpu and section.device != "cpu":
        section.device = "cpu"
        saved = "device=cpu"
        if backend == "whisper":
            config.whisper.compute_type = "int8"
            saved += ", compute_type=int8"
        console.print(f"  [dim]Set {backend}: {saved} (no NVIDIA GPU)[/dim]")

    if backend == "whisper":
        from ..core.transcriber import Transcriber

        # Unreadable free VRAM recommends no change.
        recommended = Transcriber.select_optimal_model() if has_gpu else "small"
        if recommended and recommended != config.whisper.model:
            reason = "fits the free VRAM" if has_gpu else "large models are painfully slow on CPU"
            if typer.confirm(
                f"  Use the recommended Whisper model {recommended} ({reason}) "
                f"instead of {config.whisper.model}?",
                default=True,
            ):
                config.whisper.model = recommended


def _setup_asr_model(config: AppConfig, issues: List[str]) -> None:
    """Download the configured speech recognition model if it is missing."""
    from ..core.asr import create_backend

    transcriber = create_backend(config)
    model = transcriber.model_name
    if transcriber.is_model_cached():
        console.print(f"  [green]OK[/green] Model '{model}' is ready")
        return

    size_mb = transcriber.get_model_size() / (1024 * 1024)
    console.print(f"  [yellow]MISSING[/yellow] Model '{model}' not downloaded")
    console.print(f"  Size: ~{size_mb:.0f}MB (one-time download)")
    if not typer.confirm(f"  Download '{model}' now?", default=True):
        console.print("  [dim]It will be downloaded when you first run subloom[/dim]")
        return

    console.print("  [dim]Tip: If interrupted, it will auto-download when you run subloom[/dim]\n")
    try:
        download_asr_model_with_progress(transcriber)
    except KeyboardInterrupt:
        console.print("\n[yellow]Download paused. It resumes when you run subloom.[/yellow]")
        issues.append("speech recognition model")
        return
    except Exception as e:
        console.print(f"\n[red]Download failed: {e}[/red]")
        issues.append("speech recognition model")
        return
    console.print(f"\n  [green]OK[/green] Model '{model}' downloaded successfully!")


def _setup_translation(config: AppConfig, gpu_info: dict, issues: List[str]) -> None:
    """Choose the translation backend and fill in what it needs."""
    backend = _pick("Backend", TRANSLATION_BACKENDS, config.translation.backend, TRANSLATION_NOTES)
    config.translation.backend = backend

    if backend == "ollama":
        recommended = recommend_ollama_model(gpu_info)
        if recommended != config.ollama.model and typer.confirm(
            f"  Use the recommended model {recommended} instead of {config.ollama.model}?",
            default=True,
        ):
            config.ollama.model = recommended
        return

    if backend == "openai":
        _setup_openai(config)
        return

    env = KEY_ENVS[backend]
    source = _key_source(config, backend)
    if source:
        console.print(f"  API key: from {source}")
    else:
        console.print(f"  [yellow]MISSING[/yellow] No API key. Set {env}:")
        _print_key_hint(env)
        issues.append(f"{env} not set")


def _setup_openai(config: AppConfig) -> None:
    """Ask for the server's address, then pick the model from what it offers."""
    import httpx

    from ..core.chat import create_chat_backend
    from ..core.service import ServiceError

    cfg = config.openai
    console.print(
        "  [dim]e.g. http://localhost:1234/v1 (LM Studio), http://localhost:8080/v1 "
        "(llama.cpp)[/dim]"
    )
    while True:
        url = typer.prompt("  Server base URL", default=cfg.base_url or None).strip()
        if url.startswith(("http://", "https://")):
            break
        console.print("  [red]Give an http(s) URL[/red]")
    cfg.base_url = url

    env = OPENAI_API_KEY_ENV
    source = _key_source(config, "openai")
    if source:
        console.print(f"  API key: from {source}")
    else:
        console.print(f"  [dim]No API key: fine for a local server; a cloud API needs {env}:[/dim]")
        _print_key_hint(env)

    try:
        offered = create_chat_backend(config).list_models()
    except (ServiceError, httpx.TransportError) as e:
        console.print(f"  [yellow]Could not list the server's models:[/yellow] {escape(str(e))}")
        offered = []
    if offered:
        cfg.model = _pick("Model (number or id)", offered, cfg.model or "1", free=True)
    else:
        cfg.model = typer.prompt("  Model id", default=cfg.model or None).strip()


def _check_translation(config: AppConfig, issues: List[str]) -> None:
    """Probe the configured translation service, offering to pull a missing Ollama model."""
    from ..core.chat import OllamaChat
    from ..core.service import ServiceError
    from ..core.translator import create_translator

    try:
        service = create_translator(config).backend
    except ValueError:
        console.print("  [yellow]SKIPPED[/yellow] No API key to check with")
        return

    if not service.check_connection():
        console.print(f"  [red]NOT REACHABLE[/red] Cannot connect to {service.endpoint}")
        if isinstance(service, OllamaChat):
            console.print("  [yellow]Please install and start Ollama:[/yellow]")
            console.print("    1. Download from: https://ollama.ai/download")
            console.print("    2. Start the service: ollama serve")
        issues.append(f"{service.label} not reachable")
        return

    try:
        available = service.check_model_available()
    except ServiceError as e:
        console.print(f"  [red]REFUSED[/red] {service.endpoint}: {escape(str(e))}")
        issues.append(f"{service.label} refused the check")
        return

    if not service.model:
        console.print(f"  [green]OK[/green] {service.label} accepted the API key")
    elif available:
        console.print(f"  [green]OK[/green] Model '{service.model}' is ready")
    elif isinstance(service, OllamaChat):
        _pull_ollama_model(service, issues)
    else:
        offered = service.list_models()
        console.print(
            f"  [yellow]MISSING[/yellow] {service.endpoint} does not offer '{service.model}'"
        )
        console.print(f"    It offers: {', '.join(offered) or 'nothing'}")
        issues.append("translation model")


def _pull_ollama_model(service, issues: List[str]) -> None:
    model = service.model
    console.print(f"  [yellow]MISSING[/yellow] Model '{model}' not downloaded")
    console.print("  This is a one-time download and may take a while for large models.")
    if not typer.confirm(f"  Download '{model}' now?", default=True):
        console.print("  [dim]You can download later with: subloom config pull-model[/dim]")
        issues.append("translation model")
        return

    console.print("  [dim]Tip: If interrupted, run 'subloom config pull-model' to resume[/dim]\n")
    try:
        pull_ollama_with_progress(service.model_manager.pull_model(model))
    except KeyboardInterrupt:
        console.print("\n[yellow]Download paused. Resume with: subloom config pull-model[/yellow]")
        issues.append("translation model")
        return
    except Exception as e:
        console.print(f"\n[red]Download failed: {e}[/red]")
        issues.append("translation model")
        return
    console.print(f"\n  [green]OK[/green] Model '{model}' downloaded successfully!")


def run_setup_wizard(config: AppConfig, config_path: Optional[Path] = None) -> None:
    """Run the interactive setup wizard, saving each choice to ``config_path`` as it is made
    (None = the default location)."""

    console.print(
        Panel(
            "[bold cyan]Welcome to subloom![/bold cyan]\n\n"
            "This tool generates and translates subtitles for your videos.\n"
            "Recognition runs locally; translation goes to the backend you choose.\n\n"
            "Let's make sure everything is set up correctly.",
            title="Quick Start Guide",
            border_style="cyan",
        )
    )
    issues: List[str] = []

    _step(1, "Checking ffmpeg...")
    if check_ffmpeg():
        console.print("  [green]OK[/green] ffmpeg is installed")
    else:
        console.print("  [red]MISSING[/red] ffmpeg is not installed")
        console.print()
        console.print("  [yellow]Please install ffmpeg:[/yellow]")
        console.print("    macOS:   brew install ffmpeg")
        console.print("    Ubuntu:  sudo apt install ffmpeg")
        console.print("    Windows: choco install ffmpeg")
        console.print()
        issues.append("ffmpeg")
        if not typer.confirm("  Continue setup anyway?", default=False):
            raise typer.Exit(1)

    _step(2, "Checking GPU...")
    gpu_info = check_gpu()
    has_gpu = bool(gpu_info.get("cuda_available"))
    if has_gpu:
        console.print(f"  [green]OK[/green] GPU detected: {gpu_info.get('device_name', 'Unknown')}")
        console.print(
            f"      VRAM: {gpu_info.get('total_vram_mb', 0)}MB total, "
            f"{gpu_info.get('available_vram_mb', 0)}MB available"
        )
    else:
        console.print("  [yellow]INFO[/yellow] No NVIDIA GPU detected")
        console.print("      Speech recognition will use the CPU (slower but functional)")

    # Each choice is saved before anything is downloaded by it: a recommendation that only
    # gets printed leaves the defaults to download, and those can run out of memory.
    _step(3, "Choosing speech recognition...")
    _setup_asr(config, has_gpu, issues)
    _save(config, config_path)

    _step(4, "Checking speech recognition model...")
    _setup_asr_model(config, issues)

    _step(5, "Choosing translation...")
    _setup_translation(config, gpu_info, issues)
    _save(config, config_path)

    _step(6, f"Checking translation ({config.translation.backend})...")
    _check_translation(config, issues)

    console.print()
    if issues:
        console.print(
            Panel(
                "[yellow]Setup completed with some issues.[/yellow]\n\n"
                f"Issues found: {', '.join(issues)}\n\n"
                "Please resolve the issues above before using subloom.\n"
                "Run 'subloom config check' to verify your setup.",
                title="Setup Status",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                "[bold green]Setup Complete![/bold green]\n\n"
                "Everything is configured and ready to use!\n\n"
                "[cyan]Quick Start Examples:[/cyan]\n\n"
                "  # Generate Chinese subtitles for a video\n"
                "  subloom process video.mp4 -t zh\n\n"
                "  # Generate subtitles in multiple languages\n"
                "  subloom process video.mp4 -t zh -t ja -t ko\n\n"
                "  # Create bilingual subtitles (original + translation)\n"
                "  subloom process video.mp4 -t zh --bilingual\n\n"
                "  # Process all videos in a folder\n"
                "  subloom batch ./videos/ -t zh\n\n"
                "[dim]For more options: subloom --help[/dim]",
                title="Ready!",
                border_style="green",
            )
        )
