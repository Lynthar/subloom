"""Progress display utilities using Rich."""

import logging
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, List, Sequence, Tuple
from contextlib import contextmanager
from datetime import datetime

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    Progress,
    TaskID,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.panel import Panel
from rich.table import Table

from ..models.degradation import Degradation
from ..models.task import VideoTask, TaskStatus

if TYPE_CHECKING:
    from ..core.model_manager import DownloadProgress
    from ..core.asr import AsrBackend

console = Console()

# UI switches set once by the root CLI callback. --quiet promises "only errors", so it silences
# the print_* helpers AND the progress bars; --no-progress silences only the bars; print_error
# always prints. Both flags used to be parsed and change nothing.
_quiet = False
_no_progress = False


def set_ui_options(quiet: bool = False, no_progress: bool = False) -> None:
    """Record the root CLI's --quiet / --no-progress flags."""
    global _quiet, _no_progress
    _quiet = quiet
    _no_progress = no_progress


def progress_disabled() -> bool:
    """Whether progress bars should be suppressed."""
    return _no_progress or _quiet


@contextmanager
def _download_bar(description: str, total: Optional[int]) -> Iterator[Tuple[Progress, TaskID]]:
    """A byte-counting bar with one task; total=None until the download reports a size."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        DownloadColumn(),
        console=console,
        disable=progress_disabled(),
    ) as progress:
        yield progress, progress.add_task(description, total=total)


@contextmanager
def _quiet_download_logs() -> Iterator[None]:
    """huggingface_hub and our own loggers at ERROR meanwhile: INFO lines tear the bar apart."""
    loggers = [logging.getLogger("huggingface_hub"), logging.getLogger("subloom")]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for lg, level in zip(loggers, levels):
            lg.setLevel(level)


def download_asr_model_with_progress(transcriber: "AsrBackend") -> None:
    """Download the backend's model behind a progress bar; download errors propagate."""
    with (
        _quiet_download_logs(),
        _download_bar("Downloading...", transcriber.get_model_size()) as (progress, task),
    ):
        last_completed = 0

        def on_progress(downloaded: int, _total: int) -> None:
            nonlocal last_completed
            # Only `completed` moves: the size estimate stays the total, so the bar never jumps.
            if downloaded > last_completed:
                progress.update(task, completed=downloaded)
                last_completed = downloaded

        transcriber.download_model(progress_callback=on_progress)


def pull_ollama_with_progress(updates: Iterable["DownloadProgress"]) -> None:
    """Show an Ollama pull's progress stream as a bar; errors from the stream propagate."""
    with _download_bar("Initializing...", None) as (progress, task):
        for dp in updates:
            description = dp.status.replace("_", " ").capitalize()
            # total_bytes is 0 while Ollama pulls the manifest / verifies — only the status moves.
            if dp.total_bytes:
                progress.update(
                    task,
                    total=dp.total_bytes,
                    completed=dp.completed_bytes or 0,
                    description=description,
                )
            else:
                progress.update(task, description=description)


class SubtitleProgress:
    """Subtitle processing progress manager."""

    def __init__(self, disable: bool = False):
        self.disable = disable

    def create_progress(self) -> Progress:
        """Create a progress bar instance."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("["),
            TimeElapsedColumn(),
            TextColumn("/"),
            TimeRemainingColumn(),
            TextColumn("]"),
            console=console,
            disable=self.disable or progress_disabled(),
        )

    @contextmanager
    def track_video(self, video_name: str, total_steps: int = 4):
        """
        Track progress for a single video.

        Args:
            video_name: Video file name.
            total_steps: Total number of steps.
        """
        with self.create_progress() as progress:
            task = progress.add_task(f"Processing {video_name}", total=total_steps)

            class ProgressTracker:
                def __init__(self, prog: Progress, task_id: TaskID):
                    self.progress = prog
                    self.task_id = task_id
                    self.current_step = 0
                    self._paused = False

                def update(self, description: str, advance: int = 1):
                    self.progress.update(
                        self.task_id,
                        description=description,
                        advance=advance,
                    )
                    self.current_step += advance

                def set_description(self, description: str):
                    self.progress.update(self.task_id, description=description)

                def pause(self):
                    """Pause the progress bar to allow nested progress bars."""
                    if not self._paused:
                        self.progress.stop()
                        self._paused = True

                def resume(self):
                    """Resume the progress bar after nested operations."""
                    if self._paused:
                        self.progress.start()
                        self._paused = False

            yield ProgressTracker(progress, task)

    @contextmanager
    def track_batch(self, total_videos: int):
        """
        Track progress for batch processing.

        Args:
            total_videos: Total number of videos.
        """
        with self.create_progress() as progress:
            overall_task = progress.add_task(
                "[cyan]Overall progress",
                total=total_videos,
            )
            current_task = progress.add_task(
                "[green]Current video",
                total=100,
                visible=False,
            )

            class BatchTracker:
                def __init__(self, prog: Progress, overall: TaskID, current: TaskID):
                    self.progress = prog
                    self.overall = overall
                    self.current = current

                def start_video(self, name: str):
                    self.progress.update(
                        self.current,
                        description=f"[green]{name}",
                        completed=0,
                        visible=True,
                    )

                def update_current(self, percent: float, description: str = ""):
                    desc = f"[green]{description}" if description else None
                    self.progress.update(
                        self.current,
                        completed=int(percent),
                        description=desc,
                    )

                def complete_video(self):
                    self.progress.update(self.overall, advance=1)
                    self.progress.update(self.current, visible=False)

            yield BatchTracker(progress, overall_task, current_task)


class TranslationProgressTracker:
    """
    Track translation progress with detailed batch information.

    Shows progress like: "Translating subtitles... 45/150 [batch 5/15] 1:30 ~ 3:00"
    """

    def __init__(
        self,
        total_segments: int,
        batch_size: int = 10,
        target_lang: str = "",
        disable: bool = False,
    ):
        """
        Initialize translation progress tracker.

        Args:
            total_segments: Total number of subtitle segments to translate.
            batch_size: Number of segments per translation batch.
            target_lang: Target language name for display.
            disable: Disable progress display.
        """
        self.total_segments = total_segments
        self.batch_size = batch_size
        self.target_lang = target_lang
        self.disable = disable

        self.completed_segments = 0
        self.current_batch = 0
        self.total_batches = (total_segments + batch_size - 1) // batch_size
        self.start_time: Optional[datetime] = None

        self._progress: Optional[Progress] = None
        self._task_id: Optional[TaskID] = None

    def __enter__(self):
        self.start_time = datetime.now()

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[dim]batch {task.fields[batch]}/{task.fields[total_batches]}[/dim]"),
            TimeElapsedColumn(),
            TextColumn("[dim]~[/dim]"),
            TimeRemainingColumn(),
            console=console,
            disable=self.disable or progress_disabled(),
        )
        self._progress.start()

        desc = f"Translating to {self.target_lang}..." if self.target_lang else "Translating..."
        self._task_id = self._progress.add_task(
            desc,
            total=self.total_segments,
            batch=0,
            total_batches=self.total_batches,
        )
        return self

    def __exit__(self, *args):
        if self._progress:
            self._progress.stop()

    def update(self, completed: int, total: int):
        """
        Update progress - designed to be used as translator callback.

        Args:
            completed: Number of segments completed.
            total: Total number of segments.
        """
        self.completed_segments = completed
        self.current_batch = (completed + self.batch_size - 1) // self.batch_size

        if self._progress and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=completed,
                batch=self.current_batch,
            )

    def get_stats(self) -> dict:
        """Get translation statistics."""
        elapsed = 0.0
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()

        rate = self.completed_segments / elapsed if elapsed > 0 else 0

        return {
            "completed": self.completed_segments,
            "total": self.total_segments,
            "batches_done": self.current_batch,
            "total_batches": self.total_batches,
            "elapsed_seconds": elapsed,
            "rate_per_second": rate,
        }


_explainer_shown = False


def print_translation_explainer(show_once: bool = True) -> None:
    """
    Print user-friendly explanation of the translation process.

    Args:
        show_once: Only show once per session (uses module-level flag).
    """
    global _explainer_shown
    if _quiet:
        return
    if show_once and _explainer_shown:
        return

    console.print(
        Panel(
            "[cyan]Translation Process[/cyan]\n\n"
            "Your subtitles are being translated using a local AI model.\n"
            "This runs entirely on your computer - no internet required.\n\n"
            "[dim]Progress shows: completed subtitles / total subtitles[/dim]\n"
            "[dim]Batch processing: subtitles are translated in groups for efficiency[/dim]",
            title="What's happening?",
            border_style="blue",
        )
    )

    if show_once:
        _explainer_shown = True


def print_task_summary(tasks: List[VideoTask]) -> None:
    """Print task summary table."""
    if _quiet:
        return
    table = Table(title="Processing Results")

    table.add_column("File", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right")
    table.add_column("Note")

    for task in tasks:
        if task.status == TaskStatus.COMPLETED and task.degradations:
            status = "[yellow]Degraded[/yellow]"
        elif task.status == TaskStatus.COMPLETED:
            status = "[green]Success[/green]"
        elif task.status == TaskStatus.FAILED:
            status = "[red]Failed[/red]"
        else:
            status = "[yellow]Processing[/yellow]"

        duration = ""
        if task.elapsed_time:
            duration = f"{task.elapsed_time:.1f}s"

        note = task.error if task.error else ""
        if len(note) > 50:
            note = note[:50] + "..."

        table.add_row(
            task.video_path.name,
            status,
            duration,
            note,
        )

    console.print(table)
    _print_batch_degradations([t for t in tasks if t.status == TaskStatus.COMPLETED])


def _print_batch_degradations(completed: List[VideoTask]) -> None:
    """List degradations under the summary, once for those every task shares."""
    shared: List[Degradation] = []
    if len(completed) > 1:
        shared = [
            d for d in completed[0].degradations if all(d in t.degradations for t in completed)
        ]
    if shared:
        console.print("[yellow]Degraded in every task:[/yellow]")
        console.print(_degradation_lines(shared))
    for task in completed:
        own = [d for d in task.degradations if d not in shared]
        if own:
            console.print(f"[yellow]Degraded:[/yellow] [cyan]{escape(task.video_path.name)}[/cyan]")
            console.print(_degradation_lines(own))


def _degradation_lines(degradations: Sequence[Degradation]) -> str:
    return "\n".join(f"  [yellow]![/yellow] {escape(d.message)}" for d in degradations)


def print_error(message: str) -> None:
    """Print an error message as plain text: an exception's '[whisperx]' or a '[Group] x.mkv'
    path would otherwise be read as markup and vanish."""
    console.print(Panel(escape(message), title="Error", border_style="red"))


def print_success(message: str, degradations: Sequence[Degradation] = ()) -> None:
    """Print the completion panel; degradations turn it yellow and are listed under the message."""
    if _quiet:
        return
    if not degradations:
        console.print(Panel(message, title="Complete", border_style="green"))
        return
    body = f"{message}\n\nDegraded:\n{_degradation_lines(degradations)}"
    console.print(Panel(body, title="Complete (degraded)", border_style="yellow"))


def print_info(message: str) -> None:
    """Print info message."""
    if _quiet:
        return
    console.print(f"[cyan]i[/cyan] {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    if _quiet:
        return
    console.print(f"[yellow]![/yellow] {message}")
