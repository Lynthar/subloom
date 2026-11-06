"""Task queue module for batch processing."""

import asyncio
from pathlib import Path
from typing import List, Optional, Callable, Any
from datetime import datetime
import logging

from ..models.task import TaskStatus, VideoTask

logger = logging.getLogger(__name__)


class TaskQueue:
    """Async task queue manager for batch processing."""

    def __init__(
        self,
        max_workers: int = 2,
        on_task_start: Optional[Callable[[VideoTask], None]] = None,
        on_task_complete: Optional[Callable[[VideoTask], None]] = None,
        on_task_error: Optional[Callable[[VideoTask, Exception], None]] = None,
    ):
        """
        Initialize task queue.

        Args:
            max_workers: Maximum concurrent tasks.
            on_task_start: Callback when task starts.
            on_task_complete: Callback when task completes.
            on_task_error: Callback when task fails.
        """
        self.max_workers = max_workers
        self.on_task_start = on_task_start
        self.on_task_complete = on_task_complete
        self.on_task_error = on_task_error

        self._queue: asyncio.Queue = asyncio.Queue()
        self._tasks: List[VideoTask] = []
        self._workers: List[asyncio.Task] = []
        self._running = False

    def add_task(self, task: VideoTask) -> None:
        """Add task to queue."""
        self._tasks.append(task)
        self._queue.put_nowait(task)
        logger.debug(f"Task added: {task.video_path.name}")

    def add_video(
        self,
        video_path: Path,
        target_langs: List[str],
        output_dir: Optional[Path] = None,
        **options,
    ) -> VideoTask:
        """
        Add video processing task.

        Args:
            video_path: Path to video file.
            target_langs: Target languages.
            output_dir: Output directory.
            **options: Additional options.

        Returns:
            Created task object.
        """
        task = VideoTask(
            video_path=video_path,
            target_langs=target_langs,
            output_dir=output_dir or video_path.parent,
            options=options,
        )
        self.add_task(task)
        return task

    async def _worker(
        self,
        worker_id: int,
        process_func: Callable[[VideoTask], Any],
    ) -> None:
        """Worker coroutine for processing tasks."""
        queue = self._queue
        while self._running:
            try:
                task = await asyncio.wait_for(
                    queue.get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue

            try:
                task.status = TaskStatus.PROCESSING
                task.started_at = datetime.now()

                if self.on_task_start:
                    self.on_task_start(task)

                logger.info(f"[Worker {worker_id}] Processing: {task.video_path.name}")

                # Run CPU-intensive task in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, process_func, task)

                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()

                if self.on_task_complete:
                    self.on_task_complete(task)

                logger.info(f"[Worker {worker_id}] Completed: {task.video_path.name}")

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()

                if self.on_task_error:
                    self.on_task_error(task, e)

                logger.error(f"[Worker {worker_id}] Failed: {task.video_path.name} - {e}")

            finally:
                queue.task_done()

    async def run(self, process_func: Callable[[VideoTask], Any]) -> List[VideoTask]:
        """
        Run all tasks in queue.

        Args:
            process_func: Task processing function.

        Returns:
            List of all tasks with results.
        """
        self._running = True

        # Start workers
        self._workers = [
            asyncio.create_task(self._worker(i, process_func)) for i in range(self.max_workers)
        ]

        # Wait for all tasks to complete
        await self._queue.join()

        # Stop workers
        self._running = False
        await asyncio.gather(*self._workers, return_exceptions=True)

        return self._tasks

    @property
    def pending_count(self) -> int:
        """Get number of pending tasks."""
        return sum(1 for t in self._tasks if t.status == TaskStatus.PENDING)

    @property
    def completed_count(self) -> int:
        """Get number of completed tasks."""
        return sum(1 for t in self._tasks if t.status == TaskStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        """Get number of failed tasks."""
        return sum(1 for t in self._tasks if t.status == TaskStatus.FAILED)


def run_batch_sync(
    tasks: List[VideoTask],
    process_func: Callable[[VideoTask], Any],
    max_workers: int = 2,
    on_task_start: Optional[Callable[[VideoTask], None]] = None,
    on_task_complete: Optional[Callable[[VideoTask], None]] = None,
    on_task_error: Optional[Callable[[VideoTask, Exception], None]] = None,
) -> List[VideoTask]:
    """
    Run batch processing synchronously.

    Args:
        tasks: List of tasks to process.
        process_func: Task processing function.
        max_workers: Maximum concurrent tasks.
        on_task_start: Callback when task starts.
        on_task_complete: Callback when task completes.
        on_task_error: Callback when task fails.

    Returns:
        List of all tasks with results.
    """
    queue = TaskQueue(
        max_workers=max_workers,
        on_task_start=on_task_start,
        on_task_complete=on_task_complete,
        on_task_error=on_task_error,
    )

    for task in tasks:
        queue.add_task(task)

    return asyncio.run(queue.run(process_func))
