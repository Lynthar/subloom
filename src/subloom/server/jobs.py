"""In-memory job store and async job runner.

Single-worker by default — the GPU is the bottleneck and concurrent Whisper
invocations would just thrash VRAM. The runner spins up its workers in
`start()` and tears them down in `stop()`; both are called from the FastAPI
lifespan handler.
"""

import asyncio
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from ..models.degradation import Degradation

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    job_id: str
    video_path: str
    target_languages: List[str]
    source_language: Optional[str] = None
    bilingual: bool = False
    keep_original: bool = True

    status: str = "pending"  # pending | processing | completed | failed
    created_at: datetime = field(default_factory=_utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    outputs: List[dict] = field(default_factory=list)
    degradations: List[Degradation] = field(default_factory=list)


class JobStore:
    """Thread-safe bounded in-memory job storage.

    Uses an OrderedDict to evict the oldest *terminal* job when the cap is
    reached. We never evict pending or processing jobs — losing track of an
    in-flight job would leave the GPU running with no visible status.
    """

    def __init__(self, max_jobs: int = 500):
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._max = max_jobs
        self._lock = asyncio.Lock()

    async def add(self, job: Job) -> None:
        async with self._lock:
            self._jobs[job.job_id] = job
            self._evict_if_needed()

    async def get(self, job_id: str) -> Optional[Job]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def stats(self) -> dict:
        async with self._lock:
            pending = sum(1 for j in self._jobs.values() if j.status == "pending")
            processing = sum(1 for j in self._jobs.values() if j.status == "processing")
            return {"pending": pending, "processing": processing, "total": len(self._jobs)}

    def _evict_if_needed(self) -> None:
        if len(self._jobs) <= self._max:
            return
        for jid in list(self._jobs.keys()):
            if len(self._jobs) <= self._max:
                break
            if self._jobs[jid].status in ("completed", "failed"):
                del self._jobs[jid]


JobProcessor = Callable[[Job], List[dict]]
"""Sync function that processes a job and returns its output list."""


class JobRunner:
    """Async worker pool that pulls jobs off a queue and runs them in a thread."""

    def __init__(self, store: JobStore, processor: JobProcessor, max_workers: int = 1):
        self._store = store
        self._processor = processor
        self._max_workers = max_workers
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        loop = asyncio.get_running_loop()
        for i in range(self._max_workers):
            worker = loop.create_task(self._worker(i))
            worker.add_done_callback(self._on_worker_done)
            self._workers.append(worker)
        logger.info("JobRunner started with %d worker(s)", self._max_workers)

    @staticmethod
    def _on_worker_done(task: "asyncio.Task") -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Job worker died unexpectedly: %r", exc)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def alive_workers(self) -> int:
        return sum(1 for w in self._workers if not w.done())

    async def stop(self) -> None:
        # Cancelling a worker task does NOT stop a job already inside run_in_executor — threads
        # cannot be killed, so it keeps the GPU busy and may still write output after the job is
        # failed.
        if not self._running:
            return
        self._running = False
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("JobRunner stopped")

    async def submit(self, job: Job) -> None:
        # Refuses after stop() as well as before start(): the workers are gone
        # either way, so an accepted job would sit in 'pending' forever.
        if not self._running:
            raise RuntimeError("JobRunner is not running; call start() before submit()")
        await self._store.add(job)
        await self._queue.put(job)

    async def _worker(self, idx: int) -> None:
        loop = asyncio.get_running_loop()
        queue = self._queue
        while True:
            try:
                job = await queue.get()
            except asyncio.CancelledError:
                return

            try:
                job.status = "processing"
                job.started_at = _utcnow()
                logger.info("[worker %d] processing job %s (%s)", idx, job.job_id, job.video_path)

                outputs = await loop.run_in_executor(None, self._processor, job)

                job.outputs = outputs
                job.status = "completed"
                logger.info(
                    "[worker %d] completed job %s (%d outputs)",
                    idx,
                    job.job_id,
                    len(outputs),
                )
            except asyncio.CancelledError:
                # Server shutdown mid-job. Mark failed so the job doesn't sit
                # in 'processing' forever after restart.
                job.status = "failed"
                job.error = "Server shut down before job completed"
                raise
            except Exception as e:  # noqa: BLE001
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
                logger.exception("[worker %d] job %s failed", idx, job.job_id)
            finally:
                job.completed_at = _utcnow()
                queue.task_done()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()


def new_job_id() -> str:
    return str(uuid.uuid4())


def validate_video_path(path: str) -> Optional[str]:
    """Returns an error message if the path is unusable, else None.

    This is the server-side health check that A++ promises: we surface
    `path-not-found` immediately at submit time, not minutes later when the
    worker picks the job up.
    """
    p = Path(path)
    if not p.is_absolute():
        return f"video_path must be absolute, got: {path}"
    if not p.exists():
        return f"video_path does not exist on this server: {path}"
    if not p.is_file():
        return f"video_path is not a file: {path}"
    # Reject non-video files here too — otherwise a README.md is accepted with
    # 202 and only fails minutes later when the worker reaches ffmpeg.
    from ..core.audio import AudioExtractor

    suffix = p.suffix.lower()
    if suffix not in AudioExtractor.SUPPORTED_VIDEO_FORMATS:
        supported = ", ".join(sorted(AudioExtractor.SUPPORTED_VIDEO_FORMATS))
        return f"unsupported video format {suffix or '(none)'}; supported: {supported}"
    return None
