"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status

from .. import __version__
from ..models.config import AppConfig
from .auth import noop_auth, require_token
from .jobs import (
    Job,
    JobRunner,
    JobStore,
    new_job_id,
    validate_video_path,
)
from .models import (
    HealthResponse,
    JobAccepted,
    JobDegradation,
    JobOutput,
    JobRequest,
    JobResponse,
)
from .processing import TranscriberHolder, make_processor

logger = logging.getLogger(__name__)


def create_app(
    config: Optional[AppConfig] = None,
    *,
    max_workers: int = 1,
    require_auth: bool = True,
    max_pending: int = 100,
) -> FastAPI:
    config = config or AppConfig.load()
    holder = TranscriberHolder(config)
    store = JobStore()
    runner = JobRunner(store, make_processor(config, holder), max_workers=max_workers)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await runner.start()
        try:
            yield
        finally:
            await runner.stop()

    app = FastAPI(
        title="subloom",
        version=__version__,
        lifespan=lifespan,
    )

    if not require_auth:
        # Wire the auth dependency to a no-op so endpoints don't need to
        # branch on whether auth is enabled.
        app.dependency_overrides[require_token] = noop_auth

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        stats = await store.stats()
        # A dead worker pool means accepted jobs will never run — that must
        # not look healthy. 503 here is what lets a supervisor or client
        # notice, instead of polling a forever-'pending' job against a 200.
        if runner.is_running and runner.alive_workers == 0:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No job workers alive; accepted jobs will never run",
            )
        return HealthResponse(
            version=__version__,
            queue_pending=stats["pending"],
            queue_processing=stats["processing"],
            transcriber_loaded=holder.is_loaded,
            workers_alive=runner.alive_workers,
        )

    @app.post(
        "/jobs",
        response_model=JobAccepted,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_token)],
    )
    async def submit_job(payload: JobRequest) -> JobAccepted:
        err = validate_video_path(payload.video_path)
        if err:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err)

        # Backpressure for pending jobs: the store's max_jobs only evicts *terminal* ones. The check
        # is approximate — concurrent submits can overshoot, which is fine for a cap meant to stop
        # runaway queues rather than to account precisely.
        stats = await store.stats()
        if stats["pending"] >= max_pending:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Job queue is full ({max_pending} pending); retry later",
            )

        keep_original, bilingual = config.output.resolve_flags(
            keep_original=payload.keep_original, bilingual=payload.bilingual
        )
        job = Job(
            job_id=new_job_id(),
            video_path=payload.video_path,
            target_languages=payload.target_languages,
            source_language=payload.source_language,
            bilingual=bilingual,
            keep_original=keep_original,
        )
        await runner.submit(job)
        return JobAccepted(job_id=job.job_id, status=job.status)

    @app.get(
        "/jobs/{job_id}",
        response_model=JobResponse,
        dependencies=[Depends(require_token)],
    )
    async def get_job(job_id: str) -> JobResponse:
        job = await store.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        return JobResponse(
            job_id=job.job_id,
            status=job.status,
            video_path=job.video_path,
            target_languages=job.target_languages,
            source_language=job.source_language,
            bilingual=job.bilingual,
            keep_original=job.keep_original,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error,
            outputs=[JobOutput(language=o["language"], path=o["path"]) for o in job.outputs],
            degradations=[JobDegradation(code=d.code, message=d.message) for d in job.degradations],
        )

    return app
