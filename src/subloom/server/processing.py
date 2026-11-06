"""Video processing entry point for the HTTP server.

Wraps the shared `core/pipeline.run_pipeline` with server-specific lifecycle:
a TranscriberHolder so the ASR model stays resident across jobs, and
job-state mutation so callers polling /jobs/{id} see the detected language
as soon as transcription completes.

The runner invokes `make_processor(...)`'s closure inside
`loop.run_in_executor`, so this all runs off the event loop.
"""

import logging
import threading
from pathlib import Path
from typing import List, Optional

from ..core.asr import AsrBackend, create_backend
from ..core.pipeline import PipelineOutput, StructureCheckError, run_pipeline
from ..core.translator import create_translator
from ..models.config import AppConfig
from .jobs import Job

logger = logging.getLogger(__name__)


class TranscriberHolder:
    """Lazy, thread-safe holder for the single ASR backend instance.

    Build is deferred to first use so the server boots even if the backend isn't
    available yet (helpful for diagnostics). After the first build the same
    instance is reused for every job — the model stays resident in VRAM.
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._transcriber: Optional[AsrBackend] = None
        self._lock = threading.Lock()

    def get(self) -> AsrBackend:
        # Double-checked locking — fast path skips the lock once initialized.
        if self._transcriber is not None:
            return self._transcriber
        with self._lock:
            if self._transcriber is not None:
                return self._transcriber
            backend = self._config.asr.backend
            logger.info("Building the %s ASR backend", backend)
            t = create_backend(self._config)
            if not t.is_model_cached():
                # Server should not run interactive download. Fail with a
                # clear instruction instead.
                raise RuntimeError(
                    f"The {backend} model is not cached. "
                    f"Run `subloom transcribe <some-video>` once on this "
                    f"machine to download it, then restart the server."
                )
            self._transcriber = t
            return t

    @property
    def is_loaded(self) -> bool:
        return self._transcriber is not None


def make_processor(config: AppConfig, holder: TranscriberHolder):
    """Returns a sync `(Job) -> List[dict]` callable bound to the given config."""

    def process(job: Job) -> list:
        return _run_job(job, config, holder.get())

    return process


def _run_job(job: Job, config: AppConfig, transcriber: AsrBackend) -> list:
    video_path = Path(job.video_path)

    translator = create_translator(config)

    try:
        result = run_pipeline(
            video_path,
            config,
            transcriber=transcriber,
            translator=translator,
            target_languages=job.target_languages,
            output_dir=video_path.parent,
            source_language=job.source_language,
            keep_original=job.keep_original,
            bilingual=job.bilingual,
            # No hooks — server runs silently; logging inside translator/
            # transcriber covers operational visibility.
        )
    except StructureCheckError as e:
        # The job fails, but the files are on disk: list them so the caller can find them.
        job.outputs = _describe(e.outputs)
        raise

    # Surface the detected language back through the job record so callers
    # polling /jobs/{id} see it once transcription completes.
    job.source_language = result.detected_language
    job.degradations = result.degradations

    return _describe(result.outputs)


def _describe(outputs: List[PipelineOutput]) -> List[dict]:
    return [{"language": o.language, "path": str(o.path)} for o in outputs]
