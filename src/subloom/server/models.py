"""HTTP API request/response models."""

from datetime import datetime
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from ..core.subtitle import validate_language_codes


class JobRequest(BaseModel):
    video_path: str = Field(
        ..., description="Absolute path to the video file as the server sees it"
    )
    # Each target language is one full LLM translation pass over the whole
    # video — cap the count so a single request can't queue unbounded work.
    target_languages: List[str] = Field(..., min_length=1, max_length=10)
    source_language: Optional[str] = Field(None, description="ISO 639-1 code; null = auto-detect")
    # null = the server's configured output.bilingual / output.keep_original.
    bilingual: Optional[bool] = None
    keep_original: Optional[bool] = None

    @field_validator("target_languages", "source_language")
    @classmethod
    def _filename_safe_language_codes(
        cls, value: Union[List[str], Optional[str]]
    ) -> Union[List[str], Optional[str]]:
        # Language codes end up in output filenames verbatim — reject path
        # metacharacters here so the caller gets a 422 at submit time instead
        # of a failed job minutes later (run_pipeline re-checks as backstop).
        if value is not None:
            validate_language_codes([value] if isinstance(value, str) else value)
        return value


class JobOutput(BaseModel):
    language: str
    path: str


class JobDegradation(BaseModel):
    code: str = Field(..., description="Stable identifier, e.g. alignment_failed")
    message: str


class JobResponse(BaseModel):
    job_id: str
    status: str
    video_path: str
    target_languages: List[str]
    source_language: Optional[str] = None
    bilingual: bool = False
    keep_original: bool = True
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    outputs: List[JobOutput] = []
    # Set on completion; non-empty means the job finished worse than configured.
    degradations: List[JobDegradation] = []


class JobAccepted(BaseModel):
    job_id: str
    status: str = "pending"


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    queue_pending: int
    queue_processing: int
    transcriber_loaded: bool
    # 0 before startup (lifespan not entered yet); when the runner is running
    # with 0 alive workers the endpoint returns 503 instead of this model.
    workers_alive: int = 0
