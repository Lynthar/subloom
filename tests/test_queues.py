"""Contracts of the two async queues, exercised from outside any event loop.

Both entry points build these objects before a loop exists — JobRunner/JobStore
in create_app(), TaskQueue before asyncio.run() — so the tests do the same.
"""

import asyncio
from pathlib import Path

import pytest

from subloom.core.queue import run_batch_sync
from subloom.models.task import VideoTask
from subloom.server.jobs import Job, JobRunner, JobStore


def test_job_runner_submit_before_start_is_an_error():
    runner = JobRunner(JobStore(), lambda job: [], max_workers=1)
    job = Job(job_id="x", video_path="v.mp4", target_languages=["zh"])
    with pytest.raises(RuntimeError):
        asyncio.run(runner.submit(job))


def test_run_batch_sync_completes_and_returns_statuses():
    tasks = [
        VideoTask(video_path=Path("a.mp4"), target_langs=["zh"], output_dir=Path(".")),
        VideoTask(video_path=Path("b.mp4"), target_langs=["zh"], output_dir=Path(".")),
    ]
    processed = []
    results = run_batch_sync(tasks, lambda t: processed.append(t.video_path.name), max_workers=2)
    assert [t.status.value for t in results] == ["completed", "completed"]
    assert sorted(processed) == ["a.mp4", "b.mp4"]


def test_run_batch_sync_marks_failures():
    tasks = [
        VideoTask(video_path=Path("a.mp4"), target_langs=["zh"], output_dir=Path(".")),
    ]

    def boom(task):
        raise RuntimeError("ffmpeg exploded")

    results = run_batch_sync(tasks, boom, max_workers=1)
    assert results[0].status.value == "failed"
    assert "ffmpeg exploded" in results[0].error
