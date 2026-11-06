"""GET /jobs/{id} lists a finished job's degradations as code + message pairs.

The processor is stubbed (no Whisper, no ffmpeg); skips without the [serve] extra."""

import time

import pytest

pytest.importorskip("fastapi", reason="server needs the [serve] extra")

from fastapi.testclient import TestClient  # noqa: E402

from subloom.models.config import AppConfig  # noqa: E402
from subloom.models.degradation import Degradation  # noqa: E402
from subloom.server import app as server_app  # noqa: E402

CPU = Degradation("cpu_fallback", "CUDA is unavailable; ran on the CPU")


def _degraded_processor(config, holder):
    def process(job):
        job.degradations = [CPU]
        return []

    return process


def test_a_finished_job_lists_its_degradations(monkeypatch, tmp_path):
    monkeypatch.setattr(server_app, "make_processor", _degraded_processor)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")

    with TestClient(server_app.create_app(AppConfig(), require_auth=False)) as client:
        submitted = client.post(
            "/jobs", json={"video_path": str(video), "target_languages": ["zh"]}
        )
        assert submitted.status_code == 202, submitted.text
        url = f"/jobs/{submitted.json()['job_id']}"
        deadline = time.monotonic() + 5
        while (job := client.get(url).json())["status"] != "completed":
            assert time.monotonic() < deadline, job
            time.sleep(0.02)

    assert job["degradations"] == [{"code": CPU.code, "message": CPU.message}]
