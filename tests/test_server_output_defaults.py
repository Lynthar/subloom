"""POST /jobs without bilingual / keep_original takes the configured output section — the
server used to hard-code False / True and never read config.output. The processor is stubbed
(no Whisper, no ffmpeg); skips without the [serve] extra."""

import pytest

pytest.importorskip("fastapi", reason="server needs the [serve] extra")

from fastapi.testclient import TestClient  # noqa: E402

from subloom.models.config import AppConfig  # noqa: E402
from subloom.server import app as server_app  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(server_app, "make_processor", lambda config, holder: lambda job: [])
    config = AppConfig()
    config.output.bilingual = True
    config.output.keep_original = False
    with TestClient(server_app.create_app(config, require_auth=False)) as c:
        yield c


@pytest.fixture()
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not really a video")
    return path


def _submit(client, video, **fields):
    response = client.post(
        "/jobs", json={"video_path": str(video), "target_languages": ["zh"], **fields}
    )
    assert response.status_code == 202, response.text
    return client.get(f"/jobs/{response.json()['job_id']}").json()


def test_omitted_flags_fall_back_to_config_output(client, video):
    job = _submit(client, video)
    assert job["bilingual"] is True
    assert job["keep_original"] is False


def test_explicit_flags_override_config_output(client, video):
    job = _submit(client, video, bilingual=False, keep_original=True)
    assert job["bilingual"] is False
    assert job["keep_original"] is True
