"""A cached Whisper model loads without asking the Hub, and the cache is checked where
faster-whisper looks. Skips when faster-whisper is absent."""

import os

import pytest

pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")

from faster_whisper.utils import _MODELS  # noqa: E402

from subloom.core import transcriber as transcriber_module  # noqa: E402
from subloom.core.transcriber import WHISPER_HF_REPOS, Transcriber  # noqa: E402
from subloom.models.config import AppConfig  # noqa: E402


@pytest.mark.parametrize("cached", [True, False])
def test_only_a_model_not_yet_cached_may_go_online(monkeypatch, cached):
    built = []
    monkeypatch.setattr(
        transcriber_module, "WhisperModel", lambda name, **kwargs: built.append(kwargs) or object()
    )
    backend = Transcriber.from_config(AppConfig().whisper)
    monkeypatch.setattr(backend, "is_model_cached", lambda: cached)

    backend.load_model()

    assert built[0]["local_files_only"] is cached


def test_transformers_is_told_not_to_convert_weights_on_load():
    assert os.environ.get("DISABLE_SAFETENSORS_CONVERSION") == "true"


def test_pyannote_is_told_not_to_send_telemetry():
    assert os.environ.get("PYANNOTE_METRICS_ENABLED") == "false"


def test_pyannote_itself_reads_its_telemetry_as_off():
    metrics = pytest.importorskip("pyannote.audio.telemetry.metrics")
    assert not metrics.is_metrics_enabled()


def test_every_model_name_maps_to_the_repo_faster_whisper_loads():
    assert {name: _MODELS.get(name) for name in WHISPER_HF_REPOS} == WHISPER_HF_REPOS
