"""A cached Whisper model, and WhisperX's alignment model, load without asking the Hub, and the
cache is checked where faster-whisper looks. Skips when faster-whisper is absent."""

import os
import sys
import types

import numpy as np
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


@pytest.mark.parametrize("cached", [True, False])
def test_whisperx_asks_the_hub_only_for_models_not_yet_cached(monkeypatch, tmp_path, cached):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    if cached:
        (snapshot / "pytorch_model.bin").write_bytes(b"")
    hub = types.SimpleNamespace(
        snapshot_download=lambda repo, local_files_only=False, **kwargs: (
            str(snapshot) if (repo, local_files_only) == ("org/align-ja", True) else None
        )
    )
    loads = {}

    class _Pipeline:
        def transcribe(self, audio, **kwargs):
            return {
                "language": "ja",
                "segments": [{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
            }

    def load_model(repo, **kwargs):
        loads["model"] = kwargs["local_files_only"]
        return _Pipeline()

    def load_align_model(**kwargs):
        loads["align"] = kwargs["model_cache_only"]
        return object(), {}

    whisperx = types.SimpleNamespace(
        load_model=load_model,
        load_audio=lambda path: np.zeros(16000, dtype=np.float32),
        load_align_model=load_align_model,
        align=lambda segments, *args, **kwargs: {"segments": segments},
    )
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=bool))
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(
        sys.modules,
        "whisperx.alignment",
        types.SimpleNamespace(DEFAULT_ALIGN_MODELS_HF={"ja": "org/align-ja"}),
    )
    monkeypatch.setattr(transcriber_module, "whisperx", whisperx, raising=False)
    monkeypatch.setattr(transcriber_module, "WHISPERX_AVAILABLE", True)
    backend = Transcriber.from_config(AppConfig().whisper)
    monkeypatch.setattr(backend, "is_model_cached", lambda: cached)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")

    backend.transcribe(audio, language="ja")

    assert loads == {"model": cached, "align": cached}


def test_transformers_is_told_not_to_convert_weights_on_load():
    assert os.environ.get("DISABLE_SAFETENSORS_CONVERSION") == "true"


def test_pyannote_is_told_not_to_send_telemetry():
    assert os.environ.get("PYANNOTE_METRICS_ENABLED") == "false"


def test_pyannote_itself_reads_its_telemetry_as_off():
    metrics = pytest.importorskip("pyannote.audio.telemetry.metrics")
    assert not metrics.is_metrics_enabled()


def test_a_name_faster_whisper_knows_means_the_repo_it_would_load():
    shared = {name: repo for name, repo in WHISPER_HF_REPOS.items() if name in _MODELS}
    assert shared == {name: _MODELS[name] for name in shared}
    assert set(WHISPER_HF_REPOS) - set(_MODELS) == {"kotoba-whisper-v2.0"}
