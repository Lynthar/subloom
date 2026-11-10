"""A Whisper model name loads the repo it names, with faster-whisper and with WhisperX.

faster-whisper's model class and WhisperX are fakes; skips when faster-whisper is absent."""

import sys
import types

import numpy as np
import pytest

pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")

from subloom.core import transcriber as transcriber_module  # noqa: E402
from subloom.core.transcriber import Transcriber  # noqa: E402
from subloom.models.config import AppConfig  # noqa: E402

KOTOBA = "kotoba-tech/kotoba-whisper-v2.0-faster"


def _backend(model, *, whisperx):
    cfg = AppConfig().whisper
    cfg.model, cfg.use_whisperx, cfg.device = model, whisperx, "cpu"
    return Transcriber.from_config(cfg)


@pytest.mark.parametrize(
    "model, repo",
    [
        ("large-v3", "Systran/faster-whisper-large-v3"),
        ("someone/their-model", "someone/their-model"),
    ],
)
def test_faster_whisper_loads_the_repo_a_name_stands_for(monkeypatch, model, repo):
    loaded = []
    monkeypatch.setattr(
        transcriber_module, "WhisperModel", lambda source, **kwargs: loaded.append(source)
    )

    _backend(model, whisperx=False).load_model()

    assert loaded == [repo]


def test_whisperx_loads_the_repo_a_name_stands_for(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=bool))
    )
    loaded = []

    class _Pipeline:
        def transcribe(self, audio, **kwargs):
            return {"language": "ja", "segments": []}

    whisperx = types.SimpleNamespace(
        load_model=lambda source, **kwargs: loaded.append(source) or _Pipeline(),
        load_audio=lambda path: np.zeros(16000, dtype=np.float32),
    )
    monkeypatch.setattr(transcriber_module, "whisperx", whisperx, raising=False)
    monkeypatch.setattr(transcriber_module, "WHISPERX_AVAILABLE", True)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")

    _backend("kotoba-whisper-v2.0", whisperx=True).transcribe(audio, language="ja")

    assert loaded == [KOTOBA]


@pytest.mark.parametrize("model", ["kotoba-whisper-v2.0", KOTOBA])
@pytest.mark.parametrize("asked, installed", [(False, True), (True, False)])
def test_kotoba_is_refused_where_whisperx_will_not_run(monkeypatch, model, asked, installed):
    monkeypatch.setattr(transcriber_module, "WHISPERX_AVAILABLE", installed)

    with pytest.raises(ValueError, match="only under WhisperX"):
        _backend(model, whisperx=asked)


def test_a_japanese_only_model_is_never_picked_for_you():
    assert "kotoba-whisper-v2.0" not in Transcriber.MODEL_VRAM_REQUIREMENTS
