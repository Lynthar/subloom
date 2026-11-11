"""The Whisper backend reports its fallbacks with each transcription, not only in the log.

faster-whisper's model, WhisperX and torch are fakes; skips when faster-whisper is absent."""

import sys
import types

import numpy as np
import pytest

pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")

from subloom.core import transcriber as transcriber_module  # noqa: E402
from subloom.core.transcriber import Transcriber  # noqa: E402
from subloom.models.config import AppConfig  # noqa: E402


@pytest.fixture()
def audio(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF")
    return path


class _FasterWhisperModel:
    def transcribe(self, audio, **kwargs):
        info = types.SimpleNamespace(language="en", language_probability=0.9, duration=1.0)
        return iter([]), info


def test_missing_whisperx_is_reported_with_every_transcription(monkeypatch, audio):
    monkeypatch.setattr(transcriber_module, "WHISPERX_AVAILABLE", False)
    backend = Transcriber.from_config(AppConfig().whisper)
    monkeypatch.setattr(backend, "load_model", lambda: _FasterWhisperModel())

    first = backend.transcribe(audio)[1].degradations
    second = backend.transcribe(audio)[1].degradations

    assert [d.code for d in first] == ["whisperx_missing"]
    assert second == first


def test_whisperx_reports_a_cpu_fallback_and_a_failed_alignment(monkeypatch, audio):
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch)

    class _Model:
        def detect_language(self, audio):
            return "en"

        def transcribe(self, audio, **kwargs):
            return {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "Hi"}]}

    def load_align_model(**kwargs):
        raise RuntimeError("no align model for en")

    whisperx = types.SimpleNamespace(
        load_model=lambda *args, **kwargs: _Model(),
        load_audio=lambda path: np.zeros(16000, dtype=np.float32),
        load_align_model=load_align_model,
    )
    monkeypatch.setattr(transcriber_module, "whisperx", whisperx, raising=False)
    monkeypatch.setitem(
        sys.modules, "whisperx.alignment", types.SimpleNamespace(DEFAULT_ALIGN_MODELS_HF={})
    )
    monkeypatch.setattr(transcriber_module, "WHISPERX_AVAILABLE", True)

    segments, info = Transcriber.from_config(AppConfig().whisper).transcribe(audio)

    assert [s.text for s in segments] == ["Hi"]
    assert [d.code for d in info.degradations] == ["cpu_fallback", "alignment_failed"]
    assert "no align model for en" in info.degradations[1].message
