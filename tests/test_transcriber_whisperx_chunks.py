"""WhisperX cuts shorter chunks for a given Chinese, Japanese or Korean, whose chunks it never
splits into sentences. WhisperX and torch are fakes; skips when faster-whisper is absent."""

import logging
import sys
import types

import numpy as np
import pytest

pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")

from subloom.core import transcriber as transcriber_module  # noqa: E402
from subloom.core.transcriber import Transcriber  # noqa: E402
from subloom.models.config import AppConfig  # noqa: E402


@pytest.fixture()
def whisperx_calls(monkeypatch, tmp_path):
    """Run the WhisperX path on fakes; yields (transcribe, the kwargs of each WhisperX call)."""
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=bool))
    )
    calls = []

    class _Model:
        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            return {"language": kwargs["language"] or "ja", "segments": []}

    whisperx = types.SimpleNamespace(
        load_model=lambda *args, **kwargs: _Model(),
        load_audio=lambda path: np.zeros(16000, dtype=np.float32),
    )
    monkeypatch.setattr(transcriber_module, "whisperx", whisperx, raising=False)
    monkeypatch.setattr(transcriber_module, "WHISPERX_AVAILABLE", True)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")
    backend = Transcriber.from_config(AppConfig().whisper)
    yield (lambda language: backend.transcribe(audio, language=language)), calls


@pytest.mark.parametrize(
    "language, seconds", [("ja", 7), ("zh", 7), ("ko", 7), ("yue", 7), ("en", 20), (None, 20)]
)
def test_chunk_length_follows_the_given_language(whisperx_calls, language, seconds):
    transcribe, calls = whisperx_calls

    transcribe(language)

    assert calls[0]["chunk_size"] == seconds


def test_a_detected_cjk_language_says_how_to_get_short_chunks(whisperx_calls, caplog):
    transcribe, _ = whisperx_calls

    with caplog.at_level(logging.INFO, logger=transcriber_module.__name__):
        transcribe(None)
        hinted = "give the source language" in caplog.text
        caplog.clear()
        transcribe("ja")

    assert hinted
    assert "give the source language" not in caplog.text
