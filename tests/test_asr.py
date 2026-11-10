"""`create_backend`: asr.backend picks the backend, and that backend's config section reaches it.
Both backends hold an unload until a transcription on another thread has finished."""

import threading
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from subloom.core import qwen3_asr
from subloom.core.asr import TranscriptionInfo, create_backend
from subloom.models.config import AppConfig


def test_an_unknown_backend_fails_validation_and_the_factory():
    config = AppConfig()
    config.asr.backend = "nope"

    with pytest.raises(ValueError, match="asr.backend"):
        config.validate()
    with pytest.raises(ValueError, match="nope"):
        create_backend(config)


def test_the_whisper_backend_takes_its_settings_from_the_whisper_section():
    pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")
    from subloom.core.transcriber import Transcriber

    config = AppConfig()
    config.whisper.beam_size = 3
    config.whisper.vad_filter = False
    config.whisper.batch_size = 4
    config.whisper.speech_pad_ms = 123

    backend = create_backend(config)

    assert isinstance(backend, Transcriber)
    assert (backend.beam_size, backend.vad_filter, backend.batch_size) == (3, False, 4)
    assert backend.vad_parameters["speech_pad_ms"] == 123


def test_overrides_replace_the_configured_values():
    pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")

    backend = create_backend(AppConfig(), model_name="tiny", beam_size=1)

    assert (backend.model_name, backend.beam_size) == ("tiny", 1)


class _Gate:
    """Holds whatever runs inside a backend's lock until released."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def hold(self, value):
        self.entered.set()
        assert self.release.wait(5)
        return value


class _SilentQwen:
    def transcribe(self, audio, language=None, return_time_stamps=False):
        return [SimpleNamespace(language="", text="", time_stamps=None) for _ in audio]


def _held_backend(name, monkeypatch, gate):
    config = AppConfig()
    config.asr.backend = name
    if name == "whisper":
        pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")
        config.whisper.use_whisperx = False
        backend = create_backend(config)
        info = TranscriptionInfo("en", 0.9, 4.0)
        held = lambda path, language=None: gate.hold(([], info))  # noqa: E731
        monkeypatch.setattr(backend, "_transcribe_faster_whisper", held)
    else:
        backend = create_backend(config)
        monkeypatch.setattr(backend, "_load", lambda: gate.hold(_SilentQwen()))
        monkeypatch.setattr(
            qwen3_asr, "_split_into_chunks", lambda samples, seconds: [(samples, 0.0)]
        )
    return backend


@pytest.mark.parametrize("name", ["whisper", "qwen3_asr"])
def test_an_unload_waits_for_a_transcription_in_progress(name, monkeypatch, tmp_path):
    wav = tmp_path / "audio.wav"
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(np.zeros(16000 * 4, dtype=np.int16).tobytes())
    try:  # a first import inside unload_model takes seconds and would pass for the wait
        import torch  # noqa: F401
    except ImportError:
        pass
    gate = _Gate()
    backend = _held_backend(name, monkeypatch, gate)
    finished = []
    transcribing = threading.Thread(target=lambda: finished.append(backend.transcribe(wav)))
    transcribing.start()
    assert gate.entered.wait(5)

    unloading = threading.Thread(target=backend.unload_model)
    unloading.start()
    unloading.join(0.2)
    held = unloading.is_alive()
    gate.release.set()
    transcribing.join(30)
    unloading.join(30)

    assert held
    assert finished and not unloading.is_alive()
