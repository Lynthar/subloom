"""Without VAD, faster-whisper still decodes the whole audio, guarded against hallucination loops.

faster-whisper's model, batched pipeline and audio decoding are fakes; skips when it is absent."""

import types

import numpy as np
import pytest

pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")

from subtitle_forge.core import transcriber as transcriber_module  # noqa: E402
from subtitle_forge.core.transcriber import Transcriber, whole_audio_clips  # noqa: E402
from subtitle_forge.models.config import AppConfig  # noqa: E402

_INFO = types.SimpleNamespace(language="ja", language_probability=0.9, duration=65.0)


class _Model:
    feature_extractor = types.SimpleNamespace(sampling_rate=16000)

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return iter([]), _INFO


@pytest.fixture()
def audio(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF")
    return path


def _backend(monkeypatch, *, vad_filter, batch_size=None):
    cfg = AppConfig().whisper
    cfg.use_whisperx, cfg.vad_filter, cfg.batch_size = False, vad_filter, batch_size
    backend = Transcriber.from_config(cfg)
    model = _Model()
    monkeypatch.setattr(backend, "load_model", lambda: model)
    return backend, model


def test_without_vad_whisper_neither_conditions_on_itself_nor_keeps_silence_hallucinations(
    monkeypatch, audio
):
    backend, model = _backend(monkeypatch, vad_filter=False)
    backend.transcribe(audio)
    assert model.calls[0]["condition_on_previous_text"] is False
    assert model.calls[0]["hallucination_silence_threshold"] > 0


def test_with_vad_whisper_keeps_its_own_decoding_defaults(monkeypatch, audio):
    backend, model = _backend(monkeypatch, vad_filter=True)
    backend.transcribe(audio)
    assert "condition_on_previous_text" not in model.calls[0]
    assert "hallucination_silence_threshold" not in model.calls[0]


def test_batched_decoding_without_vad_covers_the_whole_audio_in_timestamped_lines(
    monkeypatch, audio
):
    pipelines = []

    class _Pipeline(_Model):
        def __init__(self, model):
            super().__init__()
            pipelines.append(self)

    monkeypatch.setattr(transcriber_module, "BatchedInferencePipeline", _Pipeline)
    monkeypatch.setattr(transcriber_module, "decode_audio", lambda path: np.zeros(65 * 16000))
    backend, _ = _backend(monkeypatch, vad_filter=False, batch_size=8)

    backend.transcribe(audio)

    call = pipelines[0].calls[0]
    assert call["clip_timestamps"] == whole_audio_clips(65.0)
    assert call["without_timestamps"] is False


@pytest.mark.parametrize("duration", [0.5, 30.0, 30.0001, 65.0, 9186.666687])
def test_whole_audio_clips_tile_the_audio_in_windows_of_at_most_30_s(duration):
    clips = whole_audio_clips(duration)
    assert clips[0]["start"] == 0.0 and clips[-1]["end"] == duration
    assert all(a["end"] == b["start"] for a, b in zip(clips, clips[1:]))
    assert all(min(duration, 15.0) <= c["end"] - c["start"] <= 30.0 for c in clips)
