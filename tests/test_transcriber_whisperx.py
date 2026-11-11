"""WhisperX chunks: shorter for Chinese, Japanese or Korean, given or detected, whose chunks it
never splits into sentences, and dropped when made up. WhisperX, CTranslate2 and torch are fakes;
skips when faster-whisper is absent."""

import sys
import types

import numpy as np
import pytest

pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")

from subloom.core import transcriber as transcriber_module  # noqa: E402
from subloom.core.transcriber import Transcriber  # noqa: E402
from subloom.models.config import AppConfig  # noqa: E402

SECONDS = 4


@pytest.fixture()
def whisperx_calls(monkeypatch, tmp_path):
    """Run the WhisperX path on fakes. Yields (transcribe, kwargs of each WhisperX call, chunks
    WhisperX returns, the language it detects: "ja" until set); a chunk's no-speech probability is
    the loudest sample of its audio."""
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=bool))
    )
    calls, chunks, detected = [], [], ["ja"]
    ctranslate2 = types.SimpleNamespace(
        is_multilingual=True,
        generate=lambda features, prompts, **kwargs: [
            types.SimpleNamespace(no_speech_prob=float(features.max()))
        ],
    )

    class _Pipeline:
        model = types.SimpleNamespace(hf_tokenizer=None, encode=lambda f: f, model=ctranslate2)

        def detect_language(self, audio):
            return detected[0]

        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            return {"language": kwargs["language"] or detected[0], "segments": list(chunks)}

        def preprocess(self, inputs):
            return {"inputs": inputs["inputs"]}

    audio = np.zeros(SECONDS * 16000, dtype=np.float32)
    audio[2 * 16000 : 3 * 16000] = 0.6
    whisperx = types.SimpleNamespace(
        load_model=lambda *args, **kwargs: _Pipeline(), load_audio=lambda path: audio
    )
    monkeypatch.setattr(transcriber_module, "whisperx", whisperx, raising=False)
    monkeypatch.setattr(transcriber_module, "WHISPERX_AVAILABLE", True)
    monkeypatch.setattr(
        transcriber_module, "Tokenizer", lambda *a, **k: types.SimpleNamespace(sot_sequence=[0])
    )
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF")
    cfg = AppConfig().whisper
    cfg.whisperx_align = False
    backend = Transcriber.from_config(cfg)
    yield (lambda language: backend.transcribe(path, language=language)), calls, chunks, detected


@pytest.mark.parametrize(
    "language, seconds", [("ja", 7), ("zh", 7), ("ko", 7), ("yue", 7), ("en", 20)]
)
def test_chunk_length_follows_the_given_language(whisperx_calls, language, seconds):
    transcribe, calls, _, _ = whisperx_calls

    transcribe(language)

    assert calls[0] == {"language": language, "batch_size": 8, "chunk_size": seconds}


@pytest.mark.parametrize("language, seconds", [("ja", 7), ("zh", 7), ("en", 20)])
def test_chunk_length_follows_a_detected_language_too(whisperx_calls, language, seconds):
    transcribe, calls, _, detected = whisperx_calls
    detected[0] = language

    _, info = transcribe(None)

    assert calls[0] == {"language": language, "batch_size": 8, "chunk_size": seconds}
    assert info.language == language


def test_made_up_chunks_are_dropped(whisperx_calls):
    transcribe, _, chunks, _ = whisperx_calls
    chunks.extend(
        {"start": float(i), "end": i + 1.0, "text": text}
        for i, text in enumerate(
            ["おやすみなさい", "ご視聴ありがとうございました", "おやすみなさい", "気持ちいい"]
        )
    )

    segments, _ = transcribe("ja")

    # Only the third second sounds like no speech, so only that pleasantry was made up.
    assert [(s.start, s.text) for s in segments] == [(0.0, "おやすみなさい"), (3.0, "気持ちいい")]
