"""WhisperX chunks: shorter for a given Chinese, Japanese or Korean, whose chunks it never splits
into sentences, and dropped when made up. WhisperX, CTranslate2 and torch are fakes; skips when
faster-whisper is absent."""

import logging
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
    WhisperX returns); a chunk's no-speech probability is the loudest sample of its audio."""
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=bool))
    )
    calls, chunks = [], []
    ctranslate2 = types.SimpleNamespace(
        is_multilingual=True,
        generate=lambda features, prompts, **kwargs: [
            types.SimpleNamespace(no_speech_prob=float(features.max()))
        ],
    )

    class _Pipeline:
        model = types.SimpleNamespace(hf_tokenizer=None, encode=lambda f: f, model=ctranslate2)

        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            return {"language": kwargs["language"] or "ja", "segments": list(chunks)}

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
    yield (lambda language: backend.transcribe(path, language=language)), calls, chunks


@pytest.mark.parametrize(
    "language, seconds", [("ja", 7), ("zh", 7), ("ko", 7), ("yue", 7), ("en", 20), (None, 20)]
)
def test_chunk_length_follows_the_given_language(whisperx_calls, language, seconds):
    transcribe, calls, _ = whisperx_calls

    transcribe(language)

    assert calls[0]["chunk_size"] == seconds


def test_a_detected_cjk_language_says_how_to_get_short_chunks(whisperx_calls, caplog):
    transcribe, _, _ = whisperx_calls

    with caplog.at_level(logging.INFO, logger=transcriber_module.__name__):
        transcribe(None)
        hinted = "give the source language" in caplog.text
        caplog.clear()
        transcribe("ja")

    assert hinted
    assert "give the source language" not in caplog.text


def test_made_up_chunks_are_dropped(whisperx_calls):
    transcribe, _, chunks = whisperx_calls
    chunks.extend(
        {"start": float(i), "end": i + 1.0, "text": text}
        for i, text in enumerate(
            ["おやすみなさい", "ご視聴ありがとうございました", "おやすみなさい", "気持ちいい"]
        )
    )

    segments, _ = transcribe("ja")

    # Only the third second sounds like no speech, so only that pleasantry was made up.
    assert [(s.start, s.text) for s in segments] == [(0.0, "おやすみなさい"), (3.0, "気持ちいい")]
