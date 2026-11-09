"""Lines Whisper makes up over audio without speech are dropped; real lines that look like them
are kept. The transcriber test fakes faster-whisper's model and skips when it is absent."""

import types

import pytest

from subloom.core.hallucinations import NO_SPEECH_GATE, is_hallucination


def _never_asked():
    raise AssertionError("no_speech_prob was not needed")


@pytest.mark.parametrize(
    "text",
    [
        "ご視聴ありがとうございました",
        " ご視聴ありがとうございました。",
        "by H.",
        "-end-",
        "チャンネル登録よろしくお願いします！ご視聴ありがとうございました",
    ],
)
def test_outros_go_whatever_the_audio(text):
    assert is_hallucination(text, _never_asked)


@pytest.mark.parametrize(
    "text", ["おやすみなさい", "お疲れ様でした。", "おやすみなさい おやすみなさい", "バイバイ"]
)
def test_pleasantries_go_only_when_whisper_doubted_speech(text):
    assert is_hallucination(text, lambda: NO_SPEECH_GATE + 0.3)
    assert not is_hallucination(text, lambda: 0.02)


@pytest.mark.parametrize(
    "text",
    [
        "店長、お疲れ様でした。",
        "ありがとうございます",
        "ご視聴ありがとうございました、またね",
        "",
        "。",
    ],
)
def test_lines_with_anything_else_are_kept(text):
    assert not is_hallucination(text, lambda: 1.0)


def test_faster_whisper_drops_made_up_segments(monkeypatch, tmp_path):
    pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")
    from subloom.core.transcriber import Transcriber
    from subloom.models.config import AppConfig

    def segment(text, no_speech_prob):
        return types.SimpleNamespace(
            start=1.0, end=2.0, text=text, words=None, no_speech_prob=no_speech_prob
        )

    said = [
        segment(" お疲れ様でした。", 0.01),
        segment(" ご視聴ありがとうございました", 0.01),
        segment(" おやすみなさい", 0.6),
        segment(" 先に失礼します。", 0.01),
    ]
    info = types.SimpleNamespace(language="ja", language_probability=0.9, duration=3.0)
    model = types.SimpleNamespace(transcribe=lambda audio, **kwargs: (iter(said), info))
    cfg = AppConfig().whisper
    cfg.use_whisperx = False
    backend = Transcriber.from_config(cfg)
    monkeypatch.setattr(backend, "load_model", lambda: model)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")

    segments, _ = backend.transcribe(audio)

    assert [s.text for s in segments] == ["お疲れ様でした。", "先に失礼します。"]
