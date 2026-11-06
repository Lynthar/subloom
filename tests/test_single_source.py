"""Things with exactly one home in src/. A second copy — a re-pasted progress block, a
hand-typed VAD default — fails here."""

from pathlib import Path

import pytest

from subloom.models.config import WhisperConfig

SRC = Path(__file__).resolve().parent.parent / "src" / "subloom"


def test_download_progress_bar_lives_only_in_utils_progress():
    # Six copies of this Rich block drifted (--quiet ignored, stalled status text) before
    # they were folded into download_asr_model_with_progress / pull_ollama_with_progress.
    hits = set()
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "DownloadColumn(" in text or "last_completed" in text:
            hits.add(path.relative_to(SRC).as_posix())
    assert hits == {"utils/progress.py"}


def test_vad_defaults_derive_from_whisper_config():
    pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")
    from subloom.core.transcriber import Transcriber

    expected = {
        "speech_pad_ms": WhisperConfig().speech_pad_ms,
        "min_silence_duration_ms": WhisperConfig().min_silence_duration_ms,
    }
    assert Transcriber.DEFAULT_VAD_PARAMETERS == expected
    assert Transcriber.VAD_PRESETS["default"] is Transcriber.DEFAULT_VAD_PARAMETERS
