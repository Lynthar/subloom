"""A cached Whisper model loads without asking the Hub, and the cache is checked where
faster-whisper looks. Skips when faster-whisper is absent."""

import pytest

pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")

from faster_whisper.utils import _MODELS  # noqa: E402

from subtitle_forge.core import transcriber as transcriber_module  # noqa: E402
from subtitle_forge.core.transcriber import WHISPER_HF_REPOS, Transcriber  # noqa: E402
from subtitle_forge.models.config import AppConfig  # noqa: E402


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


def test_every_model_name_maps_to_the_repo_faster_whisper_loads():
    assert {name: _MODELS.get(name) for name in WHISPER_HF_REPOS} == WHISPER_HF_REPOS
