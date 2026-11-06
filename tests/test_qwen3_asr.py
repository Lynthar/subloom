"""The Qwen3-ASR backend with qwen-asr, torch and the models faked: what segments and language it
reports, how aligned units map back onto the punctuated transcript, and the FlashAttention 2
fallback, which warns and carries on."""

import logging
import sys
import types
import wave
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytest

from subloom.core import qwen3_asr
from subloom.core.asr import create_backend
from subloom.core.qwen3_asr import (
    FLASH_ATTENTION_WARNING,
    Qwen3AsrBackend,
    qwen_language,
    segments_from_words,
    words_from_alignment,
)
from subloom.exceptions import TranscriptionError
from subloom.models.config import AppConfig
from subloom.models.subtitle import WordTiming


@dataclass
class _Item:
    text: str
    start_time: float
    end_time: float


@dataclass
class _Alignment:
    items: List[_Item]


@dataclass
class _Result:
    language: str
    text: str
    time_stamps: Optional[_Alignment]


def _words(*spec):
    return [WordTiming(word, start, end) for word, start, end in spec]


def test_language_codes_map_to_qwen_names():
    assert qwen_language("ja") == "Japanese"
    assert qwen_language("zh-TW") == "Chinese"
    with pytest.raises(TranscriptionError, match="does not support"):
        qwen_language("xx")


def test_aligned_units_take_back_their_punctuation_and_spacing():
    items = [_Item("Hello", 0.0, 0.4), _Item("world", 0.5, 0.9), _Item("I'm", 1.2, 1.4)]
    words = words_from_alignment('"Hello, world." I\'m', items, offset=10.0)
    assert [(w.word, w.start, w.end) for w in words] == [
        ('"Hello,', 10.0, 10.4),
        (' world."', 10.5, 10.9),
        (" I'm", 11.2, 11.4),
    ]


def test_cjk_units_are_single_characters_with_trailing_punctuation():
    items = [_Item(ch, i, i + 0.5) for i, ch in enumerate("你好世界")]
    words = words_from_alignment("你好。世界！", items, offset=0.0)
    assert [w.word for w in words] == ["你", "好。", "世", "界！"]


def test_a_unit_missing_from_the_transcript_keeps_its_bare_text():
    items = [_Item("alpha", 0.0, 0.5), _Item("zzz", 0.6, 0.8), _Item("beta", 0.9, 1.2)]
    words = words_from_alignment("alpha beta.", items, offset=0.0)
    assert [w.word for w in words] == ["alpha", "zzz", " beta."]


def test_segments_break_at_sentence_ends_pauses_and_the_length_cap():
    words = _words(
        ("Hi.", 0.0, 0.3),
        (" Mr.", 0.4, 0.6),  # an abbreviation, not a sentence end
        (" Smith", 0.6, 1.0),
        (" waits", 2.0, 2.4),  # 1.0 s pause before it
        (" and", 2.5, 6.0),
        (" waits", 6.1, 9.5),  # would stretch the segment past 7 s
    )
    segments = segments_from_words(words)
    assert [(s.index, s.text) for s in segments] == [
        (1, "Hi."),
        (2, "Mr. Smith"),
        (3, "waits and"),
        (4, "waits"),
    ]
    assert (segments[2].start, segments[2].end) == (2.0, 6.0)
    assert all(s.has_word_timestamps() for s in segments)


@pytest.fixture()
def wav(tmp_path):
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(np.zeros(16000 * 4, dtype=np.int16).tobytes())
    return path


class _FakeModel:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def transcribe(self, audio, language=None, return_time_stamps=False):
        self.calls.append({"chunks": len(audio), "language": language})
        return self.results


def _backend(monkeypatch, results, offsets=(0.0, 2.0)):
    backend = Qwen3AsrBackend.from_config(AppConfig().qwen3_asr)
    model = _FakeModel(results)
    monkeypatch.setattr(backend, "_load", lambda: model)
    monkeypatch.setattr(
        qwen3_asr, "_split_into_chunks", lambda samples: [(samples, o) for o in offsets]
    )
    return backend, model


def test_chunks_are_offset_numbered_on_and_reported_in_our_language_code(monkeypatch, wav):
    results = [
        _Result("Chinese", "你好。", _Alignment([_Item("你", 0.1, 0.3), _Item("好", 0.3, 0.6)])),
        _Result("Chinese", "再见", _Alignment([_Item("再", 0.2, 0.4), _Item("见", 0.4, 0.7)])),
    ]
    backend, _ = _backend(monkeypatch, results)

    segments, info = backend.transcribe(wav)

    assert [(s.index, s.text, s.start, s.end) for s in segments] == [
        (1, "你好。", 0.1, 0.6),
        (2, "再见", 2.2, 2.7),
    ]
    assert (info.language, info.language_probability, info.duration) == ("zh", None, 4.0)


def test_a_forced_language_is_passed_by_name_and_reported_as_given(monkeypatch, wav):
    backend, model = _backend(monkeypatch, [_Result("Chinese", "", None)], offsets=(0.0,))

    segments, info = backend.transcribe(wav, language="zh-TW")

    assert model.calls == [{"chunks": 1, "language": "Chinese"}]
    assert segments == []
    assert info.language == "zh-TW"


def test_silence_reports_an_undetermined_language(monkeypatch, wav):
    backend, _ = _backend(monkeypatch, [_Result("", "", None)], offsets=(0.0,))
    assert backend.transcribe(wav)[1].language == "und"


def test_a_language_the_aligner_does_not_cover_is_warned_about(monkeypatch, wav, caplog):
    results = [_Result("Thai", "สวัสดี", _Alignment([_Item("สวัสดี", 0.0, 0.5)]))]
    backend, _ = _backend(monkeypatch, results, offsets=(0.0,))

    with caplog.at_level(logging.WARNING):
        _, info = backend.transcribe(wav)

    assert info.language == "th"
    assert "not trained on Thai" in caplog.text
    assert [d.code for d in info.degradations] == ["alignment_unsupported_language"]


def _fake_runtime(monkeypatch, *, cuda: bool, flash_attn: bool, model=None, hub=None):
    """Stand-ins for torch, qwen_asr and the HF download; returns the from_pretrained calls."""
    calls = []
    model = model or object()
    torch = types.SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
    )
    qwen_asr = types.SimpleNamespace(
        Qwen3ASRModel=types.SimpleNamespace(
            from_pretrained=lambda path, **kwargs: calls.append((path, kwargs)) or model
        )
    )
    hub = hub or types.SimpleNamespace(snapshot_download=lambda repo, **kwargs: f"/cache/{repo}")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", qwen_asr)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    real_find_spec = qwen3_asr.importlib.util.find_spec
    monkeypatch.setattr(
        qwen3_asr.importlib.util,
        "find_spec",
        lambda name: (
            (object() if flash_attn else None) if name == "flash_attn" else real_find_spec(name)
        ),
    )
    return calls


def test_without_flash_attention_it_warns_and_still_loads(monkeypatch, caplog):
    calls = _fake_runtime(monkeypatch, cuda=True, flash_attn=False)

    with caplog.at_level(logging.WARNING):
        Qwen3AsrBackend.from_config(AppConfig().qwen3_asr)._load()

    assert FLASH_ATTENTION_WARNING in caplog.text
    ((path, kwargs),) = calls
    assert path == "/cache/Qwen/Qwen3-ASR-1.7B"
    assert kwargs["forced_aligner"] == "/cache/Qwen/Qwen3-ForcedAligner-0.6B"
    assert "attn_implementation" not in kwargs
    assert (kwargs["dtype"], kwargs["device_map"]) == ("bf16", "cuda:0")


def test_with_flash_attention_both_models_use_it(monkeypatch, caplog):
    calls = _fake_runtime(monkeypatch, cuda=True, flash_attn=True)

    with caplog.at_level(logging.WARNING):
        Qwen3AsrBackend.from_config(AppConfig().qwen3_asr)._load()

    assert FLASH_ATTENTION_WARNING not in caplog.text
    ((_, kwargs),) = calls
    assert kwargs["attn_implementation"] == "flash_attention_2"
    assert kwargs["forced_aligner_kwargs"]["attn_implementation"] == "flash_attention_2"


def test_without_cuda_it_falls_back_to_the_cpu_and_warns(monkeypatch, caplog):
    calls = _fake_runtime(monkeypatch, cuda=False, flash_attn=True)

    with caplog.at_level(logging.WARNING):
        Qwen3AsrBackend.from_config(AppConfig().qwen3_asr)._load()

    assert FLASH_ATTENTION_WARNING in caplog.text
    ((_, kwargs),) = calls
    assert (kwargs["dtype"], kwargs["device_map"]) == ("fp32", "cpu")


class _Hub:
    """A local cache under root holding the repos given complete; going online records the repo."""

    def __init__(self, root, complete):
        self.root, self.online = root, []
        for repo in complete:
            (root / repo).mkdir(parents=True)
            (root / repo / "model.safetensors").write_bytes(b"")

    def snapshot_download(self, repo, local_files_only=False, **kwargs):
        if local_files_only:
            return str(self.root / repo)
        self.online.append(repo)
        return f"/downloaded/{repo}"


def test_cached_models_load_without_asking_the_hub(monkeypatch, tmp_path):
    config = AppConfig().qwen3_asr
    hub = _Hub(tmp_path, [config.model, config.aligner_model])
    calls = _fake_runtime(monkeypatch, cuda=True, flash_attn=True, hub=hub)

    Qwen3AsrBackend.from_config(config)._load()

    ((path, kwargs),) = calls
    assert (path, kwargs["forced_aligner"]) == (
        str(tmp_path / config.model),
        str(tmp_path / config.aligner_model),
    )
    assert hub.online == []


def test_a_model_missing_its_weights_is_fetched_on_load(monkeypatch, tmp_path):
    config = AppConfig().qwen3_asr
    hub = _Hub(tmp_path, [config.aligner_model])
    calls = _fake_runtime(monkeypatch, cuda=True, flash_attn=True, hub=hub)

    Qwen3AsrBackend.from_config(config)._load()

    ((path, _),) = calls
    assert path == f"/downloaded/{config.model}"
    assert hub.online == [config.model]


def test_the_factory_builds_it_from_the_qwen3_asr_section():
    config = AppConfig()
    config.asr.backend = "qwen3_asr"
    config.qwen3_asr.model = "Qwen/Qwen3-ASR-0.6B"
    config.qwen3_asr.batch_size = 2

    backend = create_backend(config)

    assert isinstance(backend, Qwen3AsrBackend)
    assert (backend.model_name, backend.batch_size) == ("Qwen/Qwen3-ASR-0.6B", 2)
    assert backend.get_model_size() == 1_880_000_000 + 1_840_000_000


def test_a_segment_over_the_cap_is_cut_after_its_last_clause_mark():
    words = _words(
        ("He", 0.0, 0.5),
        (" waited,", 0.6, 2.0),
        (" and", 2.1, 4.0),
        (" waited", 4.1, 6.5),
        (" more", 6.6, 7.5),  # pushes the segment past 7 s
    )
    assert [s.text for s in segments_from_words(words)] == ["He waited,", "and waited more"]


def test_load_fallbacks_are_reported_with_every_transcription(monkeypatch, wav):
    results = [_Result("English", "Hi.", _Alignment([_Item("Hi", 0.1, 0.4)]))]
    _fake_runtime(monkeypatch, cuda=False, flash_attn=False, model=_FakeModel(results))
    monkeypatch.setattr(qwen3_asr, "_split_into_chunks", lambda samples: [(samples, 0.0)])
    backend = Qwen3AsrBackend.from_config(AppConfig().qwen3_asr)

    first = backend.transcribe(wav)[1].degradations
    second = backend.transcribe(wav)[1].degradations

    assert [d.code for d in first] == ["cpu_fallback", "flash_attention_missing"]
    assert second == first


def test_long_audio_is_recognised_in_chunks_of_about_fifteen_seconds():
    # Without FlashAttention 2 a long chunk loses whole passages; the splitter may move a cut 5 s.
    pytest.importorskip("qwen_asr")
    wav = np.random.default_rng(0).standard_normal(16000 * 120).astype(np.float32) * 0.1

    chunks = qwen3_asr._split_into_chunks(wav)

    assert len(chunks) >= 6
    assert max(len(chunk) for chunk, _ in chunks) <= 16000 * 20
