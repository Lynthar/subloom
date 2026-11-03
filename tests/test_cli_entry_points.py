"""The files each CLI entry point writes; Whisper, ffmpeg and Ollama are fakes, SRTs are real.

Skips when ffmpeg-python or faster-whisper is absent, since the patched modules import them."""

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from typer.testing import CliRunner

pytest.importorskip("ffmpeg", reason="core.audio imports ffmpeg-python")
pytest.importorskip("faster_whisper", reason="core.transcriber imports faster_whisper")

from subtitle_forge.cli.app import app  # noqa: E402
from subtitle_forge.core import asr as asr_module  # noqa: E402
from subtitle_forge.core import pipeline as pipeline_module  # noqa: E402
from subtitle_forge.core import translator as translator_module  # noqa: E402
from subtitle_forge.core.asr import TranscriptionInfo  # noqa: E402
from subtitle_forge.core.asr import create_backend as real_create_backend  # noqa: E402
from subtitle_forge.core.transcriber import Transcriber  # noqa: E402
from subtitle_forge.models.degradation import Degradation  # noqa: E402
from subtitle_forge.models.subtitle import SubtitleSegment  # noqa: E402

runner = CliRunner()

# What the fake transcriber was asked to do, reset per test.
transcribe_calls = []
# What the fakes report as degraded, reset per test.
backend_degradations = []
untranslated = {}  # target language -> its Degradation
# Model unloads and translations, in the order they happened, reset per test.
events = []
# The lines the fake transcriber hears, reset per test.
spoken = []

CPU = Degradation("cpu_fallback", "CUDA is unavailable; ran on the CPU")


class _FakeTranscriber:
    def is_model_cached(self):
        return True

    def unload_model(self):
        events.append("unload")

    def transcribe(self, audio_path, **kwargs):
        transcribe_calls.append(kwargs)
        segments = [
            SubtitleSegment(index=i, start=2.0 * i, end=2.0 * i + 1.5, text=text)
            for i, text in enumerate(spoken, 1)
        ]
        return segments, TranscriptionInfo("en", 0.99, 60.0, list(backend_degradations))


class _FakeTranslator:
    LANGUAGE_NAMES: ClassVar[dict] = {}
    backend = SimpleNamespace(label="Fake", endpoint="fake://", model="fake")

    def check_connection(self):
        return True

    def check_model_available(self):
        return True

    def effective_batch_size(self):
        return 10

    def untranslated_degradation(self, target_lang, total):
        return untranslated.get(target_lang)

    def translate(self, segments, source_lang, target_lang, progress_callback=None):
        events.append("translate")
        return [
            SubtitleSegment(
                index=s.index, start=s.start, end=s.end, text=f"[{target_lang}] {s.text}"
            )
            for s in segments
        ]


class _FakeExtractor:
    def extract(self, video_path, output_path=None):
        scratch = Path(video_path).with_suffix(".scratch.wav")
        scratch.write_bytes(b"RIFF")
        return scratch


def _fake_components(monkeypatch):
    transcribe_calls.clear()
    backend_degradations.clear()
    untranslated.clear()
    events.clear()
    spoken[:] = ["Hello there"]
    monkeypatch.setattr(asr_module, "create_backend", lambda config, **o: _FakeTranscriber())
    monkeypatch.setattr(
        translator_module, "create_translator", lambda config, **kw: _FakeTranslator()
    )
    monkeypatch.setattr(pipeline_module, "AudioExtractor", _FakeExtractor)


def _video(tmp_path, name="clip.mp4"):
    path = tmp_path / name
    path.write_bytes(b"not really a video")
    return path


def _run(tmp_path, *args):
    return runner.invoke(app, ["--config", str(tmp_path / "config.yaml"), *args])


def _configure_bilingual_only(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "output:\n  keep_original: false\n  bilingual: true\n", encoding="utf-8"
    )


def test_transcribe_writes_the_output_path_it_was_given(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    video = _video(tmp_path)
    named = tmp_path / "elsewhere" / "chosen.srt"

    result = _run(tmp_path, "transcribe", str(video), "-o", str(named))

    assert result.exit_code == 0, result.output
    assert "Hello there" in named.read_text(encoding="utf-8")
    assert not (tmp_path / "clip.en.srt").exists()


def test_transcribe_defaults_to_the_detected_language_beside_the_video(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    video = _video(tmp_path)

    result = _run(tmp_path, "transcribe", str(video))

    assert result.exit_code == 0, result.output
    assert "Hello there" in (tmp_path / "clip.en.srt").read_text(encoding="utf-8")
    assert not video.with_suffix(".scratch.wav").exists()


def test_transcribe_passes_its_timing_flags_down(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    video = _video(tmp_path)
    used = []

    class _RecordingProcessor:
        def __init__(self, config, language=None):
            used.append(config)

        def process(self, segments, audio_duration=None):
            return segments

    monkeypatch.setattr(pipeline_module, "TimestampProcessor", _RecordingProcessor)

    result = _run(
        tmp_path, "transcribe", str(video), "--timestamp-mode", "full", "--no-split-sentences"
    )

    assert result.exit_code == 0, result.output
    assert [(c.mode, c.split_sentences) for c in used] == [("full", False)]


def test_transcribe_follows_the_configured_vad_filter_unless_a_flag_says_otherwise(
    tmp_path, monkeypatch
):
    _fake_components(monkeypatch)
    built = []
    monkeypatch.setattr(
        asr_module,
        "create_backend",
        lambda config, **o: (
            built.append(real_create_backend(config, **o).vad_filter) or _FakeTranscriber()
        ),
    )
    (tmp_path / "config.yaml").write_text("whisper:\n  vad_filter: false\n", encoding="utf-8")
    video = str(_video(tmp_path))

    plain = _run(tmp_path, "transcribe", video)
    flagged = _run(tmp_path, "transcribe", video, "--vad-filter")

    assert plain.exit_code == 0, plain.output
    assert flagged.exit_code == 0, flagged.output
    assert built == [False, True]


def test_batch_writes_original_and_translation_for_every_video(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    videos = tmp_path / "videos"
    videos.mkdir()
    _video(videos, "a.mp4")
    _video(videos, "b.mp4")

    result = _run(tmp_path, "batch", str(videos), "-t", "zh")

    assert result.exit_code == 0, result.output
    for stem in ("a", "b"):
        assert "Hello there" in (videos / f"{stem}.en.srt").read_text(encoding="utf-8")
        assert "[zh] Hello there" in (videos / f"{stem}.zh.srt").read_text(encoding="utf-8")


def test_batch_honours_bilingual(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    videos = tmp_path / "videos"
    videos.mkdir()
    _video(videos, "a.mp4")

    result = _run(tmp_path, "batch", str(videos), "-t", "zh", "--bilingual")

    assert result.exit_code == 0, result.output
    merged = (videos / "a.en-zh.srt").read_text(encoding="utf-8")
    assert "Hello there" in merged and "[zh] Hello there" in merged
    assert not (videos / "a.zh.srt").exists()


def test_process_falls_back_to_config_output_flags(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    _configure_bilingual_only(tmp_path)
    video = _video(tmp_path)

    result = _run(tmp_path, "process", str(video), "-t", "zh")

    assert result.exit_code == 0, result.output
    assert "[zh] Hello there" in (tmp_path / "clip.en-zh.srt").read_text(encoding="utf-8")
    assert not (tmp_path / "clip.en.srt").exists()


def test_batch_falls_back_to_config_output_flags(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    _configure_bilingual_only(tmp_path)
    videos = tmp_path / "videos"
    videos.mkdir()
    _video(videos, "a.mp4")

    result = _run(tmp_path, "batch", str(videos), "-t", "zh")

    assert result.exit_code == 0, result.output
    assert "[zh] Hello there" in (videos / "a.en-zh.srt").read_text(encoding="utf-8")
    assert not (videos / "a.en.srt").exists()


def test_translate_falls_back_to_config_bilingual(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    _configure_bilingual_only(tmp_path)
    srt = tmp_path / "clip.en.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,500\nHello there\n", encoding="utf-8")

    result = _run(tmp_path, "translate", str(srt), "-t", "zh")

    assert result.exit_code == 0, result.output
    merged = (tmp_path / "clip.en-zh.srt").read_text(encoding="utf-8")
    assert "Hello there" in merged and "[zh] Hello there" in merged

    result = _run(tmp_path, "translate", str(srt), "-t", "zh", "--no-bilingual")

    assert result.exit_code == 0, result.output
    assert "[zh] Hello there" in (tmp_path / "clip.zh.srt").read_text(encoding="utf-8")


def test_batch_rejects_an_invalid_timestamp_mode_before_transcribing(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    videos = tmp_path / "videos"
    videos.mkdir()
    _video(videos, "a.mp4")

    result = _run(tmp_path, "batch", str(videos), "-t", "zh", "--timestamp-mode", "minimal-ish")

    assert result.exit_code == 1
    assert transcribe_calls == []


def test_batch_debug_log_creates_a_new_output_dir(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    video = _video(tmp_path)
    out = tmp_path / "new" / "subs"

    result = _run(tmp_path, "batch", str(video), "-t", "zh", "-o", str(out), "--save-debug-log")

    assert result.exit_code == 0, result.output
    assert "[zh] Hello there" in (out / "clip.zh.srt").read_text(encoding="utf-8")
    assert (out / "clip_debug" / "run.log").exists()


def test_process_debug_log_creates_a_new_output_dir(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    video = _video(tmp_path)
    out = tmp_path / "new" / "subs"

    result = _run(tmp_path, "process", str(video), "-t", "zh", "-o", str(out), "--save-debug-log")

    assert result.exit_code == 0, result.output
    assert "[zh] Hello there" in (out / "clip.zh.srt").read_text(encoding="utf-8")
    assert (out / "clip_debug" / "run.log").exists()


def test_whisper_only_flags_are_reported_and_dropped_under_another_backend(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    built = []
    monkeypatch.setattr(
        asr_module, "create_backend", lambda config, **o: built.append(o) or _FakeTranscriber()
    )
    (tmp_path / "config.yaml").write_text("asr:\n  backend: qwen3_asr\n", encoding="utf-8")

    result = _run(
        tmp_path,
        "transcribe",
        str(_video(tmp_path)),
        "--whisperx",
        "--model",
        "Qwen/Qwen3-ASR-0.6B",
    )

    assert result.exit_code == 0, result.output
    assert "Whisper-only" in result.output
    assert built == [{"model_name": "Qwen/Qwen3-ASR-0.6B"}]


def test_asr_backend_flag_selects_the_backend_and_drops_whisper_flags(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    built = []
    monkeypatch.setattr(
        asr_module,
        "create_backend",
        lambda config, **o: built.append((config.asr.backend, o)) or _FakeTranscriber(),
    )

    transcribed = _run(tmp_path, "transcribe", str(_video(tmp_path)), "--asr-backend", "qwen3_asr")
    batched = _run(
        tmp_path, "batch", str(tmp_path), "-t", "zh", "--asr-backend", "qwen3_asr",
        "--whisper-model", "small",
    )  # fmt: skip

    assert transcribed.exit_code == 0, transcribed.output
    assert batched.exit_code == 0, batched.output
    assert "Whisper-only" in batched.output
    assert [backend for backend, _ in built] == ["qwen3_asr", "qwen3_asr"]


def test_an_unknown_asr_backend_exits_1_before_any_work(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    built = []
    monkeypatch.setattr(asr_module, "create_backend", lambda config, **o: built.append(o))

    result = _run(tmp_path, "process", str(_video(tmp_path)), "-t", "zh", "--asr-backend", "x")

    assert result.exit_code == 1, result.output
    assert "asr.backend must be one of" in result.output
    assert built == []


def test_a_degraded_transcription_says_so_and_exits_3(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    backend_degradations.append(CPU)
    video = _video(tmp_path)

    result = _run(tmp_path, "transcribe", str(video))

    assert result.exit_code == 3, result.output
    assert "ran on the CPU" in result.output
    assert "Hello there" in (tmp_path / "clip.en.srt").read_text(encoding="utf-8")


def test_auto_model_without_readable_vram_falls_back_to_small_and_says_so(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    built = []
    monkeypatch.setattr(
        asr_module, "create_backend", lambda config, **o: built.append(o) or _FakeTranscriber()
    )
    monkeypatch.setattr(Transcriber, "select_optimal_model", classmethod(lambda cls: None))

    result = _run(tmp_path, "transcribe", str(_video(tmp_path)), "--auto-model")

    assert result.exit_code == 3, result.output
    assert built[0]["model_name"] == "small"
    assert "fell back to small" in result.output


def test_process_reports_an_incomplete_translation_and_exits_3(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    untranslated["zh"] = Degradation("translation_incomplete", "zh: 1 of 1 segments untranslated")

    result = _run(tmp_path, "process", str(_video(tmp_path)), "-t", "zh")

    assert result.exit_code == 3, result.output
    assert "zh: 1 of 1 segments untranslated" in result.output


def test_translate_reports_an_incomplete_translation_and_exits_3(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    untranslated["zh"] = Degradation("translation_incomplete", "zh: 1 of 1 segments untranslated")
    srt = tmp_path / "clip.en.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,500\nHello there\n", encoding="utf-8")

    result = _run(tmp_path, "translate", str(srt), "-t", "zh")

    assert result.exit_code == 3, result.output
    assert "zh: 1 of 1 segments untranslated" in result.output


def test_batch_lists_a_degradation_every_task_shares_once_and_exits_3(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    backend_degradations.append(CPU)
    videos = tmp_path / "videos"
    videos.mkdir()
    _video(videos, "a.mp4")
    _video(videos, "b.mp4")

    result = _run(tmp_path, "batch", str(videos), "-t", "zh")

    assert result.exit_code == 3, result.output
    assert "Degraded in every task" in result.output
    assert result.output.count("ran on the CPU") == 1
    assert "2 of 2 task(s) completed with degradations" in result.output


def test_process_unloads_the_asr_model_before_translating(tmp_path, monkeypatch):
    _fake_components(monkeypatch)

    result = _run(tmp_path, "process", str(_video(tmp_path)), "-t", "zh")

    assert result.exit_code == 0, result.output
    assert events[:2] == ["unload", "translate"]


def test_batch_unloads_before_each_translation_only_with_one_worker(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    videos = tmp_path / "videos"
    videos.mkdir()
    _video(videos, "a.mp4")
    _video(videos, "b.mp4")

    alone = _run(tmp_path, "batch", str(videos), "-t", "zh", "--workers", "1")
    assert alone.exit_code == 0, alone.output
    assert events == ["unload", "translate", "unload", "translate", "unload"]

    events.clear()
    shared = _run(tmp_path, "batch", str(videos), "-t", "zh", "--workers", "2")
    assert shared.exit_code == 0, shared.output
    assert events == ["translate", "translate", "unload"]  # only the cleanup at the end


@pytest.mark.parametrize(
    ("command", "written"),
    [
        (["process", "{video}", "-t", "zh"], ["clip.en.srt", "clip.zh.srt"]),
        (["transcribe", "{video}"], ["clip.en.srt"]),
        (["batch", "{dir}", "-t", "zh"], ["clip.en.srt", "clip.zh.srt"]),
    ],
    ids=["process", "transcribe", "batch"],
)
def test_drop_interjections_leaves_them_out_of_every_file(tmp_path, monkeypatch, command, written):
    _fake_components(monkeypatch)
    spoken[:] = ["Uh, um.", "Hello there"]
    videos = tmp_path / "videos"
    videos.mkdir()
    video = _video(videos)
    args = [a.format(video=video, dir=videos) for a in command]

    result = _run(tmp_path, *args, "--drop-interjections")

    assert result.exit_code == 0, result.output
    for name in written:
        text = (videos / name).read_text(encoding="utf-8")
        assert "Hello there" in text and "Uh" not in text


def test_keep_interjections_overrides_the_config(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    spoken[:] = ["Uh, um.", "Hello there"]
    (tmp_path / "config.yaml").write_text("output:\n  drop_interjections: true\n", encoding="utf-8")
    video = _video(tmp_path)

    configured = _run(tmp_path, "transcribe", str(video), "-o", str(tmp_path / "configured.srt"))
    kept = _run(
        tmp_path, "transcribe", str(video), "-o", str(tmp_path / "kept.srt"), "--keep-interjections"
    )

    assert configured.exit_code == 0, configured.output
    assert kept.exit_code == 0, kept.output
    assert "Uh" not in (tmp_path / "configured.srt").read_text(encoding="utf-8")
    assert "Uh" in (tmp_path / "kept.srt").read_text(encoding="utf-8")


def test_translate_drops_interjections_before_translating(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    srt = tmp_path / "clip.en.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nUh, um.\n\n"
        "2\n00:00:02,000 --> 00:00:03,500\nHello there\n",
        encoding="utf-8",
    )

    result = _run(tmp_path, "translate", str(srt), "-t", "zh", "--drop-interjections")

    assert result.exit_code == 0, result.output
    translated = (tmp_path / "clip.zh.srt").read_text(encoding="utf-8")
    assert translated.startswith("1\n") and "[zh] Hello there" in translated
    assert "Uh" not in translated
