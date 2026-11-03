"""Contract of `core.pipeline.run_pipeline` with fake transcriber, translator and extractor.

Subtitle files are real; skips when ffmpeg-python is absent (core.audio imports it)."""

from pathlib import Path

import pytest

pytest.importorskip("ffmpeg", reason="core.audio imports ffmpeg-python")

from subtitle_forge.core import pipeline as pipeline_module  # noqa: E402
from subtitle_forge.core.asr import TranscriptionInfo  # noqa: E402
from subtitle_forge.core.pipeline import (  # noqa: E402
    PipelineHooks,
    PipelineOutput,
    PipelineResult,
    StructureCheckError,
    run_pipeline,
)
from subtitle_forge.core.subtitle import SubtitleProcessor  # noqa: E402
from subtitle_forge.models.config import AppConfig  # noqa: E402
from subtitle_forge.models.degradation import Degradation  # noqa: E402
from subtitle_forge.models.subtitle import SubtitleSegment  # noqa: E402
from subtitle_forge.server import processing  # noqa: E402
from subtitle_forge.server.jobs import Job  # noqa: E402


class _FakeTranscriber:
    """Records what it was asked to do and returns fixed segments (two clean ones by default)."""

    def __init__(self, language="en", segments=None, duration=60.0):
        self.info = TranscriptionInfo(
            language=language, language_probability=0.98, duration=duration
        )
        self.segments = segments
        self.calls = []
        self.raise_on_transcribe = None

    def transcribe(self, audio_path, **kwargs):
        self.calls.append({"audio_path": Path(audio_path), **kwargs})
        if self.raise_on_transcribe is not None:
            raise self.raise_on_transcribe
        segments = self.segments or [
            SubtitleSegment(index=1, start=0.0, end=1.5, text="Hello there"),
            SubtitleSegment(index=2, start=2.0, end=3.5, text="General Kenobi"),
        ]
        return segments, self.info


class _FakeTranslator:
    """Prefixes each line with the target language; records every call."""

    def __init__(self, drop_last=False, untranslated=None):
        self.calls = []
        self.drop_last = drop_last
        self.untranslated = untranslated or {}  # target language -> its Degradation

    def untranslated_degradation(self, target_lang, total):
        return self.untranslated.get(target_lang)

    def translate(self, segments, source_lang, target_lang, progress_callback=None):
        self.calls.append((source_lang, target_lang))
        if progress_callback is not None:
            progress_callback(len(segments), len(segments))
        translated = [
            SubtitleSegment(
                index=s.index, start=s.start, end=s.end, text=f"[{target_lang}] {s.text}"
            )
            for s in segments
        ]
        return translated[:-1] if self.drop_last else translated


@pytest.fixture()
def fake_audio(tmp_path, monkeypatch):
    """Replaces AudioExtractor so extract() just drops a scratch file."""
    created = []

    class _FakeExtractor:
        def extract(self, video_path, output_path=None):
            scratch = tmp_path / f"{Path(video_path).stem}.scratch.wav"
            scratch.write_bytes(b"RIFF")
            created.append(scratch)
            return scratch

    monkeypatch.setattr(pipeline_module, "AudioExtractor", _FakeExtractor)
    return created


@pytest.fixture()
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not really a video")
    return path


def _read(path):
    return path.read_text(encoding="utf-8")


def test_saves_original_and_one_translation(tmp_path, video, fake_audio):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    translator = _FakeTranslator()

    result = run_pipeline(
        video,
        AppConfig(),
        transcriber=_FakeTranscriber(),
        translator=translator,
        target_languages=["zh"],
        output_dir=out_dir,
    )

    assert result.detected_language == "en"
    assert result.segment_count == 2
    assert [(o.language, o.path.name) for o in result.outputs] == [
        ("en", "clip.en.srt"),
        ("zh", "clip.zh.srt"),
    ]
    assert "Hello there" in _read(out_dir / "clip.en.srt")
    assert "[zh] Hello there" in _read(out_dir / "clip.zh.srt")
    assert translator.calls == [("en", "zh")]
    # The scratch file is the pipeline's own to clean up.
    assert fake_audio and not fake_audio[0].exists()


def test_bilingual_merges_into_one_file(tmp_path, video, fake_audio):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = run_pipeline(
        video,
        AppConfig(),
        transcriber=_FakeTranscriber(),
        translator=_FakeTranslator(),
        target_languages=["zh"],
        output_dir=out_dir,
        keep_original=False,
        bilingual=True,
    )

    assert [(o.language, o.path.name) for o in result.outputs] == [("en-zh", "clip.en-zh.srt")]
    merged = _read(out_dir / "clip.en-zh.srt")
    assert "Hello there" in merged and "[zh] Hello there" in merged
    assert not (out_dir / "clip.en.srt").exists()


def test_target_equal_to_source_is_skipped(tmp_path, video, fake_audio):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    translator = _FakeTranslator()
    skipped = []

    result = run_pipeline(
        video,
        AppConfig(),
        transcriber=_FakeTranscriber(language="en"),
        translator=translator,
        target_languages=["en", "zh"],
        output_dir=out_dir,
        hooks=PipelineHooks(on_translation_skipped=skipped.append),
    )

    assert skipped == ["en"]
    assert translator.calls == [("en", "zh")]
    assert [o.language for o in result.outputs] == ["en", "zh"]


def test_transcribe_only_needs_no_translator(tmp_path, video, fake_audio):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = run_pipeline(
        video,
        AppConfig(),
        transcriber=_FakeTranscriber(),
        target_languages=[],
        output_dir=out_dir,
    )

    assert [o.path.name for o in result.outputs] == ["clip.en.srt"]
    assert not fake_audio[0].exists()


def test_translation_targets_without_a_translator_fail_before_any_work(tmp_path, video, fake_audio):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    transcriber = _FakeTranscriber()

    with pytest.raises(ValueError, match="translator"):
        run_pipeline(
            video,
            AppConfig(),
            transcriber=transcriber,
            target_languages=["zh"],
            output_dir=out_dir,
        )

    assert transcriber.calls == []
    assert fake_audio == []


def test_original_output_path_pins_the_single_output(tmp_path, video, fake_audio):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pinned = out_dir / "chosen.name.srt"

    result = run_pipeline(
        video,
        AppConfig(),
        transcriber=_FakeTranscriber(),
        target_languages=[],
        output_dir=out_dir,
        original_output_path=pinned,
    )

    assert [o.path for o in result.outputs] == [pinned]
    assert pinned.exists()
    assert not (out_dir / "clip.en.srt").exists()


def test_scratch_audio_is_removed_when_transcription_fails(tmp_path, video, fake_audio):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    transcriber = _FakeTranscriber()
    transcriber.raise_on_transcribe = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(
            video,
            AppConfig(),
            transcriber=transcriber,
            translator=_FakeTranslator(),
            target_languages=["zh"],
            output_dir=out_dir,
        )

    assert fake_audio and not fake_audio[0].exists()


def test_invalid_language_code_is_rejected_before_extraction(tmp_path, video, fake_audio):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with pytest.raises(ValueError):
        run_pipeline(
            video,
            AppConfig(),
            transcriber=_FakeTranscriber(),
            translator=_FakeTranslator(),
            target_languages=["../evil"],
            output_dir=out_dir,
        )

    assert fake_audio == []


def test_source_language_reaches_the_backend(tmp_path, video, fake_audio):
    transcriber = _FakeTranscriber()

    run_pipeline(
        video,
        AppConfig(),
        transcriber=transcriber,
        target_languages=[],
        output_dir=tmp_path,
        source_language="ja",
    )

    assert transcriber.calls[0]["language"] == "ja"


@pytest.mark.parametrize("post_process, processed", [(True, True), (False, False)])
def test_the_pipeline_post_processes_raw_backend_timing(
    tmp_path, video, fake_audio, post_process, processed
):
    # The backend hands back raw timing; the default 80 ms lead-in moves the 2.0 s cue earlier.
    run_pipeline(
        video,
        AppConfig(),
        transcriber=_FakeTranscriber(),
        target_languages=[],
        output_dir=tmp_path,
        post_process=post_process,
    )

    second_start = SubtitleProcessor().load(tmp_path / "clip.en.srt")[1].start
    assert (second_start < 1.95) is processed


@pytest.fixture()
def unprocessed(monkeypatch):
    """Lets backend timing through post-processing untouched, so the structure check sees it."""

    class _PassThrough:
        def __init__(self, config, language=None):
            pass

        def process(self, segments, audio_duration=None):
            return segments

    monkeypatch.setattr(pipeline_module, "TimestampProcessor", _PassThrough)


_OVERLAPPING = [
    SubtitleSegment(index=1, start=0.0, end=2.0, text="Hello there"),
    SubtitleSegment(index=2, start=1.5, end=3.5, text="General Kenobi"),
]


def test_structure_violation_writes_every_file_then_raises(
    tmp_path, video, fake_audio, unprocessed
):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with pytest.raises(StructureCheckError) as caught:
        run_pipeline(
            video,
            AppConfig(),
            transcriber=_FakeTranscriber(segments=_OVERLAPPING),
            translator=_FakeTranslator(),
            target_languages=["zh"],
            output_dir=out_dir,
        )

    written = [out_dir / "clip.en.srt", out_dir / "clip.zh.srt"]
    assert all(path.exists() for path in written)
    assert [o.path for o in caught.value.outputs] == written
    assert caught.value.failures == {path: ["#2: overlaps previous by 500 ms"] for path in written}
    assert "#2: overlaps previous by 500 ms" in str(caught.value)


def test_output_past_the_audio_end_fails_the_check(tmp_path, video, fake_audio, unprocessed):
    with pytest.raises(StructureCheckError, match="ends past the audio"):
        run_pipeline(
            video,
            AppConfig(),
            transcriber=_FakeTranscriber(duration=3.0),
            target_languages=[],
            output_dir=tmp_path,
        )


@pytest.mark.parametrize("overrides", [{"timestamp_mode": "off"}, {"post_process": False}])
def test_raw_timing_is_not_checked(tmp_path, video, fake_audio, overrides):
    result = run_pipeline(
        video,
        AppConfig(),
        transcriber=_FakeTranscriber(segments=_OVERLAPPING),
        target_languages=[],
        output_dir=tmp_path,
        **overrides,
    )

    assert [o.path.name for o in result.outputs] == ["clip.en.srt"]


def test_bilingual_merge_does_not_hide_a_lost_translation(tmp_path, video, fake_audio):
    # The merge falls back to the original text for a missing line, so the merged file looks
    # complete; the check has to look at the translation itself.
    with pytest.raises(StructureCheckError, match="1 segments out for 2 in"):
        run_pipeline(
            video,
            AppConfig(),
            transcriber=_FakeTranscriber(),
            translator=_FakeTranslator(drop_last=True),
            target_languages=["zh"],
            output_dir=tmp_path,
            keep_original=False,
            bilingual=True,
        )


def test_server_job_lists_the_written_files_when_the_check_fails(tmp_path, monkeypatch):
    output = PipelineOutput(language="en", path=tmp_path / "clip.en.srt")

    def failing_pipeline(*args, **kwargs):
        raise StructureCheckError({output.path: ["#2: overlaps previous by 500 ms"]}, [output])

    monkeypatch.setattr(processing, "run_pipeline", failing_pipeline)
    monkeypatch.setattr(processing, "create_translator", lambda *a, **k: None)
    job = Job(job_id="j", video_path=str(tmp_path / "clip.mp4"), target_languages=["zh"])

    with pytest.raises(StructureCheckError):
        processing._run_job(job, AppConfig(), transcriber=None)

    assert job.outputs == [{"language": "en", "path": str(output.path)}]


def test_the_result_carries_backend_then_per_language_degradations(tmp_path, video, fake_audio):
    cpu = Degradation("cpu_fallback", "CUDA is unavailable")
    ja = Degradation("translation_incomplete", "ja: 1 of 2 segments left in the source language")
    transcriber = _FakeTranscriber()
    transcriber.info.degradations = [cpu]

    result = run_pipeline(
        video,
        AppConfig(),
        transcriber=transcriber,
        translator=_FakeTranslator(untranslated={"ja": ja}),
        target_languages=["zh", "ja"],
        output_dir=tmp_path,
    )

    assert result.degradations == [cpu, ja]


def test_a_clean_run_reports_no_degradations(tmp_path, video, fake_audio):
    result = run_pipeline(
        video,
        AppConfig(),
        transcriber=_FakeTranscriber(),
        translator=_FakeTranslator(),
        target_languages=["zh"],
        output_dir=tmp_path,
    )

    assert result.degradations == []


def test_server_job_records_the_degradations(tmp_path, monkeypatch):
    cpu = Degradation("cpu_fallback", "CUDA is unavailable")
    result = PipelineResult("en", 0.9, 1, degradations=[cpu])
    monkeypatch.setattr(processing, "run_pipeline", lambda *a, **k: result)
    monkeypatch.setattr(processing, "create_translator", lambda *a, **k: None)
    job = Job(job_id="j", video_path=str(tmp_path / "clip.mp4"), target_languages=["zh"])

    processing._run_job(job, AppConfig(), transcriber=None)

    assert job.degradations == [cpu]


def _logged(events):
    """A transcriber and a translator that log unloads and translations into one list, in order."""
    transcriber, translator = _FakeTranscriber(), _FakeTranslator()
    transcriber.unload_model = lambda: events.append("unload")
    translate = translator.translate
    translator.translate = lambda *a, **k: events.append("translate") or translate(*a, **k)
    return transcriber, translator


def test_frees_the_gpu_before_an_llm_translates(tmp_path, video, fake_audio):
    events = []
    transcriber, translator = _logged(events)

    run_pipeline(
        video,
        AppConfig(),
        transcriber=transcriber,
        translator=translator,
        target_languages=["zh", "ja"],
        output_dir=tmp_path,
        free_gpu_for_translation=True,
    )

    assert events == ["unload", "translate", "translate"]


def _machine_translation():
    config = AppConfig()
    config.translation.backend = "deepl"
    return config


@pytest.mark.parametrize(
    ("config", "targets", "free_gpu"),
    [
        (AppConfig(), ["zh"], False),
        (_machine_translation(), ["zh"], True),
        (AppConfig(), ["en"], True),
    ],
    ids=["not asked", "machine translation", "only the source language"],
)
def test_keeps_the_asr_model_unless_asked_and_an_llm_will_translate(
    tmp_path, video, fake_audio, config, targets, free_gpu
):
    events = []
    transcriber, translator = _logged(events)

    run_pipeline(
        video,
        config,
        transcriber=transcriber,
        translator=translator,
        target_languages=targets,
        output_dir=tmp_path,
        free_gpu_for_translation=free_gpu,
    )

    assert "unload" not in events
