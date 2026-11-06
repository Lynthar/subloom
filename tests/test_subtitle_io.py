"""SubtitleProcessor I/O and language-normalization tests.

Locks in: atomic save (no truncated .srt after a crash-window, no temp
leftovers), UTF-16 BOM detection (used to be silently mojibake'd by the
ISO-8859-1 fallback), the audible warning on that fallback, and target
language de-duplication.
"""

import logging

import pytest

from subloom.core.subtitle import (
    SubtitleProcessor,
    normalize_target_languages,
)
from subloom.models.subtitle import SubtitleSegment

_SRT_TEXT = "1\n00:00:00,000 --> 00:00:01,000\nHello there\n\n"


def _segments():
    return [SubtitleSegment(index=1, start=0.0, end=1.0, text="Hello there")]


def test_save_round_trips_and_leaves_no_temp_files(tmp_path):
    out = tmp_path / "video.zh.srt"
    processor = SubtitleProcessor()
    processor.save(_segments(), out)
    assert processor.load(out)[0].text == "Hello there"
    leftovers = [p for p in tmp_path.iterdir() if p != out]
    assert leftovers == []


def test_save_rounds_times_to_the_nearest_millisecond(tmp_path):
    out = tmp_path / "video.en.srt"
    SubtitleProcessor().save(
        [SubtitleSegment(index=1, start=1.92, end=59.9996, text="Hello there")], out
    )
    assert "00:00:01,920 --> 00:01:00,000" in out.read_text(encoding="utf-8")


def test_save_replaces_existing_file(tmp_path):
    out = tmp_path / "video.zh.srt"
    processor = SubtitleProcessor()
    processor.save(_segments(), out)
    processor.save([SubtitleSegment(index=1, start=0.0, end=1.0, text="Second write")], out)
    assert processor.load(out)[0].text == "Second write"


def test_load_utf16_file_via_bom(tmp_path):
    path = tmp_path / "u16.srt"
    path.write_bytes("1\n00:00:00,000 --> 00:00:01,000\n你好世界\n\n".encode("utf-16"))
    segments = SubtitleProcessor().load(path)
    assert segments[0].text == "你好世界"


def test_latin1_fallback_warns(tmp_path, caplog):
    path = tmp_path / "legacy.srt"
    # 0xE9 followed by newline is invalid UTF-8 and an invalid GBK pair, so
    # only the ISO-8859-1 last resort can decode it.
    path.write_bytes(b"1\n00:00:00,000 --> 00:00:01,000\ncaf\xe9\n\n")
    with caplog.at_level(logging.WARNING, logger="subloom.core.subtitle"):
        segments = SubtitleProcessor().load(path)
    assert segments[0].text == "caf\xe9"
    assert any("ISO-8859-1" in r.message for r in caplog.records)


def test_normalize_target_languages_dedupes_preserving_order():
    assert normalize_target_languages(["zh", "ja", "zh", "ja", "ko"]) == ["zh", "ja", "ko"]


def test_normalize_target_languages_rejects_path_metacharacters():
    with pytest.raises(ValueError):
        normalize_target_languages(["zh", "../../evil"])
