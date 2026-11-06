"""Tests for config save/load round-tripping.

Locks in the fix for the data-loss bug where save() used a hand-maintained
whitelist and silently dropped fields (ollama.request_timeout / max_retries,
whisper.batch_size / download_root) on rewrite.
"""

from subtitle_forge.models.config import AppConfig


def test_save_load_preserves_all_fields(tmp_path):
    path = tmp_path / "config.yaml"
    cfg = AppConfig()
    cfg.translation.request_timeout = 600.0
    cfg.translation.max_retries = 5
    cfg.whisper.batch_size = 8
    cfg.whisper.download_root = "/models"
    cfg.timestamp.linger_ms = 250
    cfg.save(path)

    loaded = AppConfig.load(path)
    assert loaded.translation.request_timeout == 600.0
    assert loaded.translation.max_retries == 5
    assert loaded.whisper.batch_size == 8
    assert loaded.whisper.download_root == "/models"
    assert loaded.timestamp.linger_ms == 250


def test_hand_edited_field_survives_a_config_set_rewrite(tmp_path):
    # Simulate: user writes request_timeout by hand, then a later save() (as
    # `config set` performs) must not erase it.
    path = tmp_path / "config.yaml"
    cfg = AppConfig()
    cfg.translation.request_timeout = 600.0
    cfg.save(path)

    reopened = AppConfig.load(path)
    reopened.whisper.model = "large-v2"  # unrelated change
    reopened.save(path)

    assert AppConfig.load(path).translation.request_timeout == 600.0


def test_defaults_round_trip(tmp_path):
    path = tmp_path / "config.yaml"
    AppConfig().save(path)
    loaded = AppConfig.load(path)
    assert loaded.whisper.model == AppConfig().whisper.model
    assert loaded.timestamp.mode == "minimal"


def test_default_yaml_matches_code_defaults():
    # config/default.yaml is the documented reference copy of the defaults —
    # it is NOT loaded at runtime — so this test is what keeps it from
    # silently drifting away from the dataclass defaults it documents.
    import dataclasses
    from pathlib import Path

    import yaml

    repo_root = Path(__file__).resolve().parent.parent
    with open(repo_root / "config" / "default.yaml", encoding="utf-8") as f:
        documented = yaml.safe_load(f)
    code_defaults = dataclasses.asdict(AppConfig())

    def check(doc: dict, code: dict, prefix: str = "") -> None:
        undocumented = sorted(set(code) - set(doc))
        assert not undocumented, f"not in default.yaml: {prefix}{undocumented}"
        for key, value in doc.items():
            assert key in code, f"{prefix}{key} is documented but not an AppConfig field"
            if isinstance(value, dict):
                check(value, code[key], prefix=f"{prefix}{key}.")
            else:
                assert code[key] == value, (
                    f"{prefix}{key}: default.yaml documents {value!r}, "
                    f"code default is {code[key]!r}"
                )

    check(documented, code_defaults)


def test_config_path_uses_appdata_on_windows(monkeypatch):
    import sys
    from pathlib import Path

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(Path("/fake/appdata")))
    assert AppConfig.get_config_path() == Path("/fake/appdata") / "subtitle-forge" / "config.yaml"


def test_validate_rejects_zero_chars_per_second():
    # chars_per_second=0 used to save fine and only blow up later as a
    # ZeroDivisionError in the middle of timestamp post-processing.
    cfg = AppConfig()
    cfg.timestamp.chars_per_second = 0
    import pytest

    with pytest.raises(ValueError, match="chars_per_second"):
        cfg.validate()


def test_validate_reports_every_problem_at_once():
    import pytest

    cfg = AppConfig()
    cfg.timestamp.chars_per_second = 0
    cfg.timestamp.mode = "minmal"  # typo
    cfg.translation.max_retries = 0  # range(0) would silently skip translation
    cfg.timestamp.min_duration = 5.0
    cfg.timestamp.max_duration = 1.0
    with pytest.raises(ValueError) as excinfo:
        cfg.validate()
    message = str(excinfo.value)
    for fragment in ("chars_per_second", "timestamp.mode", "max_retries", "max_duration"):
        assert fragment in message, fragment


def test_validate_rejects_names_where_they_would_be_ignored():
    import pytest

    machine = AppConfig()
    machine.translation.backend, machine.translation.names = "deepl", ["光希"]
    loose = AppConfig()
    loose.translation.names = "光希"  # a bare string from a hand-edited file
    for cfg, fragment in ((machine, "needs an LLM backend"), (loose, "list of names")):
        with pytest.raises(ValueError, match=fragment):
            cfg.validate()


def test_load_rejects_invalid_config_file(tmp_path):
    import pytest

    path = tmp_path / "config.yaml"
    path.write_text("timestamp:\n  chars_per_second: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chars_per_second"):
        AppConfig.load(path)


def test_defaults_are_valid():
    AppConfig().validate()


def test_saved_config_is_owner_readable_only_on_posix(tmp_path):
    import os
    import stat
    import sys

    import pytest

    if sys.platform == "win32":
        pytest.skip("POSIX permission bits are meaningless on Windows")
    path = tmp_path / "config.yaml"
    AppConfig().save(path)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, oct(mode)
