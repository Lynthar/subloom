"""Which prompt the CLI reports, exports and reverts to, and which one the translator runs.

Covers `translation.prompt_template_id` and `config set-prompt`; no server needed."""

import pytest
from typer.testing import CliRunner

from subtitle_forge.cli.app import app
from subtitle_forge.core import prompt_library
from subtitle_forge.core.prompt_library import CUSTOM_TEMPLATE_ID, PromptLibrary, get_prompt_library
from subtitle_forge.core.translator import SubtitleTranslator
from subtitle_forge.models.config import AppConfig, TranslationConfig
from subtitle_forge.models.subtitle import SubtitleSegment

runner = CliRunner()

LIBRARY_ID = "movie-scifi"


def _run(cfg_path, *args):
    return runner.invoke(app, ["--config", str(cfg_path), *args])


def _select_library_template(cfg_path):
    result = _run(cfg_path, "config", "use-prompt", LIBRARY_ID)
    assert result.exit_code == 0, result.output
    assert AppConfig.load(cfg_path).translation.prompt_template_id == LIBRARY_ID


def test_reset_prompt_clears_a_library_selection(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    _select_library_template(cfg_path)

    result = _run(cfg_path, "config", "reset-prompt")

    assert result.exit_code == 0, result.output
    assert "Already using default" not in result.output
    assert AppConfig.load(cfg_path).translation.prompt_template_id is None


def test_reset_prompt_reports_no_change_when_already_default(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    AppConfig().save(cfg_path)

    result = _run(cfg_path, "config", "reset-prompt")

    assert result.exit_code == 0, result.output
    assert "Already using default" in result.output


def test_show_prompt_shows_the_selected_library_template(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    _select_library_template(cfg_path)

    result = _run(cfg_path, "config", "show-prompt")

    assert result.exit_code == 0, result.output
    assert "Using default prompt" not in result.output
    assert LIBRARY_ID in result.output


def test_export_prompt_writes_the_selected_library_template(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    _select_library_template(cfg_path)
    out = tmp_path / "prompt.txt"

    result = _run(cfg_path, "config", "export-prompt", "-o", str(out))

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == get_prompt_library().get_template(LIBRARY_ID).template


def test_export_prompt_writes_the_default_when_nothing_is_selected(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    AppConfig().save(cfg_path)
    out = tmp_path / "prompt.txt"

    result = _run(cfg_path, "config", "export-prompt", "-o", str(out))

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == SubtitleTranslator.DEFAULT_PROMPT_TEMPLATE


def test_process_rejects_an_unknown_prompt_template_id(tmp_path, monkeypatch):
    pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")
    from subtitle_forge.core.transcriber import Transcriber

    def _refuse(*args, **kwargs):
        raise AssertionError("process started work despite an unknown prompt id")

    # Building a Transcriber is the first thing past argument handling, and on
    # a cold cache it downloads gigabytes — guard it so a regression fails here
    # instead of hanging.
    monkeypatch.setattr(Transcriber, "__init__", _refuse)

    cfg_path = tmp_path / "config.yaml"
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")

    result = _run(
        cfg_path, "process", str(video), "-t", "zh", "--prompt-template", "no-such-template"
    )

    assert result.exit_code == 1
    assert "no-such-template" in result.output
    assert "not found" in result.output


class _RecordingChat:
    model = "qwen2.5:14b"

    def __init__(self, reply):
        self.reply = reply
        self.json_keys = []

    def chat(self, prompt, *, json_keys=None, max_tokens=None):
        self.json_keys.append(json_keys)
        return self.reply


@pytest.mark.parametrize("config", [{}, {"prompt_template_id": "no-such-template"}])
def test_the_default_prompt_runs_in_json_mode_however_it_was_reached(config):
    # A dangling id falls back to the default body, so the reply must parse and clean the same way.
    chat = _RecordingChat('{"translations": {"1": "[3] 天后"}}')
    translator = SubtitleTranslator(TranslationConfig(**config), chat)
    segment = SubtitleSegment(index=1, start=0.0, end=1.0, text="[3] days later")

    out = translator.translate_batch([segment], "en", "zh")

    assert out[0].text == "[3] 天后"
    assert chat.json_keys == [["1"]]


@pytest.fixture
def user_templates(tmp_path, monkeypatch):
    monkeypatch.setattr(PromptLibrary, "USER_TEMPLATES_DIR", tmp_path / "prompts")
    monkeypatch.setattr(prompt_library, "_library", None)
    return tmp_path / "prompts"


def test_set_prompt_stores_the_text_as_the_custom_template_and_selects_it(tmp_path, user_templates):
    cfg_path = tmp_path / "config.yaml"
    body = "Translate {source_lang} to {target_lang}:\n{segments}"
    prompt_file = tmp_path / "mine.txt"
    for text in ("first {source_lang} {target_lang} {segments}", body):  # the second replaces
        prompt_file.write_text(text, encoding="utf-8")
        result = _run(cfg_path, "config", "set-prompt", "-f", str(prompt_file))
        assert result.exit_code == 0, result.output

    assert AppConfig.load(cfg_path).translation.prompt_template_id == CUSTOM_TEMPLATE_ID
    assert get_prompt_library().get_template(CUSTOM_TEMPLATE_ID).template == body
    assert (user_templates / f"{CUSTOM_TEMPLATE_ID}.json").exists()
