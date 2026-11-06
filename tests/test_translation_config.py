"""The translation section: migrating files written before it existed, what an openai backend
requires, and how the CLI overrides and displays it."""

import logging

import pytest
import yaml
from typer.testing import CliRunner

from subloom.cli.app import app, apply_translation_flags
from subloom.core import prompt_library
from subloom.core.prompt_library import CUSTOM_TEMPLATE_ID, PromptLibrary, get_prompt_library
from subloom.models import config as config_module
from subloom.models.config import AppConfig

runner = CliRunner()

BODY = "Translate {source_lang} to {target_lang}:\n{segments}"


@pytest.fixture
def user_templates(tmp_path, monkeypatch):
    monkeypatch.setattr(PromptLibrary, "USER_TEMPLATES_DIR", tmp_path / "prompts")
    monkeypatch.setattr(prompt_library, "_library", None)
    return tmp_path / "prompts"


def _write(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_an_old_file_loads_with_its_ollama_settings_moved_and_says_so(tmp_path, caplog):
    path = _write(
        tmp_path,
        {
            "ollama": {
                "model": "qwen2.5:7b",
                "host": "http://gpu:11434",
                "temperature": 0.2,
                "max_batch_size": 5,
                "request_timeout": 60.0,
                "prompt_template_id": "movie-scifi",
            }
        },
    )

    with caplog.at_level(logging.WARNING):
        config = AppConfig.load(path)

    assert (config.ollama.model, config.ollama.host) == ("qwen2.5:7b", "http://gpu:11434")
    tr = config.translation
    assert (tr.temperature, tr.max_batch_size, tr.request_timeout) == (0.2, 5, 60.0)
    assert tr.prompt_template_id == "movie-scifi"
    assert "ollama.temperature" in caplog.text
    assert "config import" in caplog.text


def test_a_config_subcommand_warns_about_an_old_file_once(tmp_path, monkeypatch):
    path = _write(tmp_path, {"ollama": {"temperature": 0.2}})
    warned = []
    monkeypatch.setattr(config_module.logger, "warning", lambda msg, *a, **k: warned.append(msg))

    result = runner.invoke(app, ["--config", str(path), "config", "show"])

    assert result.exit_code == 0, result.output
    assert len(warned) == 1


def test_a_key_already_under_translation_wins_over_the_old_one(tmp_path):
    path = _write(tmp_path, {"ollama": {"max_batch_size": 5}, "translation": {"max_batch_size": 8}})

    assert AppConfig.load(path).translation.max_batch_size == 8


def test_an_old_custom_prompt_becomes_the_custom_template_and_stays_selected(
    tmp_path, user_templates
):
    # The old body took precedence over the old library id.
    path = _write(
        tmp_path, {"ollama": {"prompt_template": BODY, "prompt_template_id": "movie-scifi"}}
    )

    config = AppConfig.load(path)

    assert config.translation.prompt_template_id == CUSTOM_TEMPLATE_ID
    assert get_prompt_library().get_template(CUSTOM_TEMPLATE_ID).template == BODY


def test_an_old_custom_prompt_never_replaces_a_different_custom_template(tmp_path, user_templates):
    prompt_library.store_custom_template(
        "Mine {source_lang} {target_lang} {segments}", replace=True
    )
    path = _write(tmp_path, {"ollama": {"prompt_template": BODY}})

    with pytest.raises(ValueError, match="save-prompt"):
        AppConfig.load(path)
    assert get_prompt_library().get_template(CUSTOM_TEMPLATE_ID).template.startswith("Mine")


def test_the_openai_backend_needs_a_url_and_a_model():
    config = AppConfig()
    config.translation.backend = "openai"

    with pytest.raises(ValueError) as excinfo:
        config.validate()

    assert "openai.base_url" in str(excinfo.value)
    assert "openai.model" in str(excinfo.value)


def test_an_unknown_backend_is_rejected():
    config = AppConfig()
    config.translation.backend = "bing"

    with pytest.raises(ValueError, match="translation.backend"):
        config.validate()


def test_the_model_flag_applies_to_the_selected_backend():
    config = AppConfig()
    config.openai.base_url = "http://localhost:1234/v1"

    apply_translation_flags(config, "openai", "qwen2.5-14b-instruct")

    assert config.translation.backend == "openai"
    assert config.openai.model == "qwen2.5-14b-instruct"
    assert config.ollama.model == AppConfig().ollama.model


def _run(cfg_path, *args):
    return runner.invoke(app, ["--config", str(cfg_path), *args])


def test_selecting_an_unconfigured_openai_backend_fails_before_any_work(tmp_path):
    srt = tmp_path / "clip.en.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

    result = _run(
        tmp_path / "config.yaml",
        "translate",
        str(srt),
        "-t",
        "zh",
        "--translation-backend",
        "openai",
    )

    assert result.exit_code == 1
    assert "openai.base_url" in result.output
    assert not (tmp_path / "clip.zh.srt").exists()


def test_config_show_never_prints_the_api_key(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    config = AppConfig()
    config.openai.api_key = "sk-secret-value"
    config.save(cfg_path)

    result = _run(cfg_path, "config", "show")

    assert result.exit_code == 0, result.output
    assert "sk-secret-value" not in result.output
    assert "hidden" in result.output


def test_pull_model_refuses_a_backend_that_is_not_ollama(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    config = AppConfig()
    config.translation.backend = "openai"
    config.openai.base_url = "http://localhost:1234/v1"
    config.openai.model = "m"
    config.save(cfg_path)

    result = _run(cfg_path, "config", "pull-model")

    assert result.exit_code == 1
    assert "translation.backend is openai" in result.output
