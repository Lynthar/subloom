"""The quickstart wizard: which backends and settings it saves, and what it reports as unresolved.
GPU, ffmpeg and the ASR backend are fakes; translation services are fake HTTP servers."""

import httpx
import pytest
import yaml
from typer.testing import CliRunner

pytest.importorskip("faster_whisper", reason="core.transcriber imports faster_whisper")

from subloom.cli.app import app  # noqa: E402
from subloom.core import asr as asr_module  # noqa: E402
from subloom.core.model_manager import OllamaModelManager  # noqa: E402
from subloom.core.transcriber import Transcriber  # noqa: E402
from subloom.models.config import (  # noqa: E402
    DEEPL_API_KEY_ENV,
    GOOGLE_API_KEY_ENV,
    OPENAI_API_KEY_ENV,
)
from subloom.utils import setup_wizard  # noqa: E402

runner = CliRunner()

NO_GPU = {"cuda_available": False}
GPU_24GB = {
    "cuda_available": True,
    "device_name": "Fake GPU",
    "total_vram_mb": 24000,
    "available_vram_mb": 20000,
}


class _CachedBackend:
    model_name = "fake-model"

    def is_model_cached(self):
        return True


@pytest.fixture
def machine(monkeypatch):
    """A machine with ffmpeg and no API keys; ``machine(gpu, ...)`` sets the rest."""
    for env in (OPENAI_API_KEY_ENV, DEEPL_API_KEY_ENV, GOOGLE_API_KEY_ENV):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(setup_wizard, "check_ffmpeg", lambda: True)
    monkeypatch.setattr(asr_module, "create_backend", lambda config, **o: _CachedBackend())

    def configure(gpu, qwen_asr=True, whisperx=True, optimal="large-v3"):
        monkeypatch.setattr(setup_wizard, "check_gpu", lambda: gpu)
        monkeypatch.setattr(setup_wizard, "qwen3_asr_installed", lambda: qwen_asr)
        monkeypatch.setattr(setup_wizard, "whisperx_installed", lambda: whisperx)
        monkeypatch.setattr(Transcriber, "select_optimal_model", classmethod(lambda cls: optimal))

    return configure


def _wizard(tmp_path, *answers):
    path = tmp_path / "config.yaml"
    result = runner.invoke(
        app, ["--config", str(path), "quickstart"], input="".join(f"{a}\n" for a in answers)
    )
    saved = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    return result, saved


def test_a_cpu_machine_gets_cpu_settings_for_qwen3_and_a_key_hint_for_deepl(tmp_path, machine):
    machine(NO_GPU, qwen_asr=False)

    result, saved = _wizard(tmp_path, "2", "deepl")

    assert result.exit_code == 0, result.output
    assert saved["asr"]["backend"] == "qwen3_asr"
    assert saved["qwen3_asr"]["device"] == "cpu"
    assert saved["translation"]["backend"] == "deepl"
    assert saved["deepl"]["api_key"] is None
    assert DEEPL_API_KEY_ENV in result.output
    assert "qwen3-asr extra" in result.output
    assert "Setup Complete" not in result.output


def test_whisper_on_cpu_saves_the_recommendations_it_offers(tmp_path, machine, monkeypatch):
    machine(NO_GPU)
    monkeypatch.setattr(OllamaModelManager, "check_connection", lambda self: False)

    result, saved = _wizard(tmp_path, "nonsense", "whisper", "y", "ollama", "y")

    assert result.exit_code == 0, result.output
    assert "Choose a number" in result.output
    whisper = saved["whisper"]
    assert (whisper["device"], whisper["compute_type"], whisper["model"]) == (
        "cpu",
        "int8",
        "small",
    )
    assert saved["translation"]["backend"] == "ollama"
    assert saved["ollama"]["model"] == "qwen2.5:7b"
    assert "ollama serve" in result.output


def test_openai_takes_the_model_from_the_servers_list(tmp_path, machine, serve):
    machine(GPU_24GB)
    server = serve(
        lambda request, body: httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})
    )

    result, saved = _wizard(tmp_path, "1", "openai", "http://localhost:1234/v1", "2")

    assert result.exit_code == 0, result.output
    assert saved["translation"]["backend"] == "openai"
    assert saved["openai"] == {
        "base_url": "http://localhost:1234/v1",
        "model": "b",
        "api_key": None,
    }
    assert "Setup Complete" in result.output
    assert all("authorization" not in request.headers for request, _ in server.requests)


def test_openai_without_a_model_list_asks_for_an_id_and_reports_the_refusal(
    tmp_path, machine, serve
):
    machine(GPU_24GB)
    serve(lambda request, body: httpx.Response(401, json={"error": "no key"}))

    result, saved = _wizard(tmp_path, "1", "openai", "https://api.example.com/v1", "my-model")

    assert result.exit_code == 0, result.output
    assert saved["openai"]["model"] == "my-model"
    assert "REFUSED" in result.output
    assert "Setup Complete" not in result.output


def test_whisper_without_whisperx_says_how_to_install_it(tmp_path, machine, serve):
    machine(GPU_24GB, whisperx=False)
    serve(lambda request, body: httpx.Response(200, json={"data": [{"id": "a"}]}))

    result, _ = _wizard(tmp_path, "whisper", "openai", "http://localhost:1234/v1", "1")

    assert result.exit_code == 0, result.output
    assert "pip install -e '.[whisperx]'" in result.output
    assert "whisperx extra" in result.output
    assert "Setup Complete" not in result.output


def test_whisperx_turned_off_is_not_reported_missing(tmp_path, machine, serve):
    machine(GPU_24GB, whisperx=False)
    serve(lambda request, body: httpx.Response(200, json={"data": [{"id": "a"}]}))
    (tmp_path / "config.yaml").write_text("whisper:\n  use_whisperx: false\n", encoding="utf-8")

    result, _ = _wizard(tmp_path, "whisper", "openai", "http://localhost:1234/v1", "1")

    assert result.exit_code == 0, result.output
    assert "whisperx" not in result.output
    assert "Setup Complete" in result.output
