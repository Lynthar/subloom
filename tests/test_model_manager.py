"""Tests for Ollama model availability matching.

Locks in the fix for the substring false-positive: requesting "qwen2.5:32b"
with only "qwen2.5:7b" installed used to report "available" (because "qwen2.5"
is a substring of "qwen2.5:7b"), so the download was skipped and translation
died with model-not-found only after transcription had already run.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from typer.testing import CliRunner

from subloom.core.model_manager import OllamaModelManager


def _manager(installed):
    m = OllamaModelManager()
    m.list_models = lambda: list(installed)  # type: ignore[assignment]
    return m


def test_different_tag_is_not_a_false_positive():
    assert _manager(["qwen2.5:7b"]).is_model_available("qwen2.5:32b") is False


def test_exact_tag_matches():
    assert _manager(["qwen2.5:7b", "llama3:latest"]).is_model_available("qwen2.5:7b") is True


def test_bare_name_matches_latest_tag():
    assert _manager(["qwen2.5:latest"]).is_model_available("qwen2.5") is True


def test_bare_name_does_not_match_specific_tag():
    assert _manager(["qwen2.5:7b"]).is_model_available("qwen2.5") is False


def test_no_cross_model_substring_match():
    # Old behaviour: "llama3" in "tinyllama:latest" -> True. Must be False now.
    assert _manager(["tinyllama:latest"]).is_model_available("llama3") is False


def test_absent_model_is_unavailable():
    assert _manager(["qwen2.5:7b"]).is_model_available("gemma2:9b") is False


class _DownClient:
    """Stands in for ollama.Client when the daemon cannot be reached."""

    def list(self):
        raise ConnectionError("connection refused")


def _manager_with_down_daemon():
    m = OllamaModelManager()
    m._client = _DownClient()  # type: ignore[assignment]
    return m


# "Cannot ask Ollama" used to come back as [] and then as False, so every caller read an
# unreachable daemon as a missing model and offered a download that could not work.
def test_listing_failure_is_raised_not_an_empty_list():
    with pytest.raises(ConnectionError):
        _manager_with_down_daemon().list_models()


def test_listing_failure_does_not_read_as_model_missing():
    with pytest.raises(ConnectionError):
        _manager_with_down_daemon().is_model_available("qwen2.5:7b")


def test_ensure_model_does_not_start_a_download_it_cannot_finish():
    with pytest.raises(ConnectionError):
        _manager_with_down_daemon().ensure_model("qwen2.5:7b")


class _OllamaStub(BaseHTTPRequestHandler):
    """Loopback Ollama: /api/tags lists `installed`, /api/pull streams `pull_events` with 200."""

    def log_message(self, *args):
        pass

    def _reply(self, content_type, body):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        models = [{"model": name} for name in self.server.installed]
        self._reply("application/json", json.dumps({"models": models}).encode())

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.server.install_on_pull:
            self.server.installed.append(PULLED)
        lines = b"".join(json.dumps(e).encode() + b"\n" for e in self.server.pull_events)
        self._reply("application/x-ndjson", lines)


PULLED = "qwen2.5:14b"


@pytest.fixture
def ollama_stub():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OllamaStub)
    server.installed = []
    server.install_on_pull = False
    server.pull_events = [{"status": "pulling manifest"}]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def _host(server):
    return f"http://127.0.0.1:{server.server_port}"


@pytest.mark.parametrize("events", [[], [{"status": "pulling manifest"}], [{"status": "success"}]])
def test_pull_that_ends_without_the_model_is_a_failure(ollama_stub, events):
    ollama_stub.pull_events = events
    with pytest.raises(RuntimeError, match="not installed"):
        list(OllamaModelManager(host=_host(ollama_stub)).pull_model(PULLED))


def test_pull_that_installs_the_model_succeeds_without_a_success_event(ollama_stub):
    ollama_stub.install_on_pull = True
    updates = list(OllamaModelManager(host=_host(ollama_stub)).pull_model(PULLED))
    assert [u.status for u in updates] == ["pulling manifest"]


def test_process_stops_before_transcribing_when_the_pull_leaves_no_model(
    ollama_stub, tmp_path, monkeypatch
):
    pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")
    from subloom.cli.app import app
    from subloom.core import asr as asr_module
    from subloom.core import pipeline as pipeline_module

    class _CachedTranscriber:
        def is_model_cached(self):
            return True

    reached = []
    monkeypatch.setattr(asr_module, "create_backend", lambda config, **o: _CachedTranscriber())
    monkeypatch.setattr(pipeline_module, "run_pipeline", lambda *a, **k: reached.append(a))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"ollama:\n  host: {_host(ollama_stub)}\n  model: {PULLED}\n", encoding="utf-8")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")

    result = CliRunner().invoke(
        app, ["--config", str(cfg), "process", str(video), "-t", "zh"], input="y\n"
    )

    assert result.exit_code == 1, result.output
    assert "Model downloaded successfully" not in result.output
    assert reached == []
