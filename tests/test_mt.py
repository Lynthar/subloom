"""The machine translation branch: what DeepL and Google are sent, how their answers become
segments, and which failures end the run. The fake server stands in for both services."""

import httpx
import pytest
from typer.testing import CliRunner

from subloom.cli.app import app
from subloom.core.mt import DeepLBackend, GoogleBackend, MachineTranslator, create_mt_backend
from subloom.core.service import ServiceError
from subloom.core.translator import SubtitleTranslator, create_translator
from subloom.exceptions import TranslationError
from subloom.models.config import (
    DEEPL_API_KEY_ENV,
    GOOGLE_API_KEY_ENV,
    AppConfig,
    TranslationConfig,
)
from subloom.models.subtitle import SubtitleSegment


@pytest.fixture(autouse=True)
def _no_keys_from_the_environment(monkeypatch):
    monkeypatch.delenv(DEEPL_API_KEY_ENV, raising=False)
    monkeypatch.delenv(GOOGLE_API_KEY_ENV, raising=False)
    monkeypatch.setattr("subloom.core.service.time.sleep", lambda s: None)


def _segs(*texts):
    return [
        SubtitleSegment(index=i + 1, start=float(i), end=float(i) + 0.9, text=t)
        for i, t in enumerate(texts)
    ]


def _deepl_echo(request, body):
    """Answers each text with its upper-cased self."""
    return httpx.Response(200, json={"translations": [{"text": t.upper()} for t in body["text"]]})


def _deepl(key="k", max_texts=None):
    backend = DeepLBackend(key, TranslationConfig())
    if max_texts:
        backend.max_texts = max_texts
    return MachineTranslator(TranslationConfig(max_retries=3), backend)


def test_a_free_key_goes_to_the_free_endpoint(serve):
    server = serve(_deepl_echo)

    _deepl("abc:fx").translate(_segs("hi"), "en", "zh")
    _deepl("abc").translate(_segs("hi"), "en", "zh")

    hosts = [request.url.host for request, _ in server.requests]
    assert hosts == ["api-free.deepl.com", "api.deepl.com"]
    assert server.requests[0][0].headers["authorization"] == "DeepL-Auth-Key abc:fx"


@pytest.mark.parametrize(
    ("source", "target", "sent"),
    [
        ("en", "zh", ("EN", "ZH-HANS")),
        ("ja", "zh-TW", ("JA", "ZH-HANT")),
        ("zh-TW", "en", ("ZH", "EN-US")),
    ],
)
def test_deepl_language_codes(serve, source, target, sent):
    server = serve(_deepl_echo)

    _deepl().translate(_segs("hi"), source, target)

    body = server.requests[0][1]
    assert (body["source_lang"], body["target_lang"]) == sent


def test_batches_go_as_text_arrays_with_the_nearby_dialogue_as_context(serve):
    server = serve(_deepl_echo)
    segments = _segs("a", "b", "c", "d", "e")

    out = _deepl(max_texts=2).translate(segments, "en", "zh", context_window=1)

    assert [s.text for s in out] == ["A", "B", "C", "D", "E"]
    assert [b["text"] for _, b in server.requests] == [["a", "b"], ["c", "d"], ["e"]]
    assert server.requests[1][1]["context"] == "b\nc\nd\ne"
    assert [(s.index, s.start, s.end) for s in out] == [(s.index, s.start, s.end) for s in segments]


def test_an_exhausted_quota_ends_the_run_without_retrying(serve):
    server = serve(lambda request, body: httpx.Response(456, json={"message": "Quota exceeded"}))

    with pytest.raises(TranslationError, match="quota"):
        _deepl().translate(_segs("hi"), "en", "zh")
    assert len(server.requests) == 1


def test_a_rate_limit_is_retried(serve):
    replies = iter([httpx.Response(429), None])

    def respond(request, body):
        reply = next(replies)
        return reply if reply is not None else _deepl_echo(request, body)

    server = serve(respond)

    out = _deepl().translate(_segs("hi"), "en", "zh")

    assert [s.text for s in out] == ["HI"]
    assert len(server.requests) == 2


def test_a_short_answer_ends_the_run(serve):
    serve(lambda request, body: httpx.Response(200, json={"translations": [{"text": "X"}]}))

    with pytest.raises(TranslationError, match="answered 1 texts for 2"):
        _deepl().translate(_segs("a", "b"), "en", "zh")


def test_an_empty_answer_keeps_the_source_and_is_reported(serve):
    serve(lambda r, b: httpx.Response(200, json={"translations": [{"text": "A"}, {"text": " "}]}))
    translator = _deepl()

    out = translator.translate(_segs("a", "b"), "en", "zh")

    assert [s.text for s in out] == ["A", "b"]
    degradation = translator.untranslated_degradation("zh", 2)
    assert degradation is not None and degradation.code == "translation_incomplete"


def test_a_complete_run_reports_nothing_untranslated(serve):
    serve(_deepl_echo)
    translator = _deepl()

    translator.translate(_segs("a"), "en", "zh")

    assert translator.untranslated_degradation("zh", 1) is None


def test_a_rejected_key_fails_the_check(serve):
    serve(lambda request, body: httpx.Response(403, json={"message": "Forbidden"}))

    with pytest.raises(ServiceError):
        _deepl().check_model_available()


def test_google_keeps_the_key_out_of_the_url_and_asks_for_plain_text(serve):
    def respond(request, body):
        texts = [{"translatedText": f"{t}!"} for t in body["q"]]
        return httpx.Response(200, json={"data": {"translations": texts}})

    server = serve(respond)
    translator = MachineTranslator(
        TranslationConfig(), GoogleBackend("secret", TranslationConfig())
    )

    out = translator.translate(_segs("don't"), "en", "zh")

    request, body = server.requests[0]
    assert str(request.url) == "https://translation.googleapis.com/language/translate/v2"
    assert request.headers["x-goog-api-key"] == "secret"
    assert (body["source"], body["target"], body["format"]) == ("en", "zh-CN", "text")
    assert "context" not in body
    assert [s.text for s in out] == ["don't!"]


def _config(backend, **keys):
    config = AppConfig()
    config.translation.backend = backend
    for section, key in keys.items():
        getattr(config, section).api_key = key
    return config


def test_the_environment_key_wins(monkeypatch):
    monkeypatch.setenv(DEEPL_API_KEY_ENV, "env:fx")

    backend = create_mt_backend(_config("deepl", deepl="file"))

    assert backend.endpoint == "https://api-free.deepl.com"


@pytest.mark.parametrize(
    ("backend", "env"), [("deepl", DEEPL_API_KEY_ENV), ("google", GOOGLE_API_KEY_ENV)]
)
def test_a_missing_key_is_named(backend, env):
    with pytest.raises(ValueError, match=env):
        create_mt_backend(_config(backend))


def test_each_backend_gets_its_own_branch():
    assert isinstance(create_translator(_config("google", google="k")), MachineTranslator)
    assert isinstance(create_translator(_config("ollama")), SubtitleTranslator)


def test_a_missing_key_stops_the_cli_before_any_work(tmp_path):
    srt = tmp_path / "clip.en.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(tmp_path / "c.yaml"),
            "translate",
            str(srt),
            "-t",
            "zh",
            "--translation-backend",
            "deepl",
        ],
    )

    assert result.exit_code == 1
    assert DEEPL_API_KEY_ENV in result.output
    assert not (tmp_path / "clip.zh.srt").exists()


def test_a_model_flag_is_ignored_by_a_backend_without_models(capsys):
    from subloom.cli.app import apply_translation_flags

    config = _config("deepl", deepl="k")

    apply_translation_flags(config, None, "some-model")

    assert not hasattr(config.deepl, "model")
    assert "ignored" in capsys.readouterr().out
