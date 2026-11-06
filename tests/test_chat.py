"""The chat backends: what goes over the wire, how a rejected JSON constraint is handled, and
which key is sent. httpx MockTransport stands in for the server."""

import logging
from types import SimpleNamespace

import httpx
import pytest
from ollama import ResponseError

from subtitle_forge.core.chat import OllamaChat, OpenAIChat, create_chat_backend
from subtitle_forge.core.service import ServiceError
from subtitle_forge.core.translator import SubtitleTranslator
from subtitle_forge.models.config import (
    OPENAI_API_KEY_ENV,
    AppConfig,
    OllamaConfig,
    OpenAIConfig,
    TranslationConfig,
)

REPLY = {"choices": [{"message": {"content": '{"translations": {"1": "你好"}}'}}]}


def _formats(server):
    return [(b.get("response_format") or {}).get("type") for _, b in server.requests]


def _openai(api_key=None, model="m"):
    return OpenAIChat(
        OpenAIConfig(base_url="http://local/v1", model=model), TranslationConfig(), api_key
    )


def _rejecting(*kinds):
    """A server that rejects response_format types in ``kinds``, as LM Studio and DeepSeek do."""

    def respond(request, body):
        kind = (body.get("response_format") or {}).get("type")
        if kind in kinds:
            return httpx.Response(400, json={"error": f"'response_format.type' {kind} unsupported"})
        return httpx.Response(200, json=REPLY)

    return respond


def test_the_json_constraint_lists_every_segment_as_required(serve):
    server = serve(lambda request, body: httpx.Response(200, json=REPLY))

    reply = _openai().chat("prompt", json_keys=["7", "8"])

    request, body = server.requests[0]
    assert request.url == "http://local/v1/chat/completions"
    translations = body["response_format"]["json_schema"]["schema"]["properties"]["translations"]
    assert translations["required"] == ["7", "8"]
    assert translations["additionalProperties"] is False
    assert reply == '{"translations": {"1": "你好"}}'


def test_a_rejected_schema_falls_back_to_json_object_for_the_rest_of_the_run(serve):
    server = serve(_rejecting("json_schema"))
    chat = _openai()

    chat.chat("p", json_keys=["1"])
    chat.chat("p", json_keys=["1"])

    assert _formats(server) == ["json_schema", "json_object", "json_object"]


def test_a_server_rejecting_both_gets_no_constraint(serve):
    server = serve(_rejecting("json_schema", "json_object"))

    _openai().chat("p", json_keys=["1"])

    assert _formats(server) == ["json_schema", "json_object", None]


def test_a_plain_text_request_carries_no_constraint(serve):
    server = serve(lambda request, body: httpx.Response(200, json=REPLY))

    _openai().chat("p")

    assert "response_format" not in server.requests[0][1]


def test_an_error_not_about_the_constraint_surfaces_and_keeps_it(serve):
    # OpenAI's reasoning models answer temperature=0 like this.
    refusal = httpx.Response(400, json={"error": "Unsupported value: 'temperature'"})
    replies = iter([refusal, httpx.Response(200, json=REPLY)])
    server = serve(lambda request, body: next(replies))
    chat = _openai()

    with pytest.raises(ServiceError) as excinfo:
        chat.chat("p", json_keys=["1"])
    chat.chat("p", json_keys=["1"])

    assert excinfo.value.status_code == 400
    assert _formats(server) == ["json_schema", "json_schema"]


def test_a_null_content_reads_as_an_empty_reply(serve):
    serve(lambda r, b: httpx.Response(200, json={"choices": [{"message": {"content": None}}]}))

    assert _openai().chat("p") == ""


def test_an_openai_compatible_reply_says_how_much_of_the_prompt_was_read(serve):
    reply = {"choices": [{"message": {"content": "x"}}]}
    replies = iter([{**reply, "usage": {"prompt_tokens": 7}}, reply])
    serve(lambda r, b: httpx.Response(200, json=next(replies)))
    chat = _openai()

    chat.chat("p")
    assert chat.last_prompt_tokens == 7

    chat.chat("p")
    assert chat.last_prompt_tokens is None


def test_model_listing(serve):
    serve(lambda request, body: httpx.Response(200, json={"data": [{"id": "a"}, {"id": "m"}]}))

    assert _openai(model="m").check_model_available()
    assert not _openai(model="x").check_model_available()


def test_a_refused_listing_raises_rather_than_reads_as_no_models(serve):
    serve(lambda request, body: httpx.Response(401, json={"error": "bad key"}))

    with pytest.raises(ServiceError):
        _openai().list_models()


def test_an_unreachable_server_fails_the_connection_check(serve):
    def refuse(request, body):
        raise httpx.ConnectError("refused")

    serve(refuse)

    assert not _openai().check_connection()


def _openai_config(api_key=None):
    config = AppConfig()
    config.translation.backend = "openai"
    config.openai = OpenAIConfig(base_url="http://local/v1", model="m", api_key=api_key)
    return config


def _authorization(serve, config):
    server = serve(lambda request, body: httpx.Response(200, json=REPLY))
    create_chat_backend(config).chat("p")
    return server.requests[0][0].headers.get("authorization")


def test_the_environment_key_wins_over_the_config_file(serve, monkeypatch):
    monkeypatch.setenv(OPENAI_API_KEY_ENV, "from-env")

    assert _authorization(serve, _openai_config("from-file")) == "Bearer from-env"


def test_the_config_file_key_is_used_without_the_environment(serve, monkeypatch):
    monkeypatch.delenv(OPENAI_API_KEY_ENV, raising=False)

    assert _authorization(serve, _openai_config("from-file")) == "Bearer from-file"


def test_no_key_sends_no_authorization(serve, monkeypatch):
    monkeypatch.delenv(OPENAI_API_KEY_ENV, raising=False)

    assert _authorization(serve, _openai_config()) is None


def test_only_the_configured_backend_is_built():
    config = _openai_config()
    assert isinstance(create_chat_backend(config), OpenAIChat)
    config.translation.backend = "ollama"
    assert isinstance(create_chat_backend(config), OllamaChat)


def test_an_ollama_error_keeps_its_status_for_the_retry_policy():
    chat = OllamaChat(OllamaConfig(), TranslationConfig())

    def refuse(**kwargs):
        raise ResponseError("model not found", 404)

    chat._client = SimpleNamespace(chat=refuse)  # type: ignore[assignment]

    with pytest.raises(ServiceError) as excinfo:
        chat.chat("p", json_keys=["1"])
    assert excinfo.value.status_code == 404


def test_ollama_caps_the_generation_it_is_asked_to():
    chat = OllamaChat(OllamaConfig(), TranslationConfig(temperature=0.0))
    sent = []

    def answer(**kwargs):
        sent.append(kwargs["options"])
        return {"message": {"content": "你好"}}

    chat._client = SimpleNamespace(chat=answer)  # type: ignore[assignment]

    chat.chat("p", max_tokens=512)
    chat.chat("p")

    assert sent == [{"temperature": 0.0, "num_predict": 512}, {"temperature": 0.0}]


def test_an_ollama_reply_says_how_much_of_the_prompt_was_read():
    chat = OllamaChat(OllamaConfig(), TranslationConfig())
    reply = {"message": {"content": "你好"}, "prompt_eval_count": 7}
    chat._client = SimpleNamespace(chat=lambda **kwargs: reply)  # type: ignore[assignment]

    chat.chat("p")

    assert chat.last_prompt_tokens == 7


def test_a_request_may_sample_at_its_own_temperature(serve):
    ollama = OllamaChat(OllamaConfig(), TranslationConfig(temperature=0.0))
    sent = []
    ollama._client = SimpleNamespace(  # type: ignore[assignment]
        chat=lambda **kwargs: (
            sent.append(kwargs["options"]["temperature"]) or {"message": {"content": "{}"}}
        )
    )
    server = serve(lambda request, body: httpx.Response(200, json=REPLY))

    ollama.chat("p", temperature=0.6)
    ollama.chat("p")
    _openai().chat("p", temperature=0.6)
    _openai().chat("p")

    assert sent == [0.6, 0.0]
    assert [b["temperature"] for _, b in server.requests] == [0.6, 0.0]


def test_a_reply_cut_at_the_cap_is_logged(caplog):
    # A thinking model can spend the whole cap thinking; the empty reply must say why.
    chat = OllamaChat(OllamaConfig(), TranslationConfig())
    cut = {"message": {"content": ""}, "done_reason": "length"}
    chat._client = SimpleNamespace(chat=lambda **kwargs: cut)  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING):
        assert chat.chat("p", max_tokens=512) == ""

    assert "512-token cap" in caplog.text


def test_an_openai_compatible_request_carries_no_length_cap(serve):
    # GPT-5 and the o-series reject max_tokens with a 400; that would end every run.
    server = serve(lambda request, body: httpx.Response(200, json=REPLY))

    _openai().chat("prompt", json_keys=["1"], max_tokens=512)

    _, body = server.requests[0]
    assert not {"max_tokens", "max_completion_tokens"} & set(body)


@pytest.mark.parametrize(
    ("backend", "model", "expected"),
    [
        ("ollama", "qwen2.5:7b", 4),
        ("ollama", "qwen2.5:14b", 6),
        ("ollama", "qwen2.5:32b", 10),
        ("ollama", "llama3", 4),
        ("openai", "Qwen2.5-14B-Instruct", 6),
        ("openai", "qwen2.5-72b-instruct", 10),
        ("openai", "gpt-4o-mini", 10),
    ],
)
def test_batch_size_is_clamped_by_the_size_in_the_model_name(backend, model, expected):
    settings = TranslationConfig(backend=backend, max_batch_size=10)
    translator = SubtitleTranslator(settings, SimpleNamespace(model=model))

    assert translator.effective_batch_size() == expected
