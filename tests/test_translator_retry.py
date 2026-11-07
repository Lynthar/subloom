"""The LLM translator's failure contract: which errors are retried, which end the run, which
send a batch one segment at a time, and what is counted as left untranslated. A fake chat
backend stands in for the server."""

import json
import logging

import httpx
import pytest

from subloom.core.service import ServiceError
from subloom.core.translator import SubtitleTranslator, reply_token_cap, squeeze_repeats
from subloom.models.config import TranslationConfig
from subloom.exceptions import TranslationError
from subloom.models.subtitle import SubtitleSegment


class _FakeChat:
    """Yields the scripted outcomes in order: Exception -> raise, str -> reply."""

    model = "qwen2.5:14b"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.prompts = []
        self.caps = []

    def chat(self, prompt, *, json_keys=None, max_tokens=None, think=True):
        self.calls += 1
        self.prompts.append(prompt)
        self.caps.append(max_tokens)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _translator(outcomes, **cfg_kwargs):
    """No name pass unless asked: its request would take the first scripted outcome."""
    fake = _FakeChat(outcomes)
    settings = TranslationConfig(**{"find_names": False, **cfg_kwargs})
    return SubtitleTranslator(settings, fake), fake


def _segs(*texts):
    return [
        SubtitleSegment(index=i + 1, start=float(i), end=float(i) + 0.9, text=t)
        for i, t in enumerate(texts)
    ]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("subloom.core.service.time.sleep", lambda s: None)


def _good(indices_to_text):
    return json.dumps({"translations": indices_to_text})


def test_read_error_is_retried():
    translator, fake = _translator([httpx.ReadError("connection reset"), _good({"1": "你好"})])
    out = translator.translate_batch(_segs("Hello"), "en", "zh")
    assert [s.text for s in out] == ["你好"]
    assert fake.calls == 2


def test_read_error_exhausting_retries_raises_translation_error():
    translator, fake = _translator([httpx.ReadError("x")] * 3, max_retries=3)
    with pytest.raises(TranslationError):
        translator.translate_batch(_segs("Hello"), "en", "zh")
    assert fake.calls == 3


def test_client_error_is_not_retried():
    translator, fake = _translator([ServiceError("model not found", 404)])
    with pytest.raises(TranslationError):
        translator.translate_batch(_segs("Hello"), "en", "zh")
    assert fake.calls == 1


def test_server_error_is_retried():
    translator, fake = _translator([ServiceError("boom", 500), _good({"1": "你好"})])
    out = translator.translate_batch(_segs("Hello"), "en", "zh")
    assert [s.text for s in out] == ["你好"]
    assert fake.calls == 2


def test_unrelated_exception_propagates_unwrapped():
    translator, fake = _translator([ValueError("programming error")])
    with pytest.raises(ValueError):
        translator.translate_batch(_segs("Hello"), "en", "zh")
    assert fake.calls == 1


def test_healed_segment_leaves_no_stale_failure_record():
    # Batch reply drops index 2; the individual retry then succeeds. The
    # failure record made during parsing must go with it.
    translator, fake = _translator([_good({"1": "你好"}), "再见"])
    out = translator.translate_batch(_segs("Hello", "Bye"), "en", "zh")
    assert [s.text for s in out] == ["你好", "再见"]
    assert fake.calls == 2
    assert translator.get_failed_translations() == []


def test_unhealed_segment_keeps_its_failure_record():
    # Batch reply drops index 2 and the individual retry fails too — that record must survive.
    translator, fake = _translator([_good({"1": "你好"}), ServiceError("boom", 500)])
    out = translator.translate_batch(_segs("Hello", "Bye"), "en", "zh")
    assert [s.text for s in out] == ["你好", "Bye"]
    assert [f["index"] for f in translator.get_failed_translations()] == [2]


def test_a_dropped_segment_answered_with_the_shared_kanji_is_translated():
    # ja -> zh: 腰 is 腰. The batch reply dropped it; the retry's identical answer is the translation.
    translator, _ = _translator([_good({"1": "你好"}), "腰"])
    translator.translate(_segs("こんにちは", "腰"), "ja", "zh")

    assert translator.untranslated_degradation("zh", 2) is None


def test_a_batch_lost_to_a_server_error_is_translated_one_by_one(caplog):
    # Ollama answers 500 when the reply repeats one token too often; the other lines must survive.
    outcomes = [ServiceError("token repeat limit reached", 500)] * 3
    outcomes += ["你好", ServiceError("token repeat limit reached", 500), "再见"]
    translator, fake = _translator(outcomes, max_retries=3)
    with caplog.at_level(logging.WARNING):
        out = translator.translate(_segs("Hello", "Ahhh", "Bye"), "en", "zh")

    assert [s.text for s in out] == ["你好", "Ahhh", "再见"]
    assert fake.calls == 6
    assert "Batch 1-3 failed" in caplog.text
    degradation = translator.untranslated_degradation("zh", 3)
    assert degradation is not None and degradation.message.startswith("zh: 1 of 3")


def test_a_batch_whose_reply_timed_out_is_translated_one_by_one():
    outcomes = [httpx.ReadTimeout("read timed out")] * 3 + ["你好", "再见"]
    translator, _ = _translator(outcomes, max_retries=3)
    out = translator.translate(_segs("Hello", "Bye"), "en", "zh")

    assert [s.text for s in out] == ["你好", "再见"]
    assert translator.untranslated_degradation("zh", 2) is None


def test_a_service_failing_every_request_still_ends_the_run():
    # Not one segment gets through one by one either: the next batch would fare no better.
    translator, fake = _translator([ServiceError("out of memory", 500)] * 5, max_retries=3)
    with pytest.raises(TranslationError):
        translator.translate(_segs("Hello", "Bye"), "en", "zh")
    assert fake.calls == 5


@pytest.mark.parametrize(
    ("error", "calls"),
    [(ServiceError("quota exceeded", 403), 1), (httpx.ConnectError("refused"), 3)],
    ids=["refused-at-once", "unreachable-after-retries"],
)
def test_a_service_level_failure_is_not_retried_one_by_one(error, calls):
    translator, fake = _translator([error] * 3, max_retries=3)
    with pytest.raises(TranslationError):
        translator.translate(_segs("Hello", "Bye"), "en", "zh")
    assert fake.calls == calls


def test_long_runs_are_squeezed_in_the_prompt_but_kept_in_the_output():
    moan = "気持ちいい" + "あ" * 400
    translator, fake = _translator([ServiceError("x", 500)] * 3 + [ServiceError("x", 500), "嗯"])
    out = translator.translate(_segs(moan, "うん"), "ja", "zh")

    assert all("あ" * 6 not in prompt for prompt in fake.prompts)
    assert "気持ちいいあああああ" in fake.prompts[0]
    assert [s.text for s in out] == [moan, "嗯"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("あ" * 400, "あ" * 5),
        ("ああ " * 30, "ああ " * 5),
        ("お父さん、" * 12, "お父さん、" * 5),
        ("あああああ", "あああああ"),
        ("1000000円", "1000000円"),
        ("やめてやめて、本当に", "やめてやめて、本当に"),
    ],
    ids=["one-char", "with-spaces", "phrase", "five-kept", "digits-kept", "short-kept"],
)
def test_squeeze_repeats(text, expected):
    assert squeeze_repeats(text) == expected


def test_every_request_caps_its_reply_by_its_segment_count():
    translator, fake = _translator([_good({"1": "你好"}), "再见"])
    translator.translate_batch(_segs("Hello", "Bye"), "en", "zh")

    assert fake.caps == [reply_token_cap(2), reply_token_cap(1)]
    assert reply_token_cap(2) > reply_token_cap(1)


def test_a_segment_the_model_never_answered_is_reported_as_untranslated(caplog):
    translator, _ = _translator([_good({"1": "你好"}), ""])
    with caplog.at_level(logging.WARNING):
        translator.translate(_segs("Hello", "Bye"), "en", "zh")

    degradation = translator.untranslated_degradation("zh", 2)

    assert degradation is not None
    assert degradation.code == "translation_incomplete"
    assert degradation.message.startswith("zh: 1 of 2")
    assert "1 left untranslated" in caplog.text


def test_an_answer_identical_to_the_source_is_not_reported(caplog):
    # A name translates to itself; the individual retry agrees, so nothing is missing.
    translator, _ = _translator([_good({"1": "你好", "2": "Tokyo"}), "Tokyo"])
    with caplog.at_level(logging.WARNING):
        translator.translate(_segs("Hello", "Tokyo"), "en", "zh")

    assert translator.untranslated_degradation("zh", 2) is None
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
