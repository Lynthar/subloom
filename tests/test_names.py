"""The name pass: what it sends, what of the reply it keeps, when it gives up, and how the
translator carries the names into every prompt. A fake chat backend stands in for the server."""

import json

import pytest

from subloom.core import names as names_module
from subloom.core.names import Character, find_characters, prompt_block
from subloom.core.translator import SubtitleTranslator
from subloom.models.config import TranslationConfig
from subloom.models.degradation import NAMES_NOT_FOUND, NAMES_PARTIAL
from subloom.models.subtitle import SubtitleSegment

LINES = ["みつきは可愛い。", "三月に彼氏？", "うん、うん。", "みつきは可愛い。", "ことねです。"]


def _segs(*texts):
    return [
        SubtitleSegment(index=i + 1, start=float(i), end=float(i) + 0.9, text=t)
        for i, t in enumerate(texts)
    ]


class _Chat:
    """Answers name-pass prompts from ``passes`` in order and translation prompts by echoing."""

    model = "qwen3.8:27b"

    def __init__(self, *passes, tokens=None):
        self.passes = list(passes)
        self.calls = []  # (prompt, temperature)
        self.last_prompt_tokens = tokens

    def chat(self, prompt, *, json_keys=None, max_tokens=None, temperature=None):
        self.calls.append((prompt, temperature))
        if prompt.startswith("TRANSCRIPT"):
            return self.passes.pop(0)
        if json_keys is not None:
            return json.dumps({"translations": {k: f"<{k}>" for k in json_keys}})
        return "<one>"


def _names(*entries):
    return json.dumps({"names": [{"name": n, "spellings": list(s)} for n, s in entries]})


def test_the_transcript_goes_first_once_and_without_interjection_lines():
    chat = _Chat(_names())

    find_characters(chat, _segs(*LINES), "Japanese", "Simplified Chinese")

    ((prompt, temperature),) = chat.calls
    transcript = prompt[: prompt.index("TASK")]
    assert transcript.count("みつきは可愛い。") == 1 and "ことねです。" in transcript
    assert "うん、うん。" not in prompt
    assert temperature == names_module.TEMPERATURE


def test_only_spellings_that_occur_and_are_longer_than_one_character_are_kept():
    reply = _names(
        ("美月", ["みつき", "三月", "月", "美月"]),
        ("光彦", ["三月"]),
        ("琴音", ["ことね"]),
    )

    found = find_characters(_Chat(reply), _segs(*LINES), "Japanese", "Simplified Chinese")

    assert found.characters == [
        Character("美月", ["みつき", "三月"]),
        Character("琴音", ["ことね"]),
    ]


def test_a_name_left_in_kana_is_dropped_unless_the_subtitles_are_japanese():
    reply = _names(("みつき", ["みつき"]), ("安・琴音", ["ことね"]))

    to_chinese = find_characters(_Chat(reply), _segs(*LINES), "Japanese", "Simplified Chinese")
    to_japanese = find_characters(_Chat(reply), _segs(*LINES), "English", "Japanese")

    assert to_chinese.characters == [Character("安・琴音", ["ことね"])]
    assert [c.name for c in to_japanese.characters] == ["みつき", "安・琴音"]


def test_an_official_name_in_kana_is_kept_as_given():
    reply = _names(("みつき", ["みつき"]))

    found = find_characters(_Chat(reply), _segs(*LINES), "Japanese", "Chinese", ["みつき"])

    assert found.characters == [Character("みつき", ["みつき"])]


def test_official_names_reach_the_prompt_and_survive_a_reply_that_drops_one():
    chat = _Chat("```json\n" + _names(("光希", ["みつき"])) + "\n```")

    found = find_characters(chat, _segs(*LINES), "Japanese", "Simplified Chinese", ["光希", "琴音"])

    assert "光希, 琴音" in chat.calls[0][0]
    assert found.characters == [Character("光希", ["みつき"]), Character("琴音")]


def test_a_reply_without_the_json_is_asked_again_then_given_up_on():
    late = _Chat("", _names(("美月", ["みつき"])))
    never = _Chat(*[""] * names_module.ATTEMPTS)

    found = find_characters(late, _segs(*LINES), "ja", "zh")

    assert found.characters == [Character("美月", ["みつき"])]
    assert find_characters(never, _segs(*LINES), "ja", "zh") is None
    assert len(never.calls) == names_module.ATTEMPTS


# About 27,000 half-width cells: read whole that is 7,000 tokens or more, while a server with a
# 4k context keeps about 2,050 of it.
LONG_FILM = [f"これは{i}番目の台詞です。" for i in range(1000)]


@pytest.mark.parametrize(
    "tokens, partial",
    [(2050, True), (9000, False), (None, False), (0, False)],
    ids=["cut", "whole", "server did not say", "server reported nothing read"],
)
def test_a_prompt_the_server_read_only_part_of_is_flagged(tokens, partial):
    chat = _Chat(_names(("美月", ["これは"])), tokens=tokens)

    found = find_characters(chat, _segs(*LONG_FILM), "Japanese", "Simplified Chinese")

    assert found.partial is partial
    assert found.characters == [Character("美月", ["これは"])]


def test_the_prompt_block_lists_each_name_with_its_spellings():
    block = prompt_block([Character("光希", ["みつき", "三月"]), Character("琴音")])

    assert "- 光希 (the transcript may write: みつき, 三月)\n" in block
    assert "- 琴音\n" in block
    assert prompt_block([]) == ""


def _translator(chat, **settings):
    return SubtitleTranslator(TranslationConfig(**settings), chat)


@pytest.mark.parametrize("template", [None, "movie-general"], ids=["json", "library template"])
def test_every_batch_and_retry_prompt_carries_the_names(template):
    chat = _Chat(_names(("光希", ["みつき", "三月"])))
    translator = _translator(chat, prompt_template_id=template, max_batch_size=2)

    translator.translate(_segs(*LINES), "ja", "zh")
    translator._retry_failed_individually(_segs("三月"), [(0, _segs("三月")[0])], "ja", "zh")

    pass_prompt, *rest = [p for p, _ in chat.calls]
    assert pass_prompt.startswith("TRANSCRIPT")
    assert rest and all("- 光希 (the transcript may write: みつき, 三月)" in p for p in rest)


def test_given_names_without_the_pass_are_listed_as_they_are():
    chat = _Chat()
    translator = _translator(chat, names=["光希", "琴音"], find_names=False)

    translator.translate(_segs(*LINES), "ja", "zh")

    assert all("- 光希\n- 琴音\n" in p for p, _ in chat.calls)
    assert translator.names_degradation() is None


def test_a_failed_pass_is_reported_and_the_given_names_still_go_out():
    chat = _Chat(*[""] * names_module.ATTEMPTS, _names(("光希", ["みつき"])))
    translator = _translator(chat, names=["光希"])

    translator.translate(_segs(*LINES), "ja", "zh")

    assert translator.names_degradation().code == NAMES_NOT_FOUND
    assert "- 光希\n" in chat.calls[-1][0]

    translator.translate(_segs(*LINES), "ja", "en")

    assert translator.names_degradation() is None


def test_a_partial_pass_is_reported_and_what_it_found_still_goes_out():
    chat = _Chat(_names(("美月", ["これは"])), _names(("美月", ["これは"])), tokens=2050)
    translator = _translator(chat)

    translator.translate(_segs(*LONG_FILM[:12]), "ja", "zh")
    assert translator.names_degradation() is None

    translator.translate(_segs(*LONG_FILM), "ja", "en")

    assert translator.names_degradation().code == NAMES_PARTIAL
    assert "- 美月 (the transcript may write: これは)" in chat.calls[-1][0]
