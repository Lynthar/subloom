"""Lines Whisper writes over audio without speech, the outros and pleasantries of the videos it
learned from: "ご視聴ありがとうございました", "おやすみなさい"."""

import re
from typing import Callable

# Never film dialogue, so a unit holding only these goes whatever the audio.
_NEVER_SAID = (
    "ご視聴ありがとうございました",
    "ご覧いただきありがとうございました",
    "チャンネル登録よろしくお願いします",
    "by H.",
    "-end-",
)
# Said in films too, where the audio around them is speech: such a unit goes only when Whisper
# doubted anything was said. Real ones alone in a WhisperX chunk scored 0.01-0.02.
_PLEASANTRIES = (
    "おやすみなさい",
    "おやすみ",
    "どうもありがとうございました",
    "ありがとうございました",
    "お疲れ様でした",
    "おわり",
    "それではまた",
    "バイバイ",
    "さようなら",
)
NO_SPEECH_GATE = 0.1

_PUNCTUATION = re.compile(r"[\s。、，,．.！!？?…]+")


def _normalize(text: str) -> str:
    return _PUNCTUATION.sub("", text).lower()


def _any_of(phrases: tuple) -> "re.Pattern[str]":
    return re.compile("(?:" + "|".join(re.escape(_normalize(p)) for p in phrases) + ")+")


_ONLY_NEVER_SAID = _any_of(_NEVER_SAID)
_ONLY_KNOWN = _any_of(_NEVER_SAID + _PLEASANTRIES)


def is_hallucination(text: str, no_speech_prob: Callable[[], float]) -> bool:
    """Whether ``text`` is nothing but phrases Whisper makes up over audio without speech.

    Args:
        text: one unit as the recogniser gave it, a WhisperX chunk or a Whisper segment, never a
            split subtitle: splitting "お疲れ様でした。先に失礼します。" leaves a real
            pleasantry a line of its own.
        no_speech_prob: Whisper's probability that the unit holds no speech; called only when
            the text has a pleasantry, which goes when it is above NO_SPEECH_GATE.
    """
    normalized = _normalize(text)
    if not normalized:
        return False
    if _ONLY_NEVER_SAID.fullmatch(normalized):
        return True
    return bool(_ONLY_KNOWN.fullmatch(normalized)) and no_speech_prob() > NO_SPEECH_GATE
