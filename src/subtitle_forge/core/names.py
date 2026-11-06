"""The people a transcript names and every way the recogniser spelled them, read off the whole
transcript in one request so the translator can write each name the same way throughout."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..models.subtitle import SubtitleSegment
from .chat import ChatBackend
from .interjections import is_interjection
from .json_reply import extract_json_object
from .timestamp_processor import display_cells

logger = logging.getLogger(__name__)

# Thinking models loop on a whole film's text under greedy decoding (Qwen3 says so, and
# qwen3.8:27b did), so this request samples; a reply without the JSON is asked again.
TEMPERATURE = 0.6
ATTEMPTS = 3
MAX_TOKENS = 16384  # Thinking included: up to ~20,000 characters of it on a two-hour film
# Leaves out the katakana middle dot, which Chinese names use as a separator too.
KANA = re.compile(r"[ぁ-ゖァ-ヺ]")
# Whole prompts measured 0.26-0.38 tokens per half-width cell (ja, zh, en, ko on two models). A
# server that cuts a prompt keeps at most half of it, so a count below this means it was cut.
MIN_TOKENS_PER_CELL = 0.2

# The transcript goes first: Ollama cuts a prompt over its context down to the first 4 tokens
# and the tail, and the task must survive that.
PROMPT = """TRANSCRIPT ({source_lang}, from speech recognition, one subtitle per line):
{transcript}

TASK - The transcript above comes from speech recognition, so one person's name may be written in
several ways: other characters with the same reading, or a mishearing. Find the people (and pets)
who are called or mentioned by name.
{official}For each person, give the name to use in {target_lang} subtitles and every spelling of that
name exactly as it occurs in the transcript. List a spelling only if, where it occurs, it is used as
this person's name: not an ordinary word that a few lines might have misheard, and never a single
character.

Return STRICTLY a JSON object of the form:
{{"names": [{{"name": "<name for the subtitles>", "spellings": ["<as written in the transcript>", ...]}}]}}
List only personal names that occur in the transcript; return {{"names": []}} if there are none.
Keep your reasoning short. After it, write the JSON object as your answer, and nothing else.
"""
OFFICIAL = (
    "The characters' official names are: {names}. Use exactly these as the names, and map every "
    "spelling of each to it.\n"
)


@dataclass(frozen=True)
class Character:
    """A name as the translation must write it, and how the transcript spells it."""

    name: str
    spellings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class NameList:
    """What a name pass found; ``partial`` when the server read only part of the transcript."""

    characters: List[Character]
    partial: bool = False


def find_characters(
    backend: ChatBackend,
    segments: Sequence[SubtitleSegment],
    source_lang: str,
    target_lang: str,
    official: Sequence[str] = (),
) -> Optional[NameList]:
    """The people the transcript names, each official name among them; None when the model
    gave no usable answer in ATTEMPTS tries. Takes language names, not codes.

    Raises:
        ServiceError, httpx.TransportError: As ChatBackend.chat; the caller retries those.
    """
    lines = [s.text.strip() for s in segments if s.text.strip() and not is_interjection(s.text)]
    transcript = "\n".join(dict.fromkeys(lines))
    prompt = PROMPT.format(
        source_lang=source_lang,
        target_lang=target_lang,
        transcript=transcript,
        official=OFFICIAL.format(names=", ".join(official)) if official else "",
    )
    for attempt in range(1, ATTEMPTS + 1):
        reply = backend.chat(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        found = _parse(reply)
        if found is not None:
            tokens = backend.last_prompt_tokens
            return NameList(
                _checked(found, transcript, target_lang, official),
                partial=bool(tokens and tokens < MIN_TOKENS_PER_CELL * display_cells(prompt)),
            )
        logger.debug(f"Name pass attempt {attempt} gave no JSON: {reply[:200]!r}")
    return None


def prompt_block(characters: Sequence[Character]) -> str:
    """The lines a translation prompt carries about the names; empty when there are none."""
    if not characters:
        return ""
    lines = [
        f"- {c.name}"
        + (f" (the transcript may write: {', '.join(c.spellings)})" if c.spellings else "")
        for c in characters
    ]
    return (
        "\nCHARACTER NAMES - write these people's names exactly as given. The transcript comes "
        "from speech recognition; where one of the listed spellings refers to the person, "
        "translate it as that name:\n" + "\n".join(lines) + "\n"
    )


def _parse(reply: str) -> Optional[list]:
    blob = extract_json_object(reply)
    try:
        names = json.loads(blob).get("names") if blob else None
    except (ValueError, AttributeError):
        return None
    return names if isinstance(names, list) else None


def _checked(
    found: list, transcript: str, target_lang: str, official: Sequence[str]
) -> List[Character]:
    """Spellings that really occur, two characters or longer, each claimed once; every official
    name kept even when the model missed it. A name left in kana is dropped for non-Japanese
    subtitles: the translator copies listed names exactly, and it would show up as kana."""
    characters: List[Character] = []
    claimed: set = set()
    for entry in found:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            continue
        name = entry["name"].strip()
        if target_lang != "Japanese" and KANA.search(name) and name not in official:
            continue
        raw = entry.get("spellings")
        spellings = [
            s.strip()
            for s in (raw if isinstance(raw, list) else [])
            if isinstance(s, str) and len(s.strip()) >= 2 and s.strip() in transcript
        ]
        spellings = [s for s in dict.fromkeys(spellings) if s not in claimed]
        claimed.update(spellings)
        if name and (spellings or name in official):
            characters.append(Character(name, spellings))
    named = {c.name for c in characters}
    characters += [Character(n) for n in official if n not in named]
    return characters
