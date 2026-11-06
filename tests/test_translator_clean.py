"""Tests for translation text cleanup.

Locks in the fix for the silent content-corruption bug where _clean_translation
stripped a *bare* leading number, so any translation that legitimately began
with a digit lost it (and a pure-number line was erased entirely).
"""

from types import SimpleNamespace

from subloom.core.translator import SubtitleTranslator
from subloom.models.config import TranslationConfig
from subloom.models.subtitle import SubtitleSegment

# These tests never send a request.
_NO_CHAT = SimpleNamespace(model="qwen2.5:14b")


def _json_mode_translator():
    # Default config -> JSON mode (no custom/library prompt).
    return SubtitleTranslator(TranslationConfig(), _NO_CHAT)


def _legacy_mode_translator():
    # A library template forces the legacy [N]-line path.
    return SubtitleTranslator(TranslationConfig(prompt_template_id="movie-general"), _NO_CHAT)


def test_json_mode_skips_index_stripping_entirely():
    t = _json_mode_translator()
    assert t._is_json_mode()
    for text in ["10時に会いましょう", "3 days later", "42", "[5] still literal"]:
        assert t._clean_translation(text, "orig text here") == text


def test_legacy_mode_strips_only_bracketed_indices():
    t = _legacy_mode_translator()
    assert not t._is_json_mode()
    assert t._clean_translation("[5] hello", "x") == "hello"
    assert t._clean_translation("(3) world", "x") == "world"


def test_legacy_mode_preserves_leading_bare_numbers():
    t = _legacy_mode_translator()
    # These are the regression cases: bare leading digits must NOT be eaten.
    assert t._clean_translation("3 days later", "three days") == "3 days later"
    assert t._clean_translation("2024年", "year 2024") == "2024年"
    assert t._clean_translation("42", "forty two") == "42"
    assert t._clean_translation("007 is a spy", "spy") == "007 is a spy"


def test_short_translation_falls_back_to_original():
    t = _legacy_mode_translator()
    assert t._clean_translation("x", "a long original sentence") == "a long original sentence"


def test_positional_fallback_preserves_bare_leading_numbers():
    # Legacy mode, model answered with a bare line (no [N] marker): the
    # positional fallback's inline cleanup must not eat a legitimate leading
    # number — the same corruption class _clean_translation was fixed for.
    t = _legacy_mode_translator()
    seg = SubtitleSegment(index=7, start=0.0, end=1.0, text="三日後")
    out = t._parse_translation_response("3 days later", [seg])
    assert out[0].text == "3 days later"


def test_positional_fallback_still_strips_real_index_markers():
    t = _legacy_mode_translator()
    seg = SubtitleSegment(index=7, start=0.0, end=1.0, text="hello")
    out = t._parse_translation_response("7. bonjour", [seg])
    assert out[0].text == "bonjour"
