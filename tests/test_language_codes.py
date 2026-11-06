"""Tests for language-code filename safety.

Language codes are embedded verbatim in output filenames ({stem}.{lang}.srt),
so a crafted "language" like ../../x from an API caller could write .srt files
outside the output directory. validate_language_codes is the shared guard
(pipeline choke point + server request validation).
"""

import pytest

from subloom.core.subtitle import validate_language_codes


def test_normal_codes_accepted():
    validate_language_codes(["en", "zh", "zh-TW", "yue", "pt-BR", "zh_Hant"])


def test_path_metacharacters_rejected():
    for bad in ["../../../secret", "zh/..", "a/b", "a\\b", "..", ".hidden", "a.b", "", "zh:x"]:
        with pytest.raises(ValueError):
            validate_language_codes([bad])


def test_overlong_token_rejected():
    with pytest.raises(ValueError):
        validate_language_codes(["x" * 40])
