"""Prompt library template-id containment tests.

Template ids become filenames verbatim ({id}.json) — same containment rule as
language codes: without validation, `config save-prompt --id ../../x` wrote
JSON outside the templates directory, and a crafted id inside an existing JSON
file steered delete_user_template's unlink the same way.
"""

import pytest

from subloom.core.prompt_library import PromptLibrary
from subloom.models.prompt import PromptTemplate

_VALID_TEMPLATE = "translate {segments} from {source_lang} to {target_lang}"


def _template(template_id):
    return PromptTemplate(
        id=template_id,
        name="test",
        description="",
        template=_VALID_TEMPLATE,
        genre="custom",
        author="user",
    )


@pytest.fixture()
def library(tmp_path, monkeypatch):
    monkeypatch.setattr(PromptLibrary, "USER_TEMPLATES_DIR", tmp_path)
    return PromptLibrary()


@pytest.mark.parametrize(
    "bad_id",
    ["../../evil", "..", "a/b", "a\\b", "/tmp/x", "C:/x", ".hidden", ""],
)
def test_save_rejects_unsafe_ids(library, tmp_path, bad_id):
    with pytest.raises(ValueError):
        library.save_user_template(_template(bad_id))
    assert list(tmp_path.rglob("*")) == []


def test_save_and_delete_round_trip_with_safe_id(library, tmp_path):
    path = library.save_user_template(_template("my-scifi_v1.2"))
    assert path.parent == tmp_path
    assert library.is_user_defined("my-scifi_v1.2")
    assert library.delete_user_template("my-scifi_v1.2") is True
    assert not path.exists()


def test_crafted_id_inside_json_file_is_not_loaded(library, tmp_path):
    # The id FIELD (not the filename) is what save/delete join into paths, so
    # a planted file must not smuggle one in.
    (tmp_path / "innocent.json").write_text(
        '{"id": "../../evil", "name": "x", "template": "%s"}' % _VALID_TEMPLATE,
        encoding="utf-8",
    )
    assert library.is_user_defined("../../evil") is False
