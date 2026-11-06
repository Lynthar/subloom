"""Model auto-selection tests.

Locks in the tie-break: large-v3 and large-v2 both claim 6000MB, and
select_optimal_model picks the first survivor of a stable sort — with the
table ordered the other way round, every sufficiently large GPU recommended
the older large-v2 and silently downgraded users from the configured default.

No GPU or model download: get_available_vram is patched. The transcriber
module does pull in faster_whisper though, so skip rather than break the
"tests run without torch/whisper installed" contract the guide documents.
"""

import pytest

pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")

from subloom.core.transcriber import Transcriber  # noqa: E402


@pytest.mark.parametrize(
    ("vram_mb", "expected"),
    [
        (32000, "large-v3"),  # high-end card
        (8000, "large-v3"),  # exactly enough for the 6000 tier (20% headroom)
        (6000, "medium"),  # not enough headroom for large-*
        (3000, "small"),
        (0, None),  # detection failed -> no recommendation; the caller decides
    ],
)
def test_select_optimal_model_by_vram(monkeypatch, vram_mb, expected):
    monkeypatch.setattr("subloom.utils.gpu.get_available_vram", lambda: vram_mb)
    monkeypatch.setattr("subloom.core.transcriber.get_available_vram", lambda: vram_mb)
    assert Transcriber.select_optimal_model() == expected


def test_large_v3_precedes_large_v2_in_the_table():
    # The tie-break is positional, so guard the ordering itself: a future
    # edit that re-sorts this dict alphabetically would silently regress.
    keys = list(Transcriber.MODEL_VRAM_REQUIREMENTS)
    assert keys.index("large-v3") < keys.index("large-v2")
