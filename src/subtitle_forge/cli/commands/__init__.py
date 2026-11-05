"""CLI commands."""

from ...models.config import ASR_BACKENDS, TRANSLATION_BACKENDS

ASR_BACKEND_HELP = f"Speech recognition backend: {', '.join(ASR_BACKENDS)} (default: config)"
TRANSLATION_BACKEND_HELP = (
    f"Translation backend: {', '.join(TRANSLATION_BACKENDS)} (default: config translation.backend)"
)
DROP_INTERJECTIONS_HELP = (
    "Leave out subtitles that hold nothing but interjections such as 'uh' or 'ah'"
    " (default: config output.drop_interjections)"
)
MERGE_FRAGMENTS_HELP = (
    "Join a subtitle cut mid-sentence to the next one across a short pause"
    " (default: config output.merge_fragments)"
)
