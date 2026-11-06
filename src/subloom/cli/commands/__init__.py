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
NAMES_HELP = (
    "Character names as the translation must write them, comma-separated, e.g. 光希,琴音"
    " (default: config translation.names)"
)
FIND_NAMES_HELP = (
    "Read the whole transcript first for the names and how the recogniser spelled them"
    " (default: config translation.find_names)"
)
MERGE_FRAGMENTS_HELP = (
    "Join a subtitle cut mid-sentence to the next one across a short pause"
    " (default: config output.merge_fragments)"
)
