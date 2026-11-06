"""Pulling the JSON object out of a model's reply."""

from typing import Optional


def extract_json_object(response: str) -> Optional[str]:
    """The first balanced JSON object in a reply, or None. Skips markdown fences (```json) and
    stray prefixes ("Here is the translation:") that models add despite a JSON constraint."""
    if not response:
        return None

    # Strip markdown code fences if present
    text = response.strip()
    if text.startswith("```"):
        # Drop the opening fence (and optional language tag)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        # Drop trailing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    # Walk the string finding a balanced { ... } block, respecting strings
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
