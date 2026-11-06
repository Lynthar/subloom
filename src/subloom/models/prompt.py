"""Prompt template data model."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PromptTemplate:
    """A prompt template with metadata for translation."""

    id: str  # Unique identifier: "movie-scifi", "documentary", etc.
    name: str  # Display name: "科幻电影", "纪录片"
    description: str  # Description: "适用于科幻、太空、未来题材"
    template: str  # The actual template with placeholders
    genre: str  # Category: "movie", "tv", "documentary", "adult", "technical"
    tags: List[str] = field(default_factory=list)  # Tags: ["科幻", "动作", "技术术语"]
    author: str = "builtin"  # Who created this template
    version: str = "1.0"  # Template version

    def validate(self) -> List[str]:
        """
        Validate the template has required placeholders.

        Returns:
            List of missing placeholder names (empty if valid).
        """
        required = ["{source_lang}", "{target_lang}", "{segments}"]
        missing = [p for p in required if p not in self.template]
        return missing

    def is_valid(self) -> bool:
        """Check if template is valid."""
        return len(self.validate()) == 0
