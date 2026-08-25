"""Text normalization utility for deterministic dialogue matching."""

import re
import string


def normalize_text(text: str) -> str:
    """
    Normalize string for fuzzy dialogue matching:
    1. Convert to lowercase.
    2. Remove punctuation marks.
    3. Collapse multiple whitespace characters into single spaces.

    Example:
        "My mind, rebels at stagnation!" -> "my mind rebels at stagnation"
    """
    if not text:
        return ""

    # Convert to lower case
    cleaned = text.lower()

    # Remove punctuation
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))

    # Replace hyphens or special non-alphanumeric separators with space
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    # Collapse multiple whitespace
    cleaned = " ".join(cleaned.split())

    return cleaned
