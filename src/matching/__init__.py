"""
Phase 5 — Dialogue Matching Module
====================================

Deterministic dialogue search module that maps target dialogue queries to exact timestamps
using RapidFuzz and word-level sliding windows.

Usage::

    from matching import match_dialogue

    result = match_dialogue("My mind rebels at stagnation", transcript)
    if result.match_found:
        print(f"Dialogue found between {result.start_time:.2f}s and {result.end_time:.2f}s")
        print(f"Confidence: {result.confidence:.1f}%")
"""

from .errors import (
    EmptyQueryError,
    EmptyTranscriptError,
    MatchingError,
    NoWordsError,
)
from .matcher import (
    extract_words_from_transcript,
    generate_windows,
    match_dialogue,
)
from .models import (
    MatchResult,
    WordTimestamp,
    WordWindow,
)
from .normalization import normalize_text

__all__ = [
    "match_dialogue",
    "normalize_text",
    "generate_windows",
    "extract_words_from_transcript",
    "MatchResult",
    "WordTimestamp",
    "WordWindow",
    "MatchingError",
    "EmptyQueryError",
    "EmptyTranscriptError",
    "NoWordsError",
]
