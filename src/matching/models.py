"""Data models for Phase 5 Dialogue Matching module."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WordTimestamp:
    """Word-level timing metadata."""

    word: str
    start: float     # timestamp in seconds
    end: float       # timestamp in seconds
    probability: float = 1.0


@dataclass
class WordWindow:
    """A sliding window sequence of consecutive words."""

    words: List[WordTimestamp]
    raw_text: str
    normalized_text: str
    start_time: float
    end_time: float


@dataclass
class MatchResult:
    """Result of searching a target dialogue query against a transcript."""

    match_found: bool
    matched_text: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: float = 0.0
    matched_window_raw_text: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "match_found": self.match_found,
            "matched_text": self.matched_text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": round(self.confidence, 2),
            "matched_window_raw_text": self.matched_window_raw_text,
        }
