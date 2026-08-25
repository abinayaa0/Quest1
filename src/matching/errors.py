"""Exception hierarchy for Phase 5 Dialogue Matching module."""


class MatchingError(Exception):
    """Base exception for all dialogue matching failures."""
    pass


class EmptyQueryError(MatchingError):
    """Target query string is empty or whitespace-only."""
    pass


class EmptyTranscriptError(MatchingError):
    """Input transcript is empty or contains no words."""
    pass


class NoWordsError(MatchingError):
    """No valid word-level timestamps available in transcript."""
    pass
