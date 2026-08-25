"""Exception hierarchy for Phase 6 Frame Extraction module."""


class FrameExtractionError(Exception):
    """Base exception for all frame extraction failures."""
    pass


class InvalidVideoError(FrameExtractionError):
    """Video file missing, unreadable, or empty."""
    pass


class InvalidTimestampError(FrameExtractionError):
    """Timestamp is negative or out of bounds."""
    pass


class FFmpegFrameError(FrameExtractionError):
    """FFmpeg subprocess execution failed during frame extraction."""
    pass
