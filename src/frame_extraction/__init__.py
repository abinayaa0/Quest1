"""
Phase 6 — Frame Extraction Module
====================================

Extracts video frames at specific timestamps using FFmpeg seeking and probes frame dimensions.

Usage::

    from frame_extraction import extract_frame

    result = extract_frame("output/248244667877.mp4", timestamp=320.48)
    print(f"Extracted Frame: {result.frame_path} ({result.width}x{result.height})")
"""

from .errors import (
    FFmpegFrameError,
    FrameExtractionError,
    InvalidTimestampError,
    InvalidVideoError,
)
from .extractor import extract_frame
from .models import FrameResult

__all__ = [
    "extract_frame",
    "FrameResult",
    "FrameExtractionError",
    "InvalidVideoError",
    "InvalidTimestampError",
    "FFmpegFrameError",
]
