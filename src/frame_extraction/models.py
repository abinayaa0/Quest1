"""Data models for Phase 6 Frame Extraction module."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FrameResult:
    """Metadata result of an extracted video frame."""

    frame_path: Path
    timestamp: float
    frame_number: Optional[int]
    width: int
    height: int
    extraction_duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Convert FrameResult to JSON-serializable dictionary."""
        return {
            "frame_path": str(self.frame_path),
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "width": self.width,
            "height": self.height,
            "extraction_duration_seconds": self.extraction_duration_seconds,
        }
