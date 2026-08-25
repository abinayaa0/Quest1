"""Data models for ASR transcription output."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class WordTimestamp:
    """Word-level timing and confidence metadata."""

    word: str
    start: float           # timestamp in seconds
    end: float             # timestamp in seconds
    probability: float = 1.0


@dataclass
class TranscriptSegment:
    """Segment-level timing, text, and word-level timestamps."""

    text: str
    start: float           # segment start in seconds
    end: float             # segment end in seconds
    words: List[WordTimestamp] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Full transcription result returned by transcribe_audio()."""

    audio_path: Path
    segments: List[TranscriptSegment]
    language: str
    language_probability: float
    duration: float
    model_name: str
    transcription_duration_seconds: float

    @property
    def full_text(self) -> str:
        """Returns concatenated full transcript text."""
        return " ".join(seg.text.strip() for seg in self.segments if seg.text.strip())
