"""Data models and exceptions for the audio extraction module."""

from dataclasses import dataclass
from pathlib import Path


class AudioExtractionError(Exception):
    """Base exception for audio extraction failures."""
    pass


@dataclass
class AudioMetadata:
    """Metadata extracted from a WAV audio file via ffprobe."""

    duration: float        # duration in seconds
    sample_rate: int       # expected 16000
    channels: int          # expected 1 (mono)
    codec_name: str        # e.g. "pcm_s16le"
    file_size_bytes: int


@dataclass
class AudioResult:
    """Result of a successful audio extraction."""

    audio_path: Path
    source_video_path: Path
    metadata: AudioMetadata
    extraction_duration_seconds: float
