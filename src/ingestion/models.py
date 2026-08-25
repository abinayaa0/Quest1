"""Data models and exceptions for the ingestion module."""

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class IngestionError(Exception):
    """Base exception for all ingestion failures."""
    pass


class DownloadError(IngestionError):
    """Video download failed."""
    pass


class ValidationError(IngestionError):
    """Downloaded file failed validation (missing streams, corrupt, etc.)."""
    pass


class DependencyError(IngestionError):
    """Required external tool (yt-dlp, ffmpeg, ffprobe) not found."""
    pass


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class VideoMetadata:
    """Metadata extracted from a video file via ffprobe."""

    duration: float           # seconds
    width: int
    height: int
    fps: str                  # real frame rate fraction string, e.g. "25/1" or "24000/1001"
    avg_fps: str              # average frame rate fraction string
    is_vfr: bool              # True if Variable Frame Rate (avg_fps != fps)
    video_codec: str          # e.g. "h264"
    audio_codec: str          # e.g. "aac"
    container_format: str     # e.g. "mov,mp4,m4a,3gp,3g2,mj2"
    num_video_streams: int
    num_audio_streams: int


@dataclass
class IngestionResult:
    """Result of a successful video ingestion."""

    video_path: Path
    source_url: str
    metadata: VideoMetadata
    ingestion_method: str             # "yt-dlp" or "direct-ffmpeg"
    download_duration_seconds: float
