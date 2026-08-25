"""ffprobe wrapper for video file validation and metadata extraction."""

import json
import logging
import shutil
import subprocess
from pathlib import Path

from .models import VideoMetadata, ValidationError, DependencyError

logger = logging.getLogger(__name__)


def probe_file(path: Path) -> VideoMetadata:
    """
    Validate a video file and extract metadata using ffprobe.

    Checks:
    - File exists and is non-empty
    - Contains at least one video stream
    - Contains at least one audio stream
    - Duration is valid (> 0)

    Returns:
        VideoMetadata on success.

    Raises:
        ValidationError: file is invalid or missing required streams.
        DependencyError: ffprobe not installed.
    """
    if not shutil.which("ffprobe"):
        raise DependencyError(
            "ffprobe not found on PATH. Install FFmpeg: https://ffmpeg.org/download.html"
        )

    path = Path(path)
    if not path.exists():
        raise ValidationError(f"File does not exist: {path}")

    if path.stat().st_size == 0:
        raise ValidationError(f"File is empty: {path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    logger.debug(f"Running ffprobe: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise ValidationError(f"ffprobe timed out on: {path}")

    if result.returncode != 0:
        raise ValidationError(
            f"ffprobe failed (exit {result.returncode}): {result.stderr.strip()}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise ValidationError(f"ffprobe returned invalid JSON: {e}")

    # Classify streams
    streams = data.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not video_streams:
        raise ValidationError("No video stream found in the downloaded file.")

    if not audio_streams:
        raise ValidationError("No usable audio stream found in the downloaded video.")

    # Use primary (first) streams
    video = video_streams[0]
    audio = audio_streams[0]
    fmt = data.get("format", {})

    # Duration: prefer format-level, fall back to stream-level
    duration_str = fmt.get("duration") or video.get("duration") or "0"
    try:
        duration = float(duration_str)
    except (ValueError, TypeError):
        duration = 0.0

    if duration <= 0:
        raise ValidationError(f"Invalid video duration: {duration_str}")

    # FPS extraction & VFR detection
    fps = video.get("r_frame_rate") or "0/1"
    avg_fps = video.get("avg_frame_rate") or fps

    def _parse_fraction(f_str: str) -> float:
        try:
            parts = f_str.split("/")
            if len(parts) == 2 and float(parts[1]) != 0:
                return float(parts[0]) / float(parts[1])
            return float(parts[0])
        except (ValueError, TypeError, IndexError):
            return 0.0

    r_fps_val = _parse_fraction(fps)
    avg_fps_val = _parse_fraction(avg_fps)

    # Detect VFR: if difference between real FPS and average FPS is > 0.05
    # or if r_frame_rate is container timebase artifact (e.g. 90000/1)
    is_vfr = False
    if r_fps_val > 0 and avg_fps_val > 0:
        if abs(r_fps_val - avg_fps_val) > 0.05 or r_fps_val > 1000:
            is_vfr = True

    metadata = VideoMetadata(
        duration=duration,
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        fps=fps,
        avg_fps=avg_fps,
        is_vfr=is_vfr,
        video_codec=video.get("codec_name", "unknown"),
        audio_codec=audio.get("codec_name", "unknown"),
        container_format=fmt.get("format_name", "unknown"),
        num_video_streams=len(video_streams),
        num_audio_streams=len(audio_streams),
    )

    if metadata.num_audio_streams > 1:
        logger.warning(
            f"Multiple audio streams detected ({metadata.num_audio_streams}). "
            f"Using the primary/default stream."
        )

    logger.info(
        f"Probe: {metadata.width}x{metadata.height}, {metadata.duration:.1f}s, "
        f"{metadata.fps} fps, v={metadata.video_codec}, a={metadata.audio_codec}, "
        f"streams={metadata.num_video_streams}v/{metadata.num_audio_streams}a"
    )

    return metadata
