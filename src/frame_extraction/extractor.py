"""FFmpeg timestamp seeking and video frame extraction implementation."""

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Union

from .errors import (
    FFmpegFrameError,
    FrameExtractionError,
    InvalidTimestampError,
    InvalidVideoError,
)
from .models import FrameResult

logger = logging.getLogger(__name__)


def probe_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Probe extracted image width and height using ffprobe."""
    if not shutil.which("ffprobe"):
        raise FFmpegFrameError("ffprobe is not installed on PATH")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        " -show_streams",
        "-show_format",
        str(image_path),
    ]

    try:
        res = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(image_path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        data = json.loads(res.stdout)
        streams = data.get("streams", [])
        for s in streams:
            if s.get("codec_type") == "video":
                w = int(s.get("width", 0))
                h = int(s.get("height", 0))
                if w > 0 and h > 0:
                    return w, h
    except Exception as e:
        logger.warning(f"Could not probe image dimensions for {image_path}: {e}")

    return 0, 0


def extract_frame(
    video_path: Union[str, Path],
    timestamp: float,
    output_dir: Optional[Union[str, Path]] = None,
    output_filename: Optional[str] = None,
) -> FrameResult:
    """
    Extract a single video frame at the specified timestamp using FFmpeg.

    Args:
        video_path: Path to the input local video file (.mp4, .mkv, etc.).
        timestamp: Timestamp in seconds (must be >= 0).
        output_dir: Directory where the extracted frame will be saved. Default 'output/frames'.
        output_filename: Optional custom filename (e.g. 'frame_320_48.jpg').

    Returns:
        FrameResult containing frame_path, timestamp, frame_number, width, height.

    Raises:
        InvalidVideoError: Video file missing or empty.
        InvalidTimestampError: Timestamp < 0.
        FFmpegFrameError: FFmpeg subprocess failed.
    """
    path = Path(video_path).resolve()
    if not path.exists():
        raise InvalidVideoError(f"Video file does not exist: {path}")

    if path.stat().st_size == 0:
        raise InvalidVideoError(f"Video file is empty: {path}")

    if timestamp < 0:
        raise InvalidTimestampError(f"Invalid negative timestamp: {timestamp}")

    if not shutil.which("ffmpeg"):
        raise FFmpegFrameError("ffmpeg is not installed on PATH")

    # Determine output directory
    if output_dir is None:
        out_dir = path.parent / "frames"
    else:
        out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Format default filename: frame_320_48.jpg
    if not output_filename:
        sec_part = int(timestamp)
        ms_part = int(round((timestamp % 1) * 100))
        output_filename = f"frame_{sec_part}_{ms_part:02d}.jpg"

    frame_path = out_dir / output_filename

    start_time = time.time()
    logger.info(f"Extracting frame at timestamp {timestamp:.2f}s from {path} -> {frame_path}")

    # FFmpeg command for accurate timestamp seeking
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(timestamp),
        "-i", str(path),
        "-vframes", "1",
        "-q:v", "2",  # High JPEG quality
        str(frame_path),
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if res.returncode != 0 or not frame_path.exists() or frame_path.stat().st_size == 0:
            raise FFmpegFrameError(
                f"FFmpeg failed to extract frame at {timestamp}s: {res.stderr}"
            )
    except Exception as e:
        if isinstance(e, FFmpegFrameError):
            raise
        raise FFmpegFrameError(f"Subprocess error extracting frame: {e}") from e

    elapsed = round(time.time() - start_time, 3)

    # Probe image dimensions
    width, height = probe_image_dimensions(frame_path)

    # Frame number cannot be deterministically proven for VFR without full index scan -> None
    frame_number = None

    logger.info(
        f"Extracted frame successfully in {elapsed}s: "
        f"res={width}x{height}, size={frame_path.stat().st_size} bytes"
    )

    return FrameResult(
        frame_path=frame_path,
        timestamp=timestamp,
        frame_number=frame_number,
        width=width,
        height=height,
        extraction_duration_seconds=elapsed,
    )
