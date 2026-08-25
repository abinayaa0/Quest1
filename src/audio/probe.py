"""ffprobe wrapper for WAV audio file validation."""

import json
import logging
import shutil
import subprocess
from pathlib import Path

from .models import AudioMetadata, AudioExtractionError

logger = logging.getLogger(__name__)


def probe_audio_file(path: Path) -> AudioMetadata:
    """
    Validate a WAV audio file and extract metadata using ffprobe.

    Checks:
    - File exists and is non-empty
    - Contains at least one audio stream
    - Sample rate is 16000 Hz
    - Channels count is 1 (mono)
    - Duration is valid (> 0)

    Returns:
        AudioMetadata on success.

    Raises:
        AudioExtractionError: file is invalid or missing required streams.
    """
    if not shutil.which("ffprobe"):
        raise AudioExtractionError("ffprobe not found on PATH.")

    path = Path(path)
    if not path.exists():
        raise AudioExtractionError(f"Audio file does not exist: {path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise AudioExtractionError(f"Audio file is empty: {path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise AudioExtractionError(f"ffprobe timed out on: {path}")

    if result.returncode != 0:
        raise AudioExtractionError(
            f"ffprobe failed (exit {result.returncode}): {result.stderr.strip()}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise AudioExtractionError(f"ffprobe returned invalid JSON: {e}")

    streams = data.get("streams", [])
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not audio_streams:
        raise AudioExtractionError("No audio stream found in extracted WAV file.")

    audio = audio_streams[0]
    fmt = data.get("format", {})

    duration_str = fmt.get("duration") or audio.get("duration") or "0"
    try:
        duration = float(duration_str)
    except (ValueError, TypeError):
        duration = 0.0

    if duration <= 0:
        raise AudioExtractionError(f"Invalid audio duration: {duration_str}")

    sample_rate = int(audio.get("sample_rate", 0))
    channels = int(audio.get("channels", 0))
    codec_name = audio.get("codec_name", "unknown")

    logger.info(
        f"Audio Probe: {codec_name}, {sample_rate}Hz, {channels}ch, "
        f"{duration:.1f}s, {file_size / (1024*1024):.2f} MB"
    )

    return AudioMetadata(
        duration=duration,
        sample_rate=sample_rate,
        channels=channels,
        codec_name=codec_name,
        file_size_bytes=file_size,
    )
