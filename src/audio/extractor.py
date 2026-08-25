"""FFmpeg wrapper for extracting 16kHz mono WAV audio from video files."""

import logging
import shutil
import subprocess
from pathlib import Path

from .models import AudioExtractionError

logger = logging.getLogger(__name__)


def extract_audio_stream(
    video_path: Path, output_wav_path: Path, timeout: int = 600
) -> Path:
    """
    Extract audio stream from video file to 16kHz mono PCM WAV.

    Command:
        ffmpeg -i <video_path> -vn -ac 1 -ar 16000 -c:a pcm_s16le <output_wav_path>

    Args:
        video_path: Input local video file path.
        output_wav_path: Destination WAV audio file path.
        timeout: Subprocess execution timeout in seconds.

    Returns:
        Path to output WAV file.

    Raises:
        AudioExtractionError: FFmpeg execution failure or missing executable.
    """
    if not shutil.which("ffmpeg"):
        raise AudioExtractionError(
            "ffmpeg not found on PATH. Install FFmpeg: https://ffmpeg.org/download.html"
        )

    video_path = Path(video_path)
    output_wav_path = Path(output_wav_path)

    if not video_path.exists():
        raise AudioExtractionError(f"Video file does not exist: {video_path}")

    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",                   # Overwrite existing output
        "-i", str(video_path),   # Input video file
        "-vn",                  # Disable video recording stream
        "-ac", "1",             # Convert to 1 channel (mono)
        "-ar", "16000",         # Set audio sample rate to 16000 Hz
        "-c:a", "pcm_s16le",    # Standard uncompressed PCM 16-bit LE WAV codec
        str(output_wav_path),
    ]

    logger.info(f"Extracting audio from {video_path} -> {output_wav_path}")
    logger.debug(f"FFmpeg command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        if output_wav_path.exists():
            output_wav_path.unlink()
        raise AudioExtractionError(
            f"FFmpeg audio extraction timed out after {timeout}s"
        )

    if result.returncode != 0:
        if output_wav_path.exists():
            output_wav_path.unlink()
        stderr = result.stderr.strip()
        raise AudioExtractionError(
            f"FFmpeg audio extraction failed (exit {result.returncode}): "
            f"{stderr[-500:] if len(stderr) > 500 else stderr}"
        )

    if not output_wav_path.exists() or output_wav_path.stat().st_size == 0:
        raise AudioExtractionError("FFmpeg completed but output audio file is empty or missing")

    logger.info(
        f"Audio extraction complete: {output_wav_path} "
        f"({output_wav_path.stat().st_size / (1024*1024):.2f} MB)"
    )

    return output_wav_path
