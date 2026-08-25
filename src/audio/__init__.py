"""
Audio Extraction Module — Phase 3
==================================

Extracts 16kHz mono PCM WAV audio from local video files for ASR processing.

Usage::

    from audio import extract_audio

    result = extract_audio("c:/Quest1/output/248244667877.mp4")
    print(result.audio_path)
    print(result.metadata.duration)
    print(result.metadata.sample_rate)  # 16000
    print(result.metadata.channels)     # 1
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

from .models import AudioMetadata, AudioResult, AudioExtractionError
from .extractor import extract_audio_stream
from .probe import probe_audio_file

logger = logging.getLogger(__name__)

__all__ = [
    "extract_audio",
    "AudioResult",
    "AudioMetadata",
    "AudioExtractionError",
]


def extract_audio(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> AudioResult:
    """
    Extract 16kHz mono PCM WAV audio from a local video file.

    Args:
        video_path: Path to the input video file.
        output_path: Optional target path for the output WAV file.
                     Defaults to <video_name>.wav in the same directory or temp dir.

    Returns:
        AudioResult containing audio_path and AudioMetadata.

    Raises:
        AudioExtractionError: If file is missing, FFmpeg fails, or output WAV validation fails.
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise AudioExtractionError(f"Input video file does not exist: {video_path}")

    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    else:
        output_path = Path(output_path).resolve()

    start_time = time.time()

    # Step 1: Extract audio stream using FFmpeg (-vn -ac 1 -ar 16000)
    wav_path = extract_audio_stream(video_path, output_path)

    # Step 2: Validate extracted audio file using ffprobe
    metadata = probe_audio_file(wav_path)

    elapsed = round(time.time() - start_time, 2)

    logger.info(
        f"Audio extracted successfully in {elapsed}s: {wav_path} "
        f"({metadata.sample_rate}Hz, {metadata.channels}ch, {metadata.duration:.1f}s)"
    )

    return AudioResult(
        audio_path=wav_path,
        source_video_path=video_path,
        metadata=metadata,
        extraction_duration_seconds=elapsed,
    )
