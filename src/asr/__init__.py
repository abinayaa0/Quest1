"""
ASR Speech Recognition Module — Phase 4
========================================

Transcribes 16kHz mono WAV audio into timestamped segments and words using Faster-Whisper.

Usage::

    from asr import transcribe_audio

    result = transcribe_audio("output/248244667877.wav")
    print("Language:", result.language)
    for seg in result.segments:
        print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
        for w in seg.words:
            print(f"  word: {w.word} ({w.start:.2f}s - {w.end:.2f}s)")
"""

import logging
from pathlib import Path
from typing import Optional, Union

from .models import (
    WordTimestamp,
    TranscriptSegment,
    TranscriptionResult,
)
from .errors import (
    ASRError,
    AudioNotFoundError,
    ModelLoadError,
    TranscriptionError,
)
from .transcriber import transcribe_audio_file

logger = logging.getLogger(__name__)

__all__ = [
    "transcribe_audio",
    "WordTimestamp",
    "TranscriptSegment",
    "TranscriptionResult",
    "ASRError",
    "AudioNotFoundError",
    "ModelLoadError",
    "TranscriptionError",
]


def transcribe_audio(
    audio_path: Union[str, Path],
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
    mode: str = "standard",
    target_query: Optional[str] = None,
    coarse_model_size: str = "base",
    fine_model_size: str = "small",
    padding_seconds: float = 5.0,
) -> TranscriptionResult:
    """
    Transcribe a local audio file to segment and word-level timestamps.

    Args:
        audio_path: Path to the local input audio file (.wav, .mp3, etc.).
        model_size: Faster-Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3'). Default 'base'.
        device: Device to execute model ('cpu' default).
        compute_type: Computation precision ('int8' default for CPU efficiency).
        language: Optional ISO language code (e.g. 'en', 'ru'). Defaults to None (auto-detect).
        mode: ASR pipeline mode ('standard' for V1 default, 'v2' for V2 optimization).
        target_query: Spoken dialogue query string required if mode='v2'.
        coarse_model_size: Stage 1 coarse Whisper model size (default 'base').
        fine_model_size: Stage 3 fine Whisper model size (default 'small').
        padding_seconds: Padding in seconds before & after candidate region (default 5.0s).

    Returns:
        TranscriptionResult object.

    Raises:
        AudioNotFoundError: Audio file missing or empty.
        ModelLoadError: Faster-Whisper model failed to load.
        TranscriptionError: ASR inference failed.
    """
    if mode in ("v2", "coarse_to_fine"):
        if not target_query:
            raise TranscriptionError("target_query is required when mode='v2'")
        from .coarse_to_fine import transcribe_audio_coarse_to_fine

        return transcribe_audio_coarse_to_fine(
            audio_path=audio_path,
            target_query=target_query,
            coarse_model_size=coarse_model_size,
            fine_model_size=fine_model_size,
            device=device,
            compute_type=compute_type,
            padding_seconds=padding_seconds,
            language=language,
        )

    return transcribe_audio_file(
        audio_path=audio_path,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
    )
