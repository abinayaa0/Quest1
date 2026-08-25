"""Faster-Whisper inference wrapper for speech transcription."""

import logging
import time
from pathlib import Path
from typing import Optional, Union

from .models import WordTimestamp, TranscriptSegment, TranscriptionResult
from .errors import ASRError, AudioNotFoundError, ModelLoadError, TranscriptionError

logger = logging.getLogger(__name__)

# Global model cache to avoid re-loading weights for repeated calls with same parameters
_MODEL_CACHE = {}


def get_whisper_model(
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
):
    """Load or retrieve cached Faster-Whisper model instance."""
    cache_key = (model_size, device, compute_type)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ModelLoadError(
            "faster-whisper is not installed. Install with: pip install faster-whisper"
        ) from e

    logger.info(
        f"Loading Faster-Whisper model '{model_size}' "
        f"(device={device}, compute_type={compute_type})..."
    )

    try:
        model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type,
        )
        _MODEL_CACHE[cache_key] = model
        return model
    except Exception as e:
        raise ModelLoadError(
            f"Failed to load Faster-Whisper model '{model_size}': {e}"
        ) from e


def transcribe_audio_file(
    audio_path: Union[str, Path],
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
    beam_size: int = 5,
) -> TranscriptionResult:
    """
    Transcribe a local audio file to segment and word-level timestamped text using Faster-Whisper.

    Args:
        audio_path: Path to the local input audio file (.wav, .mp3, etc.).
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3'). Default 'base'.
        device: Execution device ('cpu' default).
        compute_type: Computation precision ('int8' default for CPU efficiency).
        language: Optional ISO language code (e.g. 'en', 'ru'). If None, automatically detected.
        beam_size: Beam search width (default 5).

    Returns:
        TranscriptionResult containing segment and word timestamps.

    Raises:
        AudioNotFoundError: Audio file missing or empty.
        ModelLoadError: Faster-Whisper failed to initialize.
        TranscriptionError: Inference execution failed.
    """
    path = Path(audio_path).resolve()
    if not path.exists():
        raise AudioNotFoundError(f"Audio file does not exist: {path}")

    if path.stat().st_size == 0:
        raise AudioNotFoundError(f"Audio file is empty: {path}")

    # Load model
    model = get_whisper_model(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )

    start_time = time.time()
    logger.info(f"Starting ASR transcription: {path}")

    try:
        segments_gen, info = model.transcribe(
            str(path),
            beam_size=beam_size,
            language=language,
            word_timestamps=True,
            vad_filter=True,  # Voice Activity Detection filter to remove silence
        )
    except Exception as e:
        raise TranscriptionError(f"Whisper transcription failed for {path}: {e}") from e

    parsed_segments = []

    try:
        for seg in segments_gen:
            words = []
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    words.append(
                        WordTimestamp(
                            word=w.word,
                            start=round(float(w.start), 3),
                            end=round(float(w.end), 3),
                            probability=round(float(getattr(w, "probability", 1.0)), 4),
                        )
                    )

            parsed_segments.append(
                TranscriptSegment(
                    text=seg.text.strip(),
                    start=round(float(seg.start), 3),
                    end=round(float(seg.end), 3),
                    words=words,
                )
            )
    except Exception as e:
        raise TranscriptionError(f"Error parsing segment stream from Whisper: {e}") from e

    elapsed = round(time.time() - start_time, 2)
    detected_lang = getattr(info, "language", "unknown")
    lang_prob = round(float(getattr(info, "language_probability", 1.0)), 4)
    duration = round(float(getattr(info, "duration", 0.0)), 2)

    logger.info(
        f"ASR complete in {elapsed}s: {len(parsed_segments)} segments, "
        f"lang='{detected_lang}' (prob={lang_prob:.2f}), duration={duration}s"
    )

    return TranscriptionResult(
        audio_path=path,
        segments=parsed_segments,
        language=detected_lang,
        language_probability=lang_prob,
        duration=duration,
        model_name=model_size,
        transcription_duration_seconds=elapsed,
    )
