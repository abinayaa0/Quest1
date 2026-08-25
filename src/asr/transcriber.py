"""Faster-Whisper inference wrapper for speech transcription."""

import logging
import shutil
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


def _transcribe_chunk(
    model,
    path: Path,
    time_offset: float = 0.0,
    language: Optional[str] = None,
    beam_size: int = 5,
):
    """Transcribe a single audio chunk and offset segment & word timestamps."""
    segments_gen, info = model.transcribe(
        str(path),
        beam_size=beam_size,
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )

    parsed_segments = []
    for seg in segments_gen:
        words = []
        if hasattr(seg, "words") and seg.words:
            for w in seg.words:
                words.append(
                    WordTimestamp(
                        word=w.word,
                        start=round(float(w.start) + time_offset, 3),
                        end=round(float(w.end) + time_offset, 3),
                        probability=round(float(getattr(w, "probability", 1.0)), 4),
                    )
                )

        parsed_segments.append(
            TranscriptSegment(
                text=seg.text.strip(),
                start=round(float(seg.start) + time_offset, 3),
                end=round(float(seg.end) + time_offset, 3),
                words=words,
            )
        )

    return parsed_segments, info


def transcribe_audio_file(
    audio_path: Union[str, Path],
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
    beam_size: int = 5,
    chunk_length_seconds: int = 600,  # 10 minute chunks for memory safety
) -> TranscriptionResult:
    """
    Transcribe a local audio file to segment and word-level timestamped text using Faster-Whisper.

    Automatically chunks long audio files (>10 minutes) to prevent NumPy memory errors.

    Args:
        audio_path: Path to the local input audio file (.wav, .mp3, etc.).
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3'). Default 'base'.
        device: Execution device ('cpu' default).
        compute_type: Computation precision ('int8' default for CPU efficiency).
        language: Optional ISO language code (e.g. 'en', 'ru'). If None, automatically detected.
        beam_size: Beam search width (default 5).
        chunk_length_seconds: Audio chunk size in seconds for memory safety (default 600s).

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
    logger.info(f"Starting ASR transcription: {path} (model={model_size})")

    # Get audio duration using ffprobe or wave
    total_duration = 0.0
    if shutil.which("ffprobe"):
        try:
            import json, subprocess
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                data = json.loads(r.stdout)
                total_duration = float(data.get("format", {}).get("duration", 0.0))
        except Exception:
            pass

    all_segments = []
    detected_lang = language or "unknown"
    lang_prob = 1.0

    try:
        # If audio is longer than chunk_length_seconds and FFmpeg is available, chunk it
        if total_duration > chunk_length_seconds and shutil.which("ffmpeg"):
            import subprocess, tempfile
            logger.info(
                f"Audio duration ({total_duration:.1f}s) > {chunk_length_seconds}s. "
                f"Processing in {int(total_duration // chunk_length_seconds) + 1} chunks..."
            )

            with tempfile.TemporaryDirectory(prefix="asr_chunks_") as temp_dir:
                temp_dir_path = Path(temp_dir)
                num_chunks = int(total_duration // chunk_length_seconds) + (
                    1 if total_duration % chunk_length_seconds > 0 else 0
                )

                for i in range(num_chunks):
                    chunk_start = i * chunk_length_seconds
                    chunk_wav = temp_dir_path / f"chunk_{i:03d}.wav"

                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-ss", str(chunk_start),
                        "-t", str(chunk_length_seconds),
                        "-i", str(path), "-c", "copy", str(chunk_wav),
                    ]
                    subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

                    logger.info(
                        f"Transcribing chunk {i+1}/{num_chunks} "
                        f"({chunk_start}s -> {chunk_start + chunk_length_seconds}s)..."
                    )
                    chunk_segs, info = _transcribe_chunk(
                        model, chunk_wav, time_offset=chunk_start,
                        language=language, beam_size=beam_size,
                    )
                    all_segments.extend(chunk_segs)
                    if i == 0:
                        detected_lang = getattr(info, "language", "unknown")
                        lang_prob = round(float(getattr(info, "language_probability", 1.0)), 4)

                    if chunk_wav.exists():
                        chunk_wav.unlink()

        else:
            # Short audio or no FFmpeg chunking: transcribe directly
            all_segments, info = _transcribe_chunk(
                model, path, time_offset=0.0, language=language, beam_size=beam_size
            )
            detected_lang = getattr(info, "language", "unknown")
            lang_prob = round(float(getattr(info, "language_probability", 1.0)), 4)
            if total_duration == 0.0:
                total_duration = round(float(getattr(info, "duration", 0.0)), 2)

    except Exception as e:
        raise TranscriptionError(f"Whisper transcription failed for {path}: {e}") from e

    elapsed = round(time.time() - start_time, 2)

    logger.info(
        f"ASR complete in {elapsed}s: {len(all_segments)} segments, "
        f"lang='{detected_lang}' (prob={lang_prob:.2f}), duration={total_duration:.1f}s"
    )

    return TranscriptionResult(
        audio_path=path,
        segments=all_segments,
        language=detected_lang,
        language_probability=lang_prob,
        duration=total_duration,
        model_name=model_size,
        transcription_duration_seconds=elapsed,
    )
