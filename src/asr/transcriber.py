"""Faster-Whisper inference wrapper for speech transcription with parallel chunking & CPU thread optimization."""

import concurrent.futures
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Union

from .errors import ASRError, AudioNotFoundError, ModelLoadError, TranscriptionError
from .models import TranscriptSegment, TranscriptionResult, WordTimestamp

logger = logging.getLogger(__name__)

# Global model cache to avoid re-loading weights for repeated calls with same parameters
_MODEL_CACHE = {}


def unload_model_cache():
    """Clear cached model instances and explicitly delete C++ model objects to free MKL memory."""
    global _MODEL_CACHE
    keys = list(_MODEL_CACHE.keys())
    for k in keys:
        try:
            m = _MODEL_CACHE.pop(k)
            del m
        except Exception:
            pass
    _MODEL_CACHE.clear()
    import gc
    gc.collect()


def get_whisper_model(
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    cpu_threads: Optional[int] = None,
    num_workers: int = 1,
):
    """Load or retrieve cached Faster-Whisper model instance."""
    if cpu_threads is None:
        cpu_threads = min(8, os.cpu_count() or 4)

    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Lookup any existing loaded model matching (model_size, device, compute_type)
    for (m_size, dev, comp, _, _), model_inst in list(_MODEL_CACHE.items()):
        if m_size == model_size and dev == device and comp == compute_type:
            return model_inst

    cache_key = (model_size, device, compute_type, cpu_threads, num_workers)
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
        f"(device={device}, compute_type={compute_type}, cpu_threads={cpu_threads}, num_workers={num_workers})..."
    )

    try:
        model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
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


def _chunk_worker_fn(args):
    """Worker function executed in parallel thread pool sharing single loaded model instance."""
    (
        chunk_idx,
        chunk_wav,
        chunk_start,
        model,
        language,
        beam_size,
    ) = args

    chunk_segs, info = _transcribe_chunk(
        model=model,
        path=chunk_wav,
        time_offset=chunk_start,
        language=language,
        beam_size=beam_size,
    )

    return chunk_idx, chunk_segs, info


def transcribe_audio_file(
    audio_path: Union[str, Path],
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
    beam_size: int = 5,
    chunk_length_seconds: int = 600,  # 10 minute chunks for memory safety & parallelization
) -> TranscriptionResult:
    """
    Transcribe a local audio file to segment and word-level timestamped text using Faster-Whisper.

    Automatically chunks long audio files (>10 minutes) and transcribes chunks in parallel across CPU cores.

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

    start_time = time.time()
    logger.info(f"Starting ASR transcription: {path} (model={model_size})")

    # Get audio duration using ffprobe or wave
    total_duration = 0.0
    if shutil.which("ffprobe"):
        try:
            import json
            import subprocess

            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(path),
            ]
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
        # If audio is longer than chunk_length_seconds and FFmpeg is available, chunk and parallelize
        if total_duration > chunk_length_seconds and shutil.which("ffmpeg"):
            import subprocess
            import tempfile

            num_chunks = int(total_duration // chunk_length_seconds) + (
                1 if total_duration % chunk_length_seconds > 0 else 0
            )

            total_cpus = os.cpu_count() or 4
            # Use max_workers=2 on CPU to prevent OpenMP/MKL thread oversubscription memory limits
            max_workers = min(2, num_chunks)
            threads_per_worker = max(1, total_cpus // max_workers)

            # Set OpenMP and MKL thread environment variables to prevent heap allocation spikes
            os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
            os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)

            # Load model ONCE into RAM with CTranslate2 parallel decoding worker streams
            model = get_whisper_model(
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=threads_per_worker,
                num_workers=max_workers,
            )

            logger.info(
                f"Audio duration ({total_duration:.1f}s) > {chunk_length_seconds}s. "
                f"Processing {num_chunks} chunks in PARALLEL using shared model instance..."
            )

            with tempfile.TemporaryDirectory(prefix="asr_chunks_") as temp_dir:
                temp_dir_path = Path(temp_dir)

                # 1. Slice audio chunks via FFmpeg
                chunk_tasks = []
                for i in range(num_chunks):
                    chunk_start = i * chunk_length_seconds
                    chunk_wav = temp_dir_path / f"chunk_{i:03d}.wav"

                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        str(chunk_start),
                        "-t",
                        str(chunk_length_seconds),
                        "-i",
                        str(path),
                        "-c:a",
                        "pcm_s16le",
                        str(chunk_wav),
                    ]
                    subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

                    chunk_tasks.append(
                        (
                            i,
                            chunk_wav,
                            chunk_start,
                            model,
                            language,
                            beam_size,
                        )
                    )

                # 2. Transcribe chunks concurrently in parallel ThreadPoolExecutor
                chunk_results = []
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    futures = [
                        executor.submit(_chunk_worker_fn, task)
                        for task in chunk_tasks
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        chunk_results.append(future.result())

                # Sort chunk results by chunk index to maintain strict chronological order
                chunk_results.sort(key=lambda x: x[0])

                for idx, chunk_segs, info in chunk_results:
                    all_segments.extend(chunk_segs)
                    if idx == 0:
                        detected_lang = getattr(info, "language", "unknown")
                        lang_prob = round(
                            float(getattr(info, "language_probability", 1.0)), 4
                        )

        else:
            # Short audio or single chunk: load model & transcribe directly
            total_cpus = os.cpu_count() or 4
            model = get_whisper_model(
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=total_cpus,
            )
            all_segments, info = _transcribe_chunk(
                model, path, time_offset=0.0, language=language, beam_size=beam_size
            )
            detected_lang = getattr(info, "language", "unknown")
            lang_prob = round(float(getattr(info, "language_probability", 1.0)), 4)
            if total_duration == 0.0:
                total_duration = round(float(getattr(info, "duration", 0.0)), 2)

    except Exception as e:
        raise TranscriptionError(
            f"Whisper transcription failed for {path}: {e}"
        ) from e

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


# Alias for backward compatibility
transcribe_audio = transcribe_audio_file
