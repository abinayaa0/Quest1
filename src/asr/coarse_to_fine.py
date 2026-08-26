"""
Coarse-to-Fine ASR Pipeline (V2 Optimization Extension)
========================================================
Optimizes ASR runtime on long videos using a two-stage approach:
1. Fast Coarse ASR (base/tiny, word_timestamps=False) -> locate candidate dialogue region
2. Candidate Region Detection -> fuzzy search & add +-5s padding
3. Fine ASR (small, word_timestamps=True) -> transcribe targeted candidate sub-region only
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

from rapidfuzz import fuzz

from .errors import AudioNotFoundError, ModelLoadError, TranscriptionError
from .models import TranscriptSegment, TranscriptionResult, WordTimestamp
from .transcriber import _transcribe_chunk, get_whisper_model

logger = logging.getLogger(__name__)


def detect_top_k_candidate_regions(
    target_query: str,
    coarse_segments: list[TranscriptSegment],
    total_duration: float,
    top_k: int = 3,
    padding_seconds: float = 10.0,
    confidence_threshold: float = 40.0,
) -> list[tuple[float, float, float]]:
    """
    Locate top-K candidate time ranges [(start_pad, end_pad, score)] in coarse transcript segments.
    Uses high-recall multi-metric fuzzy search weighted by target word coverage.
    """
    from matching.normalization import normalize_text

    norm_target = normalize_text(target_query)
    if not norm_target:
        raise TranscriptionError("Target dialogue query for coarse-to-fine ASR is empty")

    q_words = set(norm_target.split())
    num_q_words = max(1, len(q_words))

    scored_segments = []
    for seg in coarse_segments:
        seg_norm = normalize_text(seg.text)
        if not seg_norm:
            continue

        s1 = float(fuzz.partial_ratio(norm_target, seg_norm))
        s2 = float(fuzz.token_set_ratio(norm_target, seg_norm))
        s3 = float(fuzz.WRatio(norm_target, seg_norm))
        similarity = max(s1, s2, s3)

        t_words = set(seg_norm.split())
        common = q_words.intersection(t_words)
        coverage = len(common) / num_q_words

        # Weighted score: 40% similarity + 60% word coverage ratio
        score = (similarity * 0.4) + ((coverage * 100.0) * 0.6)

        if score >= confidence_threshold:
            scored_segments.append((score, seg.start, seg.end))

    # Sort descending by match score, then ascending by start timestamp (First Occurrence)
    scored_segments.sort(key=lambda x: (-x[0], x[1]))

    if not scored_segments:
        raise TranscriptionError(
            f"Coarse ASR failed to locate candidate region for query '{target_query}' "
            f"above confidence threshold {confidence_threshold}%"
        )

    # Take top_k best non-overlapping candidate regions
    candidates = []
    for score, start, end in scored_segments:
        pad_start = max(0.0, start - padding_seconds)
        pad_end = min(total_duration, end + padding_seconds) if total_duration > 0 else (end + padding_seconds)

        # Check for overlap with existing candidate regions
        overlap = False
        for existing_start, existing_end, _ in candidates:
            if not (pad_end < existing_start or pad_start > existing_end):
                overlap = True
                break

        if not overlap:
            candidates.append((pad_start, pad_end, score))
            if len(candidates) >= top_k:
                break

    return candidates


def detect_candidate_region(
    target_query: str,
    coarse_segments: list[TranscriptSegment],
    total_duration: float,
    padding_seconds: float = 5.0,
    confidence_threshold: float = 60.0,
) -> tuple[float, float, float]:
    """Single-candidate fallback wrapper around detect_top_k_candidate_regions."""
    candidates = detect_top_k_candidate_regions(
        target_query=target_query,
        coarse_segments=coarse_segments,
        total_duration=total_duration,
        top_k=1,
        padding_seconds=padding_seconds,
        confidence_threshold=confidence_threshold,
    )
    return candidates[0]


def transcribe_audio_coarse_to_fine(
    audio_path: Union[str, Path],
    target_query: str,
    coarse_model_size: str = "base",
    fine_model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    padding_seconds: float = 5.0,
    top_k: int = 3,
    language: Optional[str] = None,
    beam_size: int = 5,
) -> TranscriptionResult:
    """
    Execute two-stage Coarse-to-Fine ASR optimization pipeline.

    Stage 1: Fast coarse segment-level transcription (word_timestamps=False).
    Stage 2: Candidate region detection & +-5s padding.
    Stage 3: Fine word-timestamped transcription on targeted candidate region (word_timestamps=True).

    Args:
        audio_path: Path to the local input audio file (.wav, .mp3, etc.).
        target_query: Spoken dialogue query string to locate.
        coarse_model_size: Whisper model size for Stage 1 coarse search ('tiny', 'base'). Default 'base'.
        fine_model_size: Whisper model size for Stage 3 fine word-timestamped ASR ('small'). Default 'small'.
        device: Execution device ('cpu' default).
        compute_type: Precision format ('int8' default).
        padding_seconds: Safety buffer in seconds before & after candidate region (default 5.0s).
        language: Optional language code.
        beam_size: Beam search size (default 5).

    Returns:
        TranscriptionResult compatible with standard V1 transcript schema.
    """
    path = Path(audio_path).resolve()
    if not path.exists():
        raise AudioNotFoundError(f"Audio file does not exist: {path}")

    if path.stat().st_size == 0:
        raise AudioNotFoundError(f"Audio file is empty: {path}")

    if not shutil.which("ffmpeg"):
        raise TranscriptionError("FFmpeg is required on PATH for coarse-to-fine audio slicing")

    start_time = time.time()
    logger.info(f"Starting Coarse-to-Fine ASR: {path} (query='{target_query}')")

    # Get total audio duration
    total_duration = 0.0
    if shutil.which("ffprobe"):
        try:
            import json
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                data = json.loads(r.stdout)
                total_duration = float(data.get("format", {}).get("duration", 0.0))
        except Exception:
            pass

    # STAGE 1: Fast Coarse ASR (word_timestamps=False, vad_filter=True)
    coarse_json_path = path.parent / f"{path.stem}_transcript_coarse_{coarse_model_size}.json"
    coarse_segments = []
    detected_lang = language or "unknown"
    lang_prob = 1.0

    if coarse_json_path.exists():
        logger.info(f"[Stage 1/3] Loading cached coarse transcript from disk: {coarse_json_path}")
        import json
        with open(coarse_json_path, "r", encoding="utf-8") as f:
            c_data = json.load(f)
        detected_lang = c_data.get("language", "unknown")
        lang_prob = float(c_data.get("language_probability", 1.0))
        if total_duration == 0.0:
            total_duration = float(c_data.get("duration", 0.0))
        for seg in c_data.get("segments", []):
            coarse_segments.append(
                TranscriptSegment(
                    text=seg.get("text", "").strip(),
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    words=[],
                )
            )
    else:
        logger.info(f"[Stage 1/3] Running coarse ASR ({coarse_model_size}) without word timestamps...")
        coarse_model = get_whisper_model(
            model_size=coarse_model_size,
            device=device,
            compute_type=compute_type,
        )

        chunk_length_seconds = 600

        if total_duration > chunk_length_seconds and shutil.which("ffmpeg"):
            import concurrent.futures
            num_chunks = int(total_duration // chunk_length_seconds) + (
                1 if total_duration % chunk_length_seconds > 0 else 0
            )

            total_cpus = os.cpu_count() or 4
            max_workers = min(2, num_chunks)
            threads_per_worker = max(1, total_cpus // max_workers)

            coarse_model = get_whisper_model(
                model_size=coarse_model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=threads_per_worker,
                num_workers=max_workers,
            )

            with tempfile.TemporaryDirectory(prefix="coarse_chunks_") as c_dir:
                c_dir_path = Path(c_dir)
                chunk_tasks = []

                for i in range(num_chunks):
                    chunk_start = i * chunk_length_seconds
                    chunk_wav = c_dir_path / f"c_chunk_{i:03d}.wav"

                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-ss", str(chunk_start),
                        "-t", str(chunk_length_seconds),
                        "-i", str(path), "-c:a", "pcm_s16le", str(chunk_wav),
                    ]
                    subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
                    chunk_tasks.append((i, chunk_wav, chunk_start))

                def _transcribe_coarse_chunk(args):
                    c_idx, c_wav, c_start = args
                    segs_gen, info = coarse_model.transcribe(
                        str(c_wav),
                        beam_size=beam_size,
                        language=language,
                        word_timestamps=False,
                        vad_filter=True,
                    )
                    c_segs = []
                    for seg in segs_gen:
                        c_segs.append(
                            TranscriptSegment(
                                text=seg.text.strip(),
                                start=round(float(seg.start) + c_start, 3),
                                end=round(float(seg.end) + c_start, 3),
                                words=[],
                            )
                        )
                    return c_idx, c_segs, info

                chunk_results = [None] * num_chunks
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(_transcribe_coarse_chunk, task) for task in chunk_tasks]
                    for future in concurrent.futures.as_completed(futures):
                        c_idx, c_segs, info = future.result()
                        chunk_results[c_idx] = (c_segs, info)

                for i, res in enumerate(chunk_results):
                    if res:
                        c_segs, c_info = res
                        if i == 0:
                            detected_lang = getattr(c_info, "language", "unknown")
                            lang_prob = round(float(getattr(c_info, "language_probability", 1.0)), 4)
                        coarse_segments.extend(c_segs)
        else:
            coarse_segments_gen, coarse_info = coarse_model.transcribe(
                str(path),
                beam_size=beam_size,
                language=language,
                word_timestamps=False,  # Fast segment-level transcription only
                vad_filter=True,
            )

            for seg in coarse_segments_gen:
                coarse_segments.append(
                    TranscriptSegment(
                        text=seg.text.strip(),
                        start=round(float(seg.start), 3),
                        end=round(float(seg.end), 3),
                        words=[],
                    )
                )

            detected_lang = getattr(coarse_info, "language", "unknown")
            lang_prob = round(float(getattr(coarse_info, "language_probability", 1.0)), 4)
            if total_duration == 0.0:
                total_duration = round(float(getattr(coarse_info, "duration", 0.0)), 2)

        # Save coarse transcript to disk for future warm queries
        coarse_res = TranscriptionResult(
            audio_path=path,
            segments=coarse_segments,
            language=detected_lang,
            language_probability=lang_prob,
            duration=total_duration,
            model_name=f"coarse_{coarse_model_size}",
            transcription_duration_seconds=round(time.time() - start_time, 2),
        )
        coarse_res.save_json(coarse_json_path)

        # Unload Stage 1 coarse model from RAM before Stage 3 fine ASR
        if 'coarse_model' in locals() and coarse_model_size != fine_model_size:
            del coarse_model
            from .transcriber import unload_model_cache
            unload_model_cache()

    # STAGE 2: Candidate Region Detection & Padding (Top-K)
    logger.info("[Stage 2/3] Detecting top-K candidate regions for target query...")
    candidates = detect_top_k_candidate_regions(
        target_query=target_query,
        coarse_segments=coarse_segments,
        total_duration=total_duration,
        top_k=top_k,
        padding_seconds=padding_seconds,
    )

    pad_start, pad_end, match_score = candidates[0]
    slice_duration = pad_end - pad_start
    logger.info(f"Primary candidate audio slice: [{pad_start:.2f}s - {pad_end:.2f}s] (duration={slice_duration:.2f}s, score={match_score:.1f}%)")

    # STAGE 3: Fine ASR on Targeted Candidate Region Only (word_timestamps=True)
    fine_cache_path = path.parent / f"{path.stem}_v2_fine_cache.json"
    fine_segments = []
    cached_slice_found = False

    if fine_cache_path.exists():
        try:
            import json
            with open(fine_cache_path, "r", encoding="utf-8") as f:
                fine_cache_data = json.load(f)

            for cached_item in fine_cache_data.get("slices", []):
                c_start = float(cached_item.get("start", -1.0))
                c_end = float(cached_item.get("end", -1.0))
                if abs(c_start - pad_start) <= 1.0 and abs(c_end - pad_end) <= 1.0:
                    logger.info(f"[Stage 3/3] Loading cached fine sub-region from disk: [{c_start:.2f}s - {c_end:.2f}s]")
                    for seg in cached_item.get("segments", []):
                        words = [
                            WordTimestamp(
                                word=w.get("word", ""),
                                start=float(w.get("start", 0.0)),
                                end=float(w.get("end", 0.0)),
                                probability=float(w.get("probability", 1.0)),
                            )
                            for w in seg.get("words", [])
                        ]
                        fine_segments.append(
                            TranscriptSegment(
                                text=seg.get("text", "").strip(),
                                start=float(seg.get("start", 0.0)),
                                end=float(seg.get("end", 0.0)),
                                words=words,
                            )
                        )
                    cached_slice_found = True
                    break
        except Exception as e:
            logger.warning(f"Failed to read fine cache: {e}")

    if not cached_slice_found:
        logger.info(f"[Stage 3/3] Running fine ASR ({fine_model_size}) with word timestamps on candidate slice...")
        fine_model = get_whisper_model(
            model_size=fine_model_size,
            device=device,
            compute_type=compute_type,
        )

        with tempfile.TemporaryDirectory(prefix="coarse_fine_") as tmp_dir:
            candidate_wav = Path(tmp_dir) / "candidate_region.wav"
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-ss", f"{pad_start:.3f}",
                "-t", f"{slice_duration:.3f}",
                "-i", str(path),
                "-c:a", "pcm_s16le",
                str(candidate_wav),
            ]
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

            fine_segments, fine_info = _transcribe_chunk(
                model=fine_model,
                path=candidate_wav,
                time_offset=pad_start,  # Re-offset word timestamps by pad_start
                language=language,
                beam_size=beam_size,
            )

        # Save fine sub-region result to cache file for future instant queries
        try:
            import json
            existing_slices = []
            if fine_cache_path.exists():
                try:
                    with open(fine_cache_path, "r", encoding="utf-8") as f:
                        existing_slices = json.load(f).get("slices", [])
                except Exception:
                    existing_slices = []

            new_slice_entry = {
                "start": pad_start,
                "end": pad_end,
                "segments": [
                    {
                        "text": seg.text,
                        "start": seg.start,
                        "end": seg.end,
                        "words": [
                            {
                                "word": w.word,
                                "start": w.start,
                                "end": w.end,
                                "probability": w.probability,
                            }
                            for w in seg.words
                        ],
                    }
                    for seg in fine_segments
                ],
            }
            existing_slices.append(new_slice_entry)
            with open(fine_cache_path, "w", encoding="utf-8") as f:
                json.dump({"slices": existing_slices}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to write fine cache: {e}")

    elapsed = round(time.time() - start_time, 2)
    logger.info(
        f"Coarse-to-Fine ASR complete in {elapsed}s: "
        f"{len(fine_segments)} fine segments, duration={total_duration:.1f}s"
    )

    return TranscriptionResult(
        audio_path=path,
        segments=fine_segments,
        language=detected_lang,
        language_probability=lang_prob,
        duration=total_duration,
        model_name=f"{coarse_model_size}->{fine_model_size}",
        transcription_duration_seconds=elapsed,
    )
