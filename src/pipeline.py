"""
Unified Pipeline Entry Point — Video Dialogue Localization
================================────────────────────────────
Executes the full pipeline:
Ingest Video -> Extract Audio -> Transcribe Speech -> Match Dialogue -> Extract Frame
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from asr import transcribe_audio
from audio import extract_audio
from frame_extraction import extract_frame
from ingestion import ingest_video
from matching import match_dialogue

logger = logging.getLogger(__name__)


def format_timestamp_hms(seconds: Optional[float]) -> str:
    """Format floating point seconds to HH:MM:SS.sss string format."""
    if seconds is None or seconds < 0:
        return "00:00:00.000"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


@dataclass
class LocalizationResult:
    """Final output result of Video Dialogue Localization."""

    video_path: Path
    dialogue_query: str
    match_found: bool
    timestamp: Optional[float]
    frame_number: Optional[int]
    extracted_dialogue_text: Optional[str]
    frame_image_path: Optional[Path]
    confidence: float
    width: int
    height: int
    pipeline_duration_seconds: float = 0.0

    @property
    def timestamp_hms(self) -> str:
        """Returns timestamp in HH:MM:SS.sss format."""
        return format_timestamp_hms(self.timestamp)

    @property
    def confidence_quality(self) -> str:
        """Classify confidence score into human-interpretable quality tier."""
        if self.confidence > 90.0:
            return "Strong match"
        elif self.confidence >= 80.0:
            return "Acceptable"
        elif self.confidence >= 70.0:
            return "Needs review"
        else:
            return "Reject"

    def display(self):
        """Format and print the required minimum output fields."""
        print("\n" + "=" * 65)
        print("         VIDEO DIALOGUE LOCALIZATION OUTPUT RESULT")
        print("=" * 65)
        print(f" * Dialogue Query:           \"{self.dialogue_query}\"")
        print(f" * Match Found:              {self.match_found}")
        if self.match_found:
            print(f" * Timestamp of Frame (HMS): {self.timestamp_hms}")
            print(f" * Timestamp (Seconds):      {self.timestamp:.2f}s")
            frame_num_str = f"{self.frame_number}" if self.frame_number is not None else "N/A (VFR)"
            print(f" * Frame Number:             {frame_num_str}")
            print(f" * Extracted Dialogue Text:  \"{self.extracted_dialogue_text}\"")
            print(f" * Confidence Score:         {self.confidence:.1f}% ({self.confidence_quality})")
            print(f" * Corresponding Frame Image:{self.frame_image_path}")
            print(f" * Frame Resolution:         {self.width} x {self.height}")
        print(f" * Pipeline Execution Time:  {self.pipeline_duration_seconds:.3f} seconds")
        print("=" * 65 + "\n")

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "video_path": str(self.video_path),
            "dialogue_query": self.dialogue_query,
            "match_found": self.match_found,
            "timestamp": self.timestamp,
            "timestamp_hms": self.timestamp_hms,
            "frame_number": self.frame_number,
            "extracted_dialogue_text": self.extracted_dialogue_text,
            "frame_image_path": str(self.frame_image_path) if self.frame_image_path else None,
            "confidence": round(self.confidence, 2),
            "confidence_quality": self.confidence_quality,
            "width": self.width,
            "height": self.height,
            "pipeline_duration_seconds": round(self.pipeline_duration_seconds, 3),
        }


def localize_dialogue(
    video_url_or_path: Union[str, Path],
    dialogue_query: str,
    output_dir: Union[str, Path] = "output",
    model_size: str = "small",
    confidence_threshold: float = 75.0,
    mode: str = "standard",
    coarse_model_size: str = "base",
    fine_model_size: str = "small",
) -> LocalizationResult:
    """
    Run end-to-end video dialogue localization pipeline.

    Args:
        video_url_or_path: Public video URL (e.g. OK.ru, YouTube) or local video file path.
        dialogue_query: Spoken dialogue text to locate.
        output_dir: Directory for storing output artifacts.
        model_size: Faster-Whisper ASR model size ('tiny', 'base', 'small', 'medium').
        confidence_threshold: RapidFuzz minimum match score (default 75.0).
        mode: ASR pipeline mode ('standard' for V1 default, 'coarse_to_fine' for V2 optimization).
        coarse_model_size: Coarse Whisper model for Stage 1 ('tiny', 'base').
        fine_model_size: Fine Whisper model for Stage 3 ('small', 'base').

    Returns:
        LocalizationResult object containing timestamp, frame_number, extracted text, frame image path.
    """
    import time
    start_t = time.time()

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_str = str(video_url_or_path)

    # 1. Video Ingestion
    if input_str.startswith(("http://", "https://")):
        logger.info(f"Ingesting video from URL: {input_str}")
        ingest_res = ingest_video(input_str, output_dir=out_dir)
        video_path = ingest_res.video_path
        fps = ingest_res.metadata.fps
    else:
        video_path = Path(video_url_or_path).resolve()
        fps = 23.976  # Default fall-back FPS estimate

    # 2. Audio Extraction
    wav_path = out_dir / f"{video_path.stem}.wav"
    if not wav_path.exists():
        logger.info(f"Extracting 16kHz mono WAV audio from {video_path}...")
        audio_res = extract_audio(video_path, output_path=wav_path)
        wav_path = audio_res.audio_path

    # 3. ASR Speech Recognition
    if mode == "coarse_to_fine":
        logger.info(f"Transcribing audio with Coarse-to-Fine ASR mode for '{dialogue_query}'...")
        asr_res = transcribe_audio(
            wav_path,
            mode="coarse_to_fine",
            target_query=dialogue_query,
            coarse_model_size=coarse_model_size,
            fine_model_size=fine_model_size,
            device="cpu",
            compute_type="int8",
        )
        transcript_obj = asr_res
    else:
        transcript_path = out_dir / f"{video_path.stem}_transcript_{model_size}.json"
        if not transcript_path.exists():
            logger.info(f"Transcribing audio with Faster-Whisper ({model_size})...")
            asr_res = transcribe_audio(wav_path, model_size=model_size, device="cpu", compute_type="int8")
            transcript_path = asr_res.save_json(transcript_path)
        transcript_obj = transcript_path

    # 4. Dialogue Matching
    logger.info(f"Matching dialogue query: '{dialogue_query}'...")
    match_res = match_dialogue(
        target_text=dialogue_query,
        transcript=transcript_obj,
        confidence_threshold=confidence_threshold,
    )

    elapsed_t = round(time.time() - start_t, 3)

    if not match_res.match_found:
        return LocalizationResult(
            video_path=video_path,
            dialogue_query=dialogue_query,
            match_found=False,
            timestamp=None,
            frame_number=None,
            extracted_dialogue_text=None,
            frame_image_path=None,
            confidence=match_res.confidence,
            width=0,
            height=0,
            pipeline_duration_seconds=elapsed_t,
        )

    # Calculate frame number estimate if fps is available
    timestamp = match_res.start_time
    fps_val = 23.976
    if isinstance(fps, (int, float)):
        fps_val = float(fps)
    elif isinstance(fps, str) and "/" in fps:
        try:
            num, denom = fps.split("/")
            fps_val = float(num) / float(denom)
        except Exception:
            fps_val = 23.976

    frame_number = int(round(timestamp * fps_val)) if fps_val > 0 else None

    # 5. Frame Extraction
    frames_dir = out_dir / "frames"
    frame_res = extract_frame(
        video_path=video_path,
        timestamp=timestamp,
        output_dir=frames_dir,
    )
    frame_res.frame_number = frame_number

    elapsed_t = round(time.time() - start_t, 3)

    return LocalizationResult(
        video_path=video_path,
        dialogue_query=dialogue_query,
        match_found=True,
        timestamp=timestamp,
        frame_number=frame_number,
        extracted_dialogue_text=match_res.matched_window_raw_text,
        frame_image_path=frame_res.frame_path,
        confidence=match_res.confidence,
        width=frame_res.width,
        height=frame_res.height,
        pipeline_duration_seconds=elapsed_t,
    )
