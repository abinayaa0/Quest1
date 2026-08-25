"""Data models for ASR transcription output."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class WordTimestamp:
    """Word-level timing and confidence metadata."""

    word: str
    start: float           # timestamp in seconds
    end: float             # timestamp in seconds
    probability: float = 1.0


@dataclass
class TranscriptSegment:
    """Segment-level timing, text, and word-level timestamps."""

    text: str
    start: float           # segment start in seconds
    end: float             # segment end in seconds
    words: List[WordTimestamp] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Full transcription result returned by transcribe_audio()."""

    audio_path: Path
    segments: List[TranscriptSegment]
    language: str
    language_probability: float
    duration: float
    model_name: str
    transcription_duration_seconds: float

    @property
    def full_text(self) -> str:
        """Returns concatenated full transcript text."""
        return " ".join(seg.text.strip() for seg in self.segments if seg.text.strip())

    def to_dict(self) -> dict:
        """Convert TranscriptionResult to JSON-serializable dictionary."""
        return {
            "audio_path": str(self.audio_path),
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "model_name": self.model_name,
            "transcription_duration_seconds": self.transcription_duration_seconds,
            "full_text": self.full_text,
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
                for seg in self.segments
            ],
        }

    def save_json(self, output_path: Union[str, Path]) -> Path:
        """Save transcription result as JSON file."""
        import json
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return output_path
