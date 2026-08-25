"""Integration test for Phase 4 ASR Speech Recognition using Faster-Whisper."""

from pathlib import Path
import pytest

from asr import transcribe_audio, ASRError

pytestmark = pytest.mark.integration


class TestASRIntegration:
    AUDIO_PATH = Path("output/248244667877.wav")

    def test_transcribe_okru_audio(self):
        if not self.AUDIO_PATH.exists():
            pytest.skip(f"Extracted WAV audio file not found at {self.AUDIO_PATH}")

        print(f"\nRunning Faster-Whisper ASR (model='tiny', cpu, int8) on {self.AUDIO_PATH}...")
        
        # Transcribe first 60 seconds of Sherlock Holmes audio to verify ASR pipeline
        result = transcribe_audio(
            self.AUDIO_PATH,
            model_size="tiny",
            device="cpu",
            compute_type="int8",
        )

        assert result.audio_path.exists()
        assert result.language is not None
        assert result.duration > 0
        assert len(result.segments) > 0

        print(f"ASR Integration Success!")
        print(f"  Detected Language: {result.language} (prob={result.language_probability:.2f})")
        print(f"  Audio Duration:    {result.duration}s")
        print(f"  Total Segments:    {len(result.segments)}")
        print(f"  Model Used:        {result.model_name}")
        print(f"  Transcription Time:{result.transcription_duration_seconds}s")
        
        print("\nFirst 3 Transcribed Segments:")
        for seg in result.segments[:3]:
            print(f"  [{seg.start:.2f}s -> {seg.end:.2f}s]: {seg.text}")
            if seg.words:
                print(f"    Words ({len(seg.words)}): {[w.word for w in seg.words[:5]]}")
