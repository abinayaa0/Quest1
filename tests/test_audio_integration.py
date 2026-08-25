"""Integration test for Phase 3 Audio Extraction against real downloaded video."""

import tempfile
from pathlib import Path
import pytest

from audio import extract_audio

pytestmark = pytest.mark.integration


class TestAudioExtractionIntegration:
    VIDEO_PATH = Path("output/248244667877.mp4")

    def test_extract_audio_from_okru_video(self):
        if not self.VIDEO_PATH.exists():
            pytest.skip(f"Downloaded video not found at {self.VIDEO_PATH}")

        with tempfile.TemporaryDirectory() as td:
            out_wav = Path(td) / "okru_audio.wav"
            result = extract_audio(self.VIDEO_PATH, output_path=out_wav)

            assert result.audio_path.exists()
            assert result.audio_path.stat().st_size > 0
            assert result.metadata.sample_rate == 16000
            assert result.metadata.channels == 1
            assert result.metadata.duration > 0
            assert "pcm" in result.metadata.codec_name.lower()

            print("\nReal Audio Extraction Successful!")
            print(f"  Source Video: {result.source_video_path}")
            print(f"  Audio Path:   {result.audio_path}")
            print(f"  Duration:     {result.metadata.duration:.1f}s")
            print(f"  Sample Rate:  {result.metadata.sample_rate} Hz")
            print(f"  Channels:     {result.metadata.channels} (mono)")
            print(f"  Codec:        {result.metadata.codec_name}")
            print(f"  Size:         {result.metadata.file_size_bytes / (1024*1024):.2f} MB")
            print(f"  Extracted in: {result.extraction_duration_seconds}s")
