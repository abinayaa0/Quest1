"""
Integration test for direct media URL fallback.
"""

import tempfile
from pathlib import Path
import pytest

from ingestion import ingest_video

pytestmark = pytest.mark.integration

class TestDirectMedia:
    URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4"

    def test_direct_mp4(self):
        with tempfile.TemporaryDirectory() as td:
            result = ingest_video(self.URL, output_dir=td)
            assert result.video_path.exists()
            assert result.video_path.stat().st_size > 0
            assert result.metadata.duration > 0
            assert result.metadata.width > 0
            assert result.metadata.height > 0
            assert result.metadata.num_video_streams >= 1
            assert result.metadata.num_audio_streams >= 1
            print(f"\nDirect MP4 Ingestion Success!")
            print(f"Path: {result.video_path}")
            print(f"Method: {result.ingestion_method}")
            print(f"Duration: {result.metadata.duration}s")
            print(f"Resolution: {result.metadata.width}x{result.metadata.height}")
            print(f"Codecs: {result.metadata.video_codec} / {result.metadata.audio_codec}")
