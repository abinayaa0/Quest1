"""Integration test for Phase 6 Frame Extraction using real OK.ru Sherlock Holmes video."""

from pathlib import Path
import pytest

from frame_extraction import extract_frame

pytestmark = pytest.mark.integration


class TestFrameIntegration:
    VIDEO_PATH = Path("output/248244667877.mp4")

    def test_extract_sherlock_stagnation_frame(self):
        if not self.VIDEO_PATH.exists():
            pytest.skip(f"Video file not found at {self.VIDEO_PATH}")

        # Dialogue timestamp from Phase 5 matching: "My mind rebels at stagnation"
        target_timestamp = 320.48

        print(f"\nExtracting real video frame at timestamp {target_timestamp}s from {self.VIDEO_PATH}...")
        result = extract_frame(
            video_path=self.VIDEO_PATH,
            timestamp=target_timestamp,
            output_dir="output/frames",
        )

        assert result.frame_path.exists()
        assert result.frame_path.stat().st_size > 0
        assert result.width == 960
        assert result.height == 720
        assert result.timestamp == target_timestamp

        print("Frame Extraction Integration Success!")
        print(f"  Frame Path:  {result.frame_path}")
        print(f"  Timestamp:   {result.timestamp}s")
        print(f"  Resolution:  {result.width}x{result.height}")
        print(f"  File Size:   {result.frame_path.stat().st_size} bytes")
        print(f"  Time Taken:  {result.extraction_duration_seconds}s")
