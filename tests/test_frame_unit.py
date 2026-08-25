"""Unit tests for Phase 6 Frame Extraction module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from frame_extraction import (
    FFmpegFrameError,
    FrameExtractionError,
    FrameResult,
    InvalidTimestampError,
    InvalidVideoError,
    extract_frame,
)


class TestFrameErrors:
    def test_nonexistent_video(self):
        with pytest.raises(InvalidVideoError, match="does not exist"):
            extract_frame("/nonexistent/video.mp4", timestamp=10.0)

    def test_empty_video(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            pass
        with pytest.raises(InvalidVideoError, match="empty"):
            extract_frame(f.name, timestamp=10.0)

    def test_negative_timestamp(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            f.flush()
            with pytest.raises(InvalidTimestampError, match="negative"):
                extract_frame(f.name, timestamp=-5.0)


class TestFrameExtractorMocked:
    @patch("subprocess.run")
    def test_successful_mocked_extraction(self, mock_run):
        # Mock FFmpeg and ffprobe subprocess calls
        ffmpeg_res = MagicMock(returncode=0, stdout="", stderr="")
        ffprobe_res = MagicMock(
            returncode=0,
            stdout='{"streams": [{"codec_type": "video", "width": 960, "height": 720}]}',
            stderr="",
        )
        mock_run.side_effect = [ffmpeg_res, ffprobe_res]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "test.mp4"
            video_file.write_bytes(b"dummy video content")

            out_dir = tmp_path / "frames"

            # Create mock output frame file that FFmpeg would generate
            expected_frame = out_dir / "frame_320_48.jpg"

            # Side effect function to create frame file when subprocess.run is called for FFmpeg
            def fake_run(cmd, *args, **kwargs):
                if "ffmpeg" in cmd[0]:
                    expected_frame.parent.mkdir(parents=True, exist_ok=True)
                    expected_frame.write_bytes(b"fake jpeg data")
                    return ffmpeg_res
                return ffprobe_res

            mock_run.side_effect = fake_run

            res = extract_frame(video_file, timestamp=320.48, output_dir=out_dir)

            assert res.frame_path.exists()
            assert res.timestamp == 320.48
            assert res.width == 960
            assert res.height == 720
            assert res.frame_number is None

            # Verify FFmpeg command flags
            ffmpeg_cmd = mock_run.call_args_list[0][0][0]
            assert "ffmpeg" in ffmpeg_cmd[0]
            assert "-ss" in ffmpeg_cmd
            assert "320.48" in ffmpeg_cmd
            assert "-vframes" in ffmpeg_cmd
            assert "1" in ffmpeg_cmd
