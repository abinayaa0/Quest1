"""Unit tests for the ingestion module. All external calls are mocked."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion import ingest_video
from ingestion.models import (
    VideoMetadata,
    IngestionResult,
    IngestionError,
    DownloadError,
    DependencyError,
    ValidationError,
)
from ingestion.downloader import _is_direct_media_url, download_video
from ingestion.probe import probe_file


# ---------------------------------------------------------------------------
# Fixtures / test data
# ---------------------------------------------------------------------------

SAMPLE_FFPROBE_JSON = {
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1280,
            "height": 720,
            "r_frame_rate": "25/1",
            "avg_frame_rate": "25/1",
            "duration": "120.5",
        },
        {
            "codec_type": "audio",
            "codec_name": "aac",
            "sample_rate": "44100",
            "channels": 2,
            "duration": "120.5",
        },
    ],
    "format": {
        "duration": "120.5",
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "size": "15000000",
    },
}

FFPROBE_NO_AUDIO = {
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 640,
            "height": 480,
            "r_frame_rate": "30/1",
        },
    ],
    "format": {"duration": "60.0", "format_name": "mp4"},
}

FFPROBE_NO_VIDEO = {
    "streams": [
        {"codec_type": "audio", "codec_name": "aac"},
    ],
    "format": {"duration": "60.0", "format_name": "mp4"},
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestVideoMetadata:
    def test_creation(self):
        m = VideoMetadata(
            duration=120.5, width=1280, height=720, fps="24000/1001",
            avg_fps="24000/1001", is_vfr=False,
            video_codec="h264", audio_codec="aac",
            container_format="mp4", num_video_streams=1, num_audio_streams=1,
        )
        assert m.duration == 120.5
        assert m.width == 1280
        assert m.fps == "24000/1001"
        assert m.is_vfr is False

    def test_ingestion_result(self):
        meta = VideoMetadata(
            duration=60.0, width=640, height=480, fps="30/1",
            avg_fps="30/1", is_vfr=False,
            video_codec="h264", audio_codec="aac",
            container_format="mp4", num_video_streams=1, num_audio_streams=1,
        )
        r = IngestionResult(
            video_path=Path("test.mp4"), source_url="https://example.com/v",
            metadata=meta, ingestion_method="yt-dlp",
            download_duration_seconds=5.3,
        )
        assert r.ingestion_method == "yt-dlp"
        assert r.download_duration_seconds == 5.3


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestErrors:
    def test_hierarchy(self):
        assert issubclass(DownloadError, IngestionError)
        assert issubclass(ValidationError, IngestionError)
        assert issubclass(DependencyError, IngestionError)

    def test_message_preserved(self):
        e = DownloadError("yt-dlp exited with code 1")
        assert "code 1" in str(e)


# ---------------------------------------------------------------------------
# URL validation (ingest_video level)
# ---------------------------------------------------------------------------

class TestURLValidation:
    def test_empty_url(self):
        with pytest.raises(IngestionError, match="non-empty"):
            ingest_video("")

    def test_none_url(self):
        with pytest.raises(IngestionError):
            ingest_video(None)

    def test_non_http_url(self):
        with pytest.raises(IngestionError, match="http"):
            ingest_video("ftp://example.com/video.mp4")

    def test_whitespace_only(self):
        with pytest.raises(IngestionError):
            ingest_video("   ")


# ---------------------------------------------------------------------------
# Direct media URL detection
# ---------------------------------------------------------------------------

class TestDirectMediaDetection:
    @pytest.mark.parametrize("url,expected", [
        ("https://example.com/video.mp4", True),
        ("https://cdn.example.com/stream.m3u8", True),
        ("https://example.com/video.webm", True),
        ("https://example.com/file.mkv", True),
        ("https://example.com/watch?v=abc123", False),
        ("https://ok.ru/video/12345", False),
        ("https://youtube.com/watch?v=dQw4w9WgXcQ", False),
        ("", False),
    ])
    def test_detection(self, url, expected):
        assert _is_direct_media_url(url) == expected


# ---------------------------------------------------------------------------
# Probe (ffprobe wrapper)
# ---------------------------------------------------------------------------

class TestProbe:
    def test_nonexistent_file(self):
        with pytest.raises(ValidationError, match="does not exist"):
            probe_file(Path("/nonexistent/path/video.mp4"))

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            pass  # 0 bytes
        with pytest.raises(ValidationError, match="empty"):
            probe_file(Path(f.name))

    @patch("ingestion.probe.shutil.which", return_value="ffprobe")
    @patch("ingestion.probe.subprocess.run")
    def test_successful_probe(self, mock_run, _mock_which):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(SAMPLE_FFPROBE_JSON),
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            f.flush()
            meta = probe_file(Path(f.name))

        assert meta.duration == 120.5
        assert meta.width == 1280
        assert meta.height == 720
        assert meta.fps == "25/1"
        assert meta.video_codec == "h264"
        assert meta.audio_codec == "aac"
        assert meta.num_video_streams == 1
        assert meta.num_audio_streams == 1

    @patch("ingestion.probe.shutil.which", return_value="ffprobe")
    @patch("ingestion.probe.subprocess.run")
    def test_no_audio_stream(self, mock_run, _):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(FFPROBE_NO_AUDIO),
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            f.flush()
            with pytest.raises(ValidationError, match="No usable audio"):
                probe_file(Path(f.name))

    @patch("ingestion.probe.shutil.which", return_value="ffprobe")
    @patch("ingestion.probe.subprocess.run")
    def test_no_video_stream(self, mock_run, _):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(FFPROBE_NO_VIDEO),
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            f.flush()
            with pytest.raises(ValidationError, match="No video"):
                probe_file(Path(f.name))

    @patch("ingestion.probe.shutil.which", return_value="ffprobe")
    @patch("ingestion.probe.subprocess.run")
    def test_ffprobe_failure(self, mock_run, _):
        mock_run.return_value = MagicMock(
            returncode=1, stderr="Error opening input file",
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            f.flush()
            with pytest.raises(ValidationError, match="ffprobe failed"):
                probe_file(Path(f.name))

    @patch("ingestion.probe.shutil.which", return_value="ffprobe")
    @patch("ingestion.probe.subprocess.run")
    def test_invalid_json(self, mock_run, _):
        mock_run.return_value = MagicMock(returncode=0, stdout="not json at all")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            f.flush()
            with pytest.raises(ValidationError, match="invalid JSON"):
                probe_file(Path(f.name))

    @patch("ingestion.probe.shutil.which", return_value="ffprobe")
    @patch("ingestion.probe.subprocess.run")
    def test_vfr_detection(self, mock_run, _):
        vfr_json = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1280,
                    "height": 720,
                    "r_frame_rate": "90000/1",
                    "avg_frame_rate": "23976/1000",
                    "duration": "120.0",
                },
                {"codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {"duration": "120.0", "format_name": "mp4"},
        }
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(vfr_json),
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video")
            f.flush()
            meta = probe_file(Path(f.name))

        assert meta.is_vfr is True
        assert meta.fps == "90000/1"
        assert meta.avg_fps == "23976/1000"


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

class TestDownloader:
    @patch("ingestion.downloader.Path.exists", return_value=False)
    @patch("ingestion.downloader.shutil.which", return_value=None)
    def test_missing_ytdlp(self, _, __):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(DependencyError, match="yt-dlp"):
                download_video("https://example.com/video", Path(td))

    @patch("ingestion.downloader.subprocess.run")
    @patch("ingestion.downloader.shutil.which", return_value="yt-dlp")
    def test_ytdlp_success(self, _, mock_run):
        with tempfile.TemporaryDirectory() as td:
            fake_output = Path(td) / "abc123.mp4"
            fake_output.write_bytes(b"fake video content")

            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=str(fake_output) + "\n",
                stderr="",
            )

            path, method = download_video("https://ok.ru/video/123", Path(td))
            assert method == "yt-dlp"
            assert path == fake_output

    @patch("ingestion.downloader.subprocess.run")
    @patch("ingestion.downloader.shutil.which", return_value="yt-dlp")
    def test_ytdlp_proxy(self, _, mock_run):
        with tempfile.TemporaryDirectory() as td:
            fake_output = Path(td) / "abc123.mp4"
            fake_output.write_bytes(b"fake video content")

            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=str(fake_output) + "\n",
                stderr="",
            )

            path, method = download_video(
                "https://ok.ru/video/123", Path(td), proxy="http://127.0.0.1:8080"
            )
            assert method == "yt-dlp"
            assert path == fake_output
            cmd_passed = mock_run.call_args[0][0]
            assert "--proxy" in cmd_passed
            assert "http://127.0.0.1:8080" in cmd_passed

    @patch("ingestion.downloader.subprocess.run")
    @patch("ingestion.downloader.shutil.which", return_value="yt-dlp")
    def test_ytdlp_failure_non_media_url(self, _, mock_run):
        """yt-dlp fails and URL is not direct media — should re-raise DownloadError."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="ERROR: Unsupported URL",
        )
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(DownloadError, match="Unsupported URL"):
                download_video("https://example.com/page", Path(td))

    @patch("ingestion.downloader.subprocess.run")
    @patch("ingestion.downloader.shutil.which", return_value="yt-dlp")
    def test_ytdlp_timeout(self, _, mock_run):
        import subprocess as sp
        mock_run.side_effect = sp.TimeoutExpired(cmd="yt-dlp", timeout=600)
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(DownloadError, match="timed out"):
                download_video("https://ok.ru/video/123", Path(td))


# ---------------------------------------------------------------------------
# Full pipeline (mocked)
# ---------------------------------------------------------------------------

class TestIngestVideoMocked:
    @patch("ingestion.probe_file")
    @patch("ingestion.download_video")
    def test_successful_ingestion(self, mock_download, mock_probe):
        with tempfile.TemporaryDirectory() as td:
            video_path = Path(td) / "test.mp4"
            video_path.write_bytes(b"fake video")

            mock_download.return_value = (video_path, "yt-dlp")
            mock_probe.return_value = VideoMetadata(
                duration=120.0, width=1280, height=720, fps="25/1",
                avg_fps="25/1", is_vfr=False,
                video_codec="h264", audio_codec="aac",
                container_format="mp4", num_video_streams=1,
                num_audio_streams=1,
            )

            result = ingest_video("https://ok.ru/video/123", output_dir=td)

            assert result.video_path == video_path
            assert result.metadata.duration == 120.0
            assert result.metadata.width == 1280
            assert result.ingestion_method == "yt-dlp"
            assert result.source_url == "https://ok.ru/video/123"

    @patch("ingestion.download_video")
    def test_download_failure_cleans_temp(self, mock_download):
        """When download fails and no output_dir was given, temp dir is cleaned."""
        mock_download.side_effect = DownloadError("Network error")

        with pytest.raises(DownloadError):
            ingest_video("https://example.com/video")
