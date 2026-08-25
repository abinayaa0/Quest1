"""
Integration tests — real downloads over the network.

Run with:  pytest tests/test_integration.py -v -m integration
"""

import logging
import shutil
import tempfile
from pathlib import Path

import pytest

from ingestion import ingest_video, IngestionError
from ingestion.models import DownloadError

logger = logging.getLogger(__name__)

# All tests in this file require network access
pytestmark = pytest.mark.integration


@pytest.fixture()
def output_dir():
    """Temporary directory cleaned up after each test."""
    d = Path(tempfile.mkdtemp(prefix="test_ingestion_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _assert_valid_result(result):
    """Shared assertions for any successful ingestion."""
    assert result.video_path.exists(), f"File missing: {result.video_path}"
    assert result.video_path.stat().st_size > 0, "File is empty"
    assert result.metadata.duration > 0, f"Bad duration: {result.metadata.duration}"
    assert result.metadata.width > 0
    assert result.metadata.height > 0
    assert result.metadata.num_video_streams >= 1
    assert result.metadata.num_audio_streams >= 1
    assert result.metadata.video_codec
    assert result.metadata.audio_codec
    assert result.metadata.fps

    size_mb = result.video_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"  URL:        {result.source_url}\n"
        f"  Method:     {result.ingestion_method}\n"
        f"  File:       {result.video_path}\n"
        f"  Size:       {size_mb:.1f} MB\n"
        f"  Duration:   {result.metadata.duration:.1f}s\n"
        f"  Resolution: {result.metadata.width}x{result.metadata.height}\n"
        f"  FPS:        {result.metadata.fps}\n"
        f"  Video:      {result.metadata.video_codec}\n"
        f"  Audio:      {result.metadata.audio_codec}\n"
        f"  Download:   {result.download_duration_seconds:.1f}s"
    )


# ── OK.ru (primary test URL) ────────────────────────────────────────────────

class TestOKRu:
    URL = "https://ok.ru/video/248244667877"

    def test_full_pipeline(self, output_dir):
        result = ingest_video(self.URL, output_dir=output_dir)
        _assert_valid_result(result)


# ── YouTube ──────────────────────────────────────────────────────────────────

class TestYouTube:
    # Big Buck Bunny trailer — short, public, stable
    URL = "https://www.youtube.com/watch?v=aqz-KE-bpKQ"

    def test_full_pipeline(self, output_dir):
        result = ingest_video(self.URL, output_dir=output_dir)
        _assert_valid_result(result)


# ── Vimeo ────────────────────────────────────────────────────────────────────

class TestVimeo:
    URL = "https://vimeo.com/148751763"

    @pytest.mark.xfail(
        reason="Vimeo requires login for web client access (platform policy). "
               "Our code correctly returns a clear DownloadError.",
        raises=DownloadError,
    )
    def test_full_pipeline(self, output_dir):
        result = ingest_video(self.URL, output_dir=output_dir)
        _assert_valid_result(result)


# ── Invalid / unsupported URLs ───────────────────────────────────────────────

class TestInvalidURLs:
    def test_not_a_url(self):
        with pytest.raises(IngestionError):
            ingest_video("not-a-url")

    def test_nonexistent_video(self, output_dir):
        with pytest.raises((DownloadError, IngestionError)):
            ingest_video(
                "https://ok.ru/video/99999999999999999", output_dir=output_dir
            )
