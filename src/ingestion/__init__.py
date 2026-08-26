"""
Video Ingestion Module
======================

Download and validate video from public URLs.

Usage::

    from ingestion import ingest_video

    result = ingest_video("https://ok.ru/video/248244667877")
    print(result.video_path)
    print(result.metadata.duration)
"""

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

from .models import (
    IngestionError,
    IngestionResult,
    VideoMetadata,
    DownloadError,
    DependencyError,
    ValidationError,
)
from .downloader import download_video
from .probe import probe_file

logger = logging.getLogger(__name__)

__all__ = [
    "ingest_video",
    "IngestionResult",
    "VideoMetadata",
    "IngestionError",
    "DownloadError",
    "ValidationError",
    "DependencyError",
]


import os

def ingest_video(
    url: str,
    output_dir: Optional[Union[str, Path]] = None,
    proxy: Optional[str] = None,
) -> IngestionResult:
    """
    Ingest a video from a public URL.

    Downloads the video, validates it has video + audio streams,
    and extracts metadata.

    Args:
        url: Public video URL (platform page or direct media link).
        output_dir: Directory for the downloaded file.
                    Creates a temp directory if not provided.
        proxy: Optional proxy/VPN URL (e.g., 'http://127.0.0.1:8080' or
               'socks5://127.0.0.1:1080'). If not provided, checks env vars
               INGESTION_PROXY, HTTPS_PROXY, and HTTP_PROXY.

    Returns:
        IngestionResult with the local video path and metadata.

    Raises:
        IngestionError: URL validation failed.
        DownloadError: Video could not be downloaded.
        ValidationError: Downloaded file is missing video/audio streams.
        DependencyError: yt-dlp or FFmpeg/ffprobe not installed.
    """
    # --- URL validation ---
    if not url or not isinstance(url, str):
        raise IngestionError("URL must be a non-empty string")

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise IngestionError(f"URL must start with http:// or https://: {url}")

    # Resolve proxy from argument or environment variables
    if not proxy:
        proxy = (
            os.environ.get("INGESTION_PROXY")
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("HTTP_PROXY")
        )

    # --- Output directory ---
    created_temp = False
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="video_ingestion_"))
        created_temp = True
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Ingestion started: {url}")
    if proxy:
        logger.info(f"Using proxy: {proxy}")
    logger.info(f"Output directory: {output_dir}")

    start_time = time.time()

    try:
        # Check if video ID already exists locally in output directory to avoid network failures
        video_id = url.rstrip("/").split("/")[-1]
        local_existing = output_dir / f"{video_id}.mp4"
        if local_existing.exists() and local_existing.stat().st_size > 1024:
            logger.info(f"Reusing existing downloaded video: {local_existing}")
            video_path = local_existing
            method = "cached-disk"
        else:
            # Step 1: Download
            video_path, method = download_video(url, output_dir, proxy=proxy)
        
        elapsed = time.time() - start_time
        logger.info(f"Download complete ({method}) in {elapsed:.1f}s: {video_path}")

        # Step 2: Validate + extract metadata
        metadata = probe_file(video_path)

        logger.info(
            f"Ingestion successful: {metadata.width}x{metadata.height}, "
            f"{metadata.duration:.1f}s, method={method}"
        )

        return IngestionResult(
            video_path=video_path,
            source_url=url,
            metadata=metadata,
            ingestion_method=method,
            download_duration_seconds=round(elapsed, 2),
        )

    except Exception:
        # Clean up temp directory on failure (not user-supplied dirs)
        if created_temp and output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temp directory: {output_dir}")
        raise
