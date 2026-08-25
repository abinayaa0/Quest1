"""Video download: yt-dlp primary, FFmpeg direct-media fallback."""

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional
from typing import Tuple
from urllib.parse import urlparse

from .models import DownloadError, DependencyError

logger = logging.getLogger(__name__)

# Extensions that indicate a direct media URL (for fallback only)
_DIRECT_MEDIA_EXTENSIONS = frozenset({
    ".mp4", ".webm", ".mkv", ".avi", ".mov", ".flv",
    ".m3u8", ".mpd", ".ts",
})


def _is_direct_media_url(url: str) -> bool:
    """Check if URL path ends with a known media extension."""
    try:
        path = urlparse(url).path.lower()
        return any(path.endswith(ext) for ext in _DIRECT_MEDIA_EXTENSIONS)
    except Exception:
        return False


def _download_with_ytdlp(
    url: str,
    output_dir: Path,
    proxy: Optional[str] = None,
    timeout: int = 1800,
) -> Path:
    """Download video using yt-dlp CLI tool."""
    venv_ytdlp = Path(sys.executable).parent / "yt-dlp.exe"
    ytdlp_bin = shutil.which("yt-dlp") or (str(venv_ytdlp) if venv_ytdlp.exists() else None)

    if not ytdlp_bin:
        raise DependencyError(
            "yt-dlp not found on PATH. Install with: pip install yt-dlp"
        )

    output_template = str(output_dir / "%(id)s.%(ext)s")

    cmd = [
        ytdlp_bin,
        "--no-playlist",
        "-f",
        "bestvideo[height<=720]+bestaudio/best[height<=720]/bestvideo+bestaudio/best",
        "--no-write-subs",
        "--no-write-auto-subs",
        "--retries", "3",
        "--extractor-retries", "3",
        "--socket-timeout", "30",
        "--print", "after_move:filepath",
        "-o", output_template,
    ]

    if proxy:
        cmd.extend(["--proxy", proxy])

    cmd.append(url)

    logger.info(f"Downloading with yt-dlp: {url} (proxy={proxy})")
    logger.debug(f"yt-dlp command: {cmd}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise DownloadError(f"yt-dlp timed out after {timeout}s for: {url}")
    except FileNotFoundError:
        raise DependencyError("yt-dlp executable not found")

    if result.returncode != 0:
        stderr = result.stderr.strip()
        logger.debug(f"yt-dlp stderr:\n{stderr}")
        raise DownloadError(
            f"yt-dlp failed (exit {result.returncode}): "
            f"{stderr[-500:] if len(stderr) > 500 else stderr}"
        )

    # Parse output path from --print after_move:filepath
    stdout_lines = [
        line.strip() for line in result.stdout.strip().split("\n") if line.strip()
    ]

    if stdout_lines:
        filepath = Path(stdout_lines[-1])
        if filepath.exists():
            logger.info(f"yt-dlp output: {filepath} ({filepath.stat().st_size} bytes)")
            return filepath

    # Fallback: scan output directory for most recently modified file
    candidates = sorted(
        output_dir.glob("*.*"), key=lambda f: f.stat().st_mtime, reverse=True
    )
    if candidates:
        logger.warning(
            f"Could not parse yt-dlp output path from stdout, using: {candidates[0]}"
        )
        return candidates[0]

    raise DownloadError(
        f"yt-dlp completed but no output file found.\n"
        f"stdout: {result.stdout[:300]}\nstderr: {result.stderr[:300]}"
    )


def _download_with_ffmpeg(
    url: str, output_dir: Path, timeout: int = 1800, proxy: str | None = None
) -> Path:
    """Download a direct media URL using FFmpeg stream copy (no re-encoding)."""
    if not shutil.which("ffmpeg"):
        raise DependencyError(
            "ffmpeg not found on PATH. Install FFmpeg: https://ffmpeg.org/download.html"
        )

    output_path = output_dir / "direct_download.mp4"

    cmd = ["ffmpeg", "-y"]

    if proxy:
        cmd.extend(["-http_proxy", proxy])

    cmd.extend([
        "-i", url,
        "-c", "copy",
        str(output_path),
    ])

    logger.info(f"Downloading with FFmpeg: {url}")
    logger.debug(f"FFmpeg command: {cmd}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        if output_path.exists():
            output_path.unlink()
        raise DownloadError(f"FFmpeg timed out after {timeout}s for: {url}")

    if result.returncode != 0:
        if output_path.exists():
            output_path.unlink()
        raise DownloadError(
            f"FFmpeg download failed (exit {result.returncode}): "
            f"{result.stderr.strip()[-500:]}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise DownloadError("FFmpeg produced no output or an empty file")

    logger.info(f"FFmpeg output: {output_path} ({output_path.stat().st_size} bytes)")
    return output_path


def download_video(
    url: str, output_dir: Path, proxy: str | None = None
) -> Tuple[Path, str]:
    """
    Download video from URL.

    Strategy:
        1. Try yt-dlp (handles platform URLs, format selection, stream merging).
        2. If yt-dlp fails AND URL looks like direct media, try FFmpeg stream copy.

    Returns:
        (output_file_path, method_used)

    Raises:
        DownloadError: All download methods failed.
        DependencyError: Required tool not installed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Primary: yt-dlp
    try:
        path = _download_with_ytdlp(url, output_dir, proxy=proxy)
        return path, "yt-dlp"
    except DependencyError:
        raise  # Don't mask missing tools
    except DownloadError as ytdlp_error:
        logger.warning(f"yt-dlp failed: {ytdlp_error}")

        # Fallback: FFmpeg for direct media URLs only
        if _is_direct_media_url(url):
            logger.info("URL appears to be direct media, attempting FFmpeg fallback")
            try:
                path = _download_with_ffmpeg(url, output_dir, proxy=proxy)
                return path, "direct-ffmpeg"
            except (DownloadError, DependencyError) as ffmpeg_error:
                raise DownloadError(
                    f"All download methods failed for: {url}\n"
                    f"  yt-dlp: {ytdlp_error}\n"
                    f"  FFmpeg: {ffmpeg_error}"
                ) from ffmpeg_error

        # Not a direct media URL — re-raise the yt-dlp error
        raise
