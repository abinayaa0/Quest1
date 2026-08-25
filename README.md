# Video Dialogue Localization System — System Technical Manual

An end-to-end Python pipeline for ingesting public video URLs (YouTube, OK.ru, Vimeo, direct streams) and extracting 16kHz mono PCM WAV speech audio for downstream Automatic Speech Recognition (ASR) and video frame localization.

---

## 📌 System Overview

The system processes public video content across structured pipeline phases to localize target dialogue queries to exact video frames:

```
Public Video URL
       │
       ▼
┌────────────────────────────────────────┐
│ Phase 2: Video Ingestion               │  yt-dlp + FFmpeg + ffprobe
│ - Resolves platform URLs & extractors  │
│ - Selects <=720p A/V streams & merges   │
│ - Validates video/audio & rational FPS │
└──────────────┬─────────────────────────┘
               │
               ▼  Local Video (.mp4)
┌────────────────────────────────────────┐
│ Phase 3: Audio Extraction              │  FFmpeg (-vn -ac 1 -ar 16000)
│ - Extracts speech-ready 16kHz mono WAV │
│ - Validates pcm_s16le audio stream     │
└──────────────┬─────────────────────────┘
               │
               ▼  Speech Audio (.wav, 16kHz mono)
┌────────────────────────────────────────┐
│ Phase 4: ASR & Dialogue Localization   │  (Upcoming Phase)
│ Phase 5: Frame Extraction & Result     │  (Upcoming Phase)
└────────────────────────────────────────┘
```

---

## 🛠️ Module Architecture

### 1. Ingestion Module (`src/ingestion/`)
* **`ingest_video(url, output_dir=None, proxy=None)`**: Public entry point.
* **`downloader.py`**: Invokes `yt-dlp` as a subprocess with format `bestvideo[height<=720]+bestaudio/best`. Includes direct media stream fallback (`ffmpeg -c copy`) for `.mp4`/`.webm`/`.m3u8` links.
* **`probe.py`**: Runs `ffprobe` to validate stream counts (`video >= 1`, `audio >= 1`), duration > 0, captures exact rational frame rates (`"24000/1001"`, `"25/1"`), and detects Variable Frame Rate (VFR) vs Constant Frame Rate (CFR) (`is_vfr: bool`).
* **`models.py`**: Dataclasses `IngestionResult`, `VideoMetadata`, and exception hierarchy (`IngestionError`, `DownloadError`, `ValidationError`, `DependencyError`).

### 2. Audio Module (`src/audio/`)
* **`extract_audio(video_path, output_path=None)`**: Public entry point.
* **`extractor.py`**: Executes `ffmpeg -y -i <video> -vn -ac 1 -ar 16000 -c:a pcm_s16le <wav>` to convert multi-channel audio into a 16kHz 1-channel mono PCM WAV file.
* **`probe.py`**: Runs `ffprobe` to validate `sample_rate == 16000`, `channels == 1`, `duration > 0`, and `codec_name == "pcm_s16le"`.
* **`models.py`**: Dataclasses `AudioResult`, `AudioMetadata`, and `AudioExtractionError`.

---

## 💻 Python API Usage Guide

### Complete End-to-End Ingestion & Audio Extraction Example

```python
import sys
from pathlib import Path

# Add src/ directory to import path
sys.path.insert(0, "src")

from ingestion import ingest_video, IngestionError
from audio import extract_audio, AudioExtractionError

# Target Video URL (Sherlock Holmes on OK.ru)
url = "https://ok.ru/video/248244667877"

try:
    # 1. Ingest Video (Phase 2)
    print("Ingesting video URL...")
    video_res = ingest_video(url, output_dir="output")
    print(f"✅ Video saved to: {video_res.video_path.absolute()}")
    print(f"   Duration:    {video_res.metadata.duration:.1f} seconds")
    print(f"   Resolution:  {video_res.metadata.width}x{video_res.metadata.height}")
    print(f"   FPS String:  {video_res.metadata.fps}")
    print(f"   Video Codec: {video_res.metadata.video_codec}")
    print(f"   Audio Codec: {video_res.metadata.audio_codec}")
    print(f"   Downloaded:  {video_res.download_duration_seconds}s")

    # 2. Extract Speech Audio (Phase 3)
    print("\nExtracting 16kHz mono WAV audio...")
    audio_res = extract_audio(video_res.video_path)
    print(f"✅ Audio saved to: {audio_res.audio_path.absolute()}")
    print(f"   Sample Rate: {audio_res.metadata.sample_rate} Hz")
    print(f"   Channels:    {audio_res.metadata.channels} (mono)")
    print(f"   Codec:       {audio_res.metadata.codec_name}")
    print(f"   File Size:   {audio_res.metadata.file_size_bytes / (1024*1024):.2f} MB")
    print(f"   Extracted:   {audio_res.extraction_duration_seconds}s")

except (IngestionError, AudioExtractionError) as e:
    print(f"❌ Pipeline failed: {e}")
```

---

## 📊 Verification Benchmarks (Supplied OK.ru Video)

### Target URL
`https://ok.ru/video/248244667877` (*The Adventures of Sherlock Holmes: A Scandal in Bohemia*)

### Phase 2 Video Ingestion Results
* **Status**: **PASSED** ✅
* **Video Output Path**: `c:\Quest1\output\248244667877.mp4`
* **File Size**: **1.04 GB** (1,046,151,944 bytes)
* **Duration**: **3,261.8 seconds** (~54 minutes)
* **Resolution**: **960 × 720**
* **Frame Rate**: **24000/1001 FPS** (23.976 FPS)
* **Streams**: 1 Video (`h264`), 1 Audio (`aac`)
* **Download Time**: **133.5 seconds** (2 min 13 sec)

### Phase 3 Audio Extraction Results
* **Status**: **PASSED** ✅
* **Audio Output Path**: `c:\Quest1\output\248244667877.wav`
* **File Size**: **99.54 MB** (104,377,644 bytes)
* **Duration**: **3,261.8 seconds** (~54 minutes)
* **Sample Rate**: **16,000 Hz**
* **Channels**: **1 (mono)**
* **Codec**: `pcm_s16le` (uncompressed PCM WAV)
* **Extraction Time**: **7.87 seconds**

---

## 🛡️ Anti-Bot & Network Troubleshooting (OK.ru)

OK.ru uses VKontakte edge anti-bot protection (`st-ok.cdn-vk.ru`) that drops automated Python socket connections with `ConnectionResetError 10054` when rate limits trigger.

### Solution: Cloudflare WARP or Proxy
1. **Cloudflare WARP (Free 1-Click VPN)**: Enable Cloudflare WARP (`1.1.1.1` in "Private Browsing" mode). This routes traffic through Cloudflare's network, instantly bypassing IP rate limiting.
2. **Explicit Proxy Parameter**: Pass `proxy="http://YOUR_PROXY:PORT"` to `ingest_video()` or set `$env:INGESTION_PROXY = "http://YOUR_PROXY:PORT"`.

---

## 🧪 Test Suite Guide

### 1. Run Unit Tests (37/37 PASSED in 0.13s)
Runs mocked unit tests covering metadata parsing, error handling, ffprobe JSON validation, and proxy parameter forwarding without network calls:

```bash
python -m pytest tests/test_unit.py tests/test_audio_unit.py -v
```

### 2. Run Integration Tests (Real Media)
Runs real network downloads and audio extraction:

```bash
python -m pytest tests/test_integration.py tests/test_audio_integration.py tests/test_direct_media.py -v -m integration
```

---

## 📂 Artifact Inventory

The local `output/` directory contains the complete generated pipeline artifacts:

```text
c:\Quest1\output\
 ├── 248244667877.mp4   (1.04 GB — 54-min 720p H.264 video + AAC audio)
 └── 248244667877.wav   (99.5 MB — 16kHz mono PCM WAV speech audio for ASR)
```

For comprehensive engineering design trade-offs, see **[design.md](file:///c:/Quest1/design.md)**.
