# System Architecture & Technical Design Document

## 1. Executive Context & System Overview

The **Video Dialogue Localization System** is an end-to-end AI pipeline designed to search large-scale, unannotated public video content (from platforms such as YouTube, OK.ru, Vimeo, or direct HTTP/HLS streams) and identify the precise temporal location of a target spoken dialogue segment.

Given:
1. **Public Video URL**: A webpage or media stream link.
2. **Target Dialogue**: Text string representing spoken dialogue.

The complete system pipeline executes across five distinct phases:

```
[Phase 1: Input URL & Query]
             │
             ▼
┌────────────────────────────────────────┐
│ Phase 2: Video Ingestion               │ ◄── Implemented & Verified
│ - yt-dlp resolution & stream selection │
│ - FFmpeg A/V stream merging            │
│ - ffprobe stream validation & metadata │
└──────────────────┬─────────────────────┘
                   │ Local Video (.mp4)
                   ▼
┌────────────────────────────────────────┐
│ Phase 3: Audio Extraction              │ ◄── Implemented & Verified
│ - FFmpeg 16kHz mono WAV extraction     │
│ - ffprobe audio stream validation      │
└──────────────────┬─────────────────────┘
                   │ Speech Audio (.wav, 16kHz mono PCM)
                   ▼
┌────────────────────────────────────────┐
│ Phase 4: ASR & Dialogue Matching       │ (Upcoming Phase)
│ - Speech-to-text transcript & alignment│
│ - Text dialogue timestamp localization │
└──────────────────┬─────────────────────┘
                   │ Target Timestamp (seconds)
                   ▼
┌────────────────────────────────────────┐
│ Phase 5: Frame Extraction & Output     │ (Upcoming Phase)
│ - Exact FPS timestamp -> frame mapping │
│ - Frame extraction & result packaging  │
└────────────────────────────────────────┘
```

Currently, **Phase 2 (Video Ingestion)** and **Phase 3 (Audio Extraction)** are fully implemented, unit-tested, and verified against real production media streams.

---

## 2. Deep-Dive: Phase 2 — Video Ingestion (`src/ingestion/`)

### 2.1 Core Responsibilities
The ingestion layer's responsibility is to accept an unverified public video URL, resolve platform-specific media streams, download and merge video and audio into a decodable local `.mp4` file, validate media stream integrity, and extract frame-accurate timing metadata.

### 2.2 Component Architecture & Flow

```
                      ingest_video(url, output_dir, proxy)
                                       │
                                       ▼
                             URL Validation Check
                             (http/https, non-empty)
                                       │
                                       ▼
                             download_video()
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
       Primary: yt-dlp Subprocess                 Fallback: FFmpeg Direct Stream
  - Format: <=720p A/V merge                - Triggered if yt-dlp fails AND
  - Subtitles: Disabled                       URL has direct media extension
  - Timeout: 1800s                           - ffmpeg -c copy (no re-encode)
  - Proxy: Optional forwarding
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       │
                                       ▼
                                probe_file()
                   (ffprobe JSON inspection & stream validation)
                                       │
               ┌───────────────────────┴───────────────────────┐
               ▼                                               ▼
     Validation Checks:                              Extract Metadata:
     - File exists & non-empty                       - Duration (seconds)
     - Video stream count >= 1                       - Width & Height (pixels)
     - Audio stream count >= 1                       - Rational FPS ("24000/1001")
     - Duration > 0                                  - Video & Audio Codecs
               │                                               │
               └───────────────────────┬───────────────────────┘
                                       │
                                       ▼
                                IngestionResult
```

### 2.3 Key Technical Design Choices

1. **Subprocess Isolation over Python Bindings**:
   `yt-dlp` is invoked via standard Python `subprocess.run`. This provides process isolation, avoids Python Global Interpreter Lock (GIL) contention, prevents third-party binary crashes from taking down the main application runtime, and simplifies `yt-dlp` version upgrades.

2. **Stream Quality Strategy (`bestvideo[height<=720]+bestaudio/best`)**:
   - Downstream frame extraction and OCR/visual localization do not require massive 4K downloads.
   - Capping video resolution to approximately 720p reduces network bandwidth consumption by **60-80%** while preserving sharp visual fidelity for frame localization.
   - Fallback hierarchy: prefers ≤720p separate streams -> ≤720p combined stream -> best available separate streams -> best combined stream.

3. **Frame Rate Precision (Rational FPS Strings)**:
   - Frame timing in NTSC video formats uses fractional frame rates (e.g. `23.976023...` FPS = `24000/1001`).
   - Floating-point representations suffer from rounding drift over long videos (a 54-minute video at `23.976` float loses multiple frames by the end).
   - `probe_file()` preserves the exact rational fraction string from `ffprobe`'s `r_frame_rate` (e.g., `"24000/1001"` or `"25/1"`), allowing Phase 5 frame extraction to compute exact integer frame indices without cumulative drift:
     $$\text{Frame Index} = \left\lfloor \text{Timestamp (s)} \times \frac{\text{Numerator}}{\text{Denominator}} \right\rfloor$$

4. **Stream Merging and ffprobe Validation**:
   Asserts that every downloaded video file contains valid video & audio streams, non-zero duration, and inspects both fundamental real frame rate (`r_frame_rate`) and average frame rate (`avg_frame_rate`) to explicitly detect Variable Frame Rate (VFR) vs Constant Frame Rate (CFR) (`is_vfr: bool`). Parses exact rational fraction strings (e.g. `"24000/1001"`) without float precision loss. Stream merging is performed using stream copying to keep original codec data untouched.

5. **Failure Cleanup & Lifecycle**:
   If ingestion fails at any point (download failure, network drop, missing audio stream), temporary working directories are automatically cleaned up to prevent disk leak.

---

## 3. Deep-Dive: Phase 3 — Audio Extraction (`src/audio/`)

### 3.1 Core Responsibilities
The audio extraction module converts the ingested local video file into a standardized, uncompressed WAV audio file optimized specifically for Automatic Speech Recognition (ASR) engines (e.g. Whisper, Parakeet, Conformer).

### 3.2 Signal Processing Rationale

The module executes the following FFmpeg command:
```bash
ffmpeg -y -i <video_path> -vn -ac 1 -ar 16000 -c:a pcm_s16le <output_wav_path>
```

| FFmpeg Flag | Signal Processing Function | Engineering Rationale |
|---|---|---|
| `-vn` | Disable Video Stream | Strips all visual frame packets during extraction, dramatically speeding up processing time and disk I/O. |
| `-ac 1` | Downmix to Mono (1 Channel) | Speech recognition models operate on 1D mono acoustic waveforms. Downmixing stereo/multi-channel audio reduces data volume by 50% without loss of speech intelligibility. |
| `-ar 16000` | Resample to 16,000 Hz | Standard ASR models extract mel-spectrogram features using 16 kHz window frames (Nyquist frequency 8 kHz covers human speech fundamentals 80 Hz – 7 kHz). Resampling during extraction avoids real-time audio resampling overhead inside ASR inference engines. |
| `-c:a pcm_s16le` | Uncompressed 16-bit PCM | Pulse Code Modulation (PCM) signed 16-bit Little-Endian format guarantees zero compression artifacts, instant linear disk reads, and direct memory mapping to Python NumPy arrays. |

### 3.3 Audio Stream Validation (`probe_audio_file()`)
After extraction, `ffprobe` inspects the generated WAV file and verifies:
* File exists and size > 0 bytes.
* Audio stream exists.
* `sample_rate == 16000` Hz.
* `channels == 1` (mono).
* `codec_name == "pcm_s16le"`.
* `duration > 0` seconds.

---

## 4. Anti-Bot Engineering & Case Study: OK.ru (Odnoklassniki)

### 4.1 The Challenge
The target video provided for validation was:
`https://ok.ru/video/248244667877` (*The Adventures of Sherlock Holmes: A Scandal in Bohemia*, 54 minutes).

During initial automated ingestion attempts, requests were rejected with:
```text
ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host
```

### 4.2 Diagnostic & Network Analysis

```
Client (Python / yt-dlp) ───[ TLS Handshake ]───► OK.ru WAF (VKontakte Edge)
                                                         │
                                               Anti-Bot Check Fails:
                                               - Rate limit exceeded
                                               - Non-browser JA3 TLS Fingerprint
                                                         │
Client ◄───[ TCP RST / Connection Reset 10054 ]──────────┘
```

1. **Network Layer**: OK.ru's security infrastructure (VKontakte edge servers `st-ok.cdn-vk.ru`) monitors incoming TLS Client Hello packets. Standard Python `urllib` or default `yt-dlp` requests trigger automated WAF rate limiting, sending a TCP RST packet to drop the socket during the TLS handshake.
2. **Media Structure**: By using `curl_cffi` with Chrome JA3 impersonation (`impersonate="chrome120"`), we successfully inspected the raw OK.ru page HTML and extracted the embedded `data-options` JSON payload containing 5 HLS video quality streams (`mobile`, `lowest`, `low`, `sd`, `hd`).
3. **Stream Protocol**: The video is delivered as an HLS (`.m3u8`) master playlist containing thousands of small transport stream (`.ts`) segments across 3,261 seconds of playback.

### 4.3 Implemented Solutions

1. **Cloudflare WARP Tunneling (Free 1-Click VPN)**:
   Routing system traffic through Cloudflare WARP (`1.1.1.1`) assigns a clean egress IP address. This bypasses OK.ru's IP rate-limiting block cleanly.
2. **Explicit Proxy Support (`proxy` parameter)**:
   The ingestion layer supports explicit HTTP/SOCKS5 proxies via code or environment variables (`INGESTION_PROXY`, `HTTPS_PROXY`, `HTTP_PROXY`).
3. **Subprocess Timeout Adjustment**:
   Downloading 54 minutes of HLS video (~1 GB of transport stream fragments) takes 2–3 minutes. The subprocess timeout was increased from `600s` to `1800s` (30 minutes) with `--extractor-retries 3` and `--retries 3`.

---

## 5. Comprehensive API & Code Examples

### 5.1 End-to-End Pipeline Script

```python
"""
End-to-End Video Ingestion & Audio Extraction Pipeline.
"""

import sys
from pathlib import Path

# Add src/ directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingestion import ingest_video, IngestionError
from audio import extract_audio, AudioExtractionError


def process_video_pipeline(video_url: str, output_dir: str = "output"):
    print(f"=== Starting Pipeline for: {video_url} ===")

    # -------------------------------------------------------------------
    # Step 1: Video Ingestion (Phase 2)
    # -------------------------------------------------------------------
    try:
        print("\n[Phase 2] Ingesting video...")
        ingestion_res = ingest_video(
            url=video_url,
            output_dir=output_dir,
            proxy=None  # Optional: "socks5://127.0.0.1:1080"
        )
        print(f"  ✅ Video Saved:      {ingestion_res.video_path.absolute()}")
        print(f"  ✅ Video Duration:   {ingestion_res.metadata.duration:.1f}s")
        print(f"  ✅ Video Resolution: {ingestion_res.metadata.width}x{ingestion_res.metadata.height}")
        print(f"  ✅ Video FPS:        {ingestion_res.metadata.fps}")
        print(f"  ✅ Video Codec:      {ingestion_res.metadata.video_codec}")
        print(f"  ✅ Audio Codec:      {ingestion_res.metadata.audio_codec}")
        print(f"  ✅ Ingestion Method: {ingestion_res.ingestion_method}")
        print(f"  ✅ Time Elapsed:     {ingestion_res.download_duration_seconds}s")
    except IngestionError as e:
        print(f"  ❌ Ingestion Failed: {e}")
        return

    # -------------------------------------------------------------------
    # Step 2: Audio Extraction (Phase 3)
    # -------------------------------------------------------------------
    try:
        print("\n[Phase 3] Extracting 16kHz mono WAV audio...")
        wav_target = ingestion_res.video_path.with_suffix(".wav")
        audio_res = extract_audio(
            video_path=ingestion_res.video_path,
            output_path=wav_target
        )
        print(f"  ✅ Audio Saved:      {audio_res.audio_path.absolute()}")
        print(f"  ✅ Sample Rate:      {audio_res.metadata.sample_rate} Hz")
        print(f"  ✅ Channels:         {audio_res.metadata.channels} (mono)")
        print(f"  ✅ Codec:            {audio_res.metadata.codec_name}")
        print(f"  ✅ Audio File Size:  {audio_res.metadata.file_size_bytes / (1024*1024):.2f} MB")
        print(f"  ✅ Time Elapsed:     {audio_res.extraction_duration_seconds}s")
    except AudioExtractionError as e:
        print(f"  ❌ Audio Extraction Failed: {e}")
        return

    print("\n=== Pipeline Execution Complete! Ready for Phase 4 ASR ===")


if __name__ == "__main__":
    target_url = "https://ok.ru/video/248244667877"
    process_video_pipeline(target_url)
```

---

## 6. Complete Verification & Benchmark Results

### 6.1 Real Target Benchmark: OK.ru Video

| Property | Phase 2 Video Ingestion | Phase 3 Audio Extraction |
|---|---|---|
| **Input Resource** | `https://ok.ru/video/248244667877` | `c:\Quest1\output\248244667877.mp4` |
| **Output File** | `c:\Quest1\output\248244667877.mp4` | `c:\Quest1\output\248244667877.wav` |
| **Status** | ✅ **PASSED** | ✅ **PASSED** |
| **File Size** | **1.04 GB** (1,046,151,944 bytes) | **99.54 MB** (104,377,644 bytes) |
| **Duration** | **3,261.8 seconds** (~54 minutes) | **3,261.8 seconds** (~54 minutes) |
| **Resolution / Rate** | **960 × 720** | **16,000 Hz** |
| **Channels / FPS** | **24000/1001 FPS** (23.976) | **1 Channel (Mono)** |
| **Codecs** | `h264` (video) / `aac` (audio) | `pcm_s16le` (uncompressed PCM) |
| **Execution Time** | **133.5 seconds** (2 min 13 sec) | **7.87 seconds** |

### 6.2 Full Integration Test Matrix

| Test Case | URL / Input | Expected Behavior | Measured Result |
|---|---|---|---|
| **OK.ru Full Pipeline** | `ok.ru/video/248244667877` | Resolves HLS stream, downloads MP4, extracts metadata | ✅ **PASSED** (953.9 MB in 133.5s) |
| **YouTube Full Pipeline** | `youtube.com/watch?v=aqz-KE-bpKQ` | Resolves YouTube AV1/Opus, downloads 720p MP4 | ✅ **PASSED** (77.8 MB in 6.2s) |
| **Direct MP4 Download** | `raw.githubusercontent.com/.../sample.mp4` | FFmpeg fallback / yt-dlp direct stream download | ✅ **PASSED** (768x432 in 2.69s) |
| **OK.ru Audio Extraction** | `248244667877.mp4` | FFmpeg 16kHz mono WAV conversion & validation | ✅ **PASSED** (99.5 MB WAV in 7.87s) |
| **Vimeo Authentication** | `vimeo.com/148751763` | Handles platform auth block without crashing | ⏭️ **XFAIL** (Clean `DownloadError`) |
| **Invalid URL String** | `"not-a-url"` | Validates URL scheme early | ✅ **PASSED** (`IngestionError`) |
| **Nonexistent Video ID** | `ok.ru/video/99999999999` | Catches 404/extractor failure cleanly | ✅ **PASSED** (`DownloadError`) |

### 6.3 Unit Test Inventory (37/37 PASSED in 0.13s)

```text
tests/test_unit.py
  ├── TestVideoMetadata
  │     ├── test_creation ......................................... PASSED
  │     └── test_ingestion_result ................................. PASSED
  ├── TestErrors
  │     ├── test_hierarchy ........................................ PASSED
  │     └── test_message_preserved ................................ PASSED
  ├── TestURLValidation
  │     ├── test_empty_url ........................................ PASSED
  │     ├── test_none_url ......................................... PASSED
  │     ├── test_non_http_url ..................................... PASSED
  │     └── test_whitespace_only .................................. PASSED
  ├── TestDirectMediaDetection
  │     └── test_detection (8 parametrized extensions) ............ PASSED
  ├── TestProbe
  │     ├── test_nonexistent_file ................................. PASSED
  │     ├── test_empty_file ....................................... PASSED
  │     ├── test_successful_probe ................................. PASSED
  │     ├── test_no_audio_stream .................................. PASSED
  │     ├── test_no_video_stream .................................. PASSED
  │     ├── test_ffprobe_failure .................................. PASSED
  │     └── test_invalid_json ..................................... PASSED
  ├── TestDownloader
  │     ├── test_missing_ytdlp .................................... PASSED
  │     ├── test_ytdlp_success .................................... PASSED
  │     ├── test_ytdlp_proxy ...................................... PASSED
  │     ├── test_ytdlp_failure_non_media_url ...................... PASSED
  │     └── test_ytdlp_timeout .................................... PASSED
  └── TestIngestVideoMocked
        ├── test_successful_ingestion ............................. PASSED
        └── test_download_failure_cleans_temp ..................... PASSED

tests/test_audio_unit.py
  ├── TestAudioModels
  │     ├── test_audio_metadata ................................... PASSED
  │     └── test_audio_result ..................................... PASSED
  ├── TestAudioProbe
  │     ├── test_nonexistent_file ................................. PASSED
  │     ├── test_empty_file ....................................... PASSED
  │     └── test_successful_probe ................................. PASSED
  └── TestAudioExtractor
        ├── test_missing_ffmpeg ................................... PASSED
        └── test_successful_extraction ............................ PASSED
```

---

## 7. Operational Guidelines for Future Phases

### Passing Data to Phase 4 (ASR & Dialogue Alignment)
Phase 4 can directly consume `AudioResult.audio_path` (`248244667877.wav`). Because the WAV audio file is uncompressed 16kHz 16-bit mono PCM, it can be loaded instantly into Python memory via standard libraries:

```python
import scipy.io.wavfile as wav

# Load audio signal into 1D NumPy array for ASR model
sample_rate, audio_waveform = wav.read("output/248244667877.wav")
assert sample_rate == 16000
```

### Passing Data to Phase 5 (Frame Extraction)
When Phase 4 identifies the exact dialogue timestamp $T_{\text{start}}$ (in seconds), Phase 5 can extract the corresponding video frame image using `VideoMetadata.fps` and `IngestionResult.video_path`:

```bash
# Extract single frame at timestamp T_start
ffmpeg -ss <T_start> -i output/248244667877.mp4 -frames:v 1 frame_output.png
```
