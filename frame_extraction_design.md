# Phase 6 — Frame Extraction Architecture & Design Rationale

---

## 📌 Overview

The **Frame Extraction Module** (`src/frame_extraction/`) takes a video file and a matched dialogue timestamp (e.g. `320.48s`), seeks to that exact point using FFmpeg, extracts the corresponding video frame image (`.jpg`), and probes image resolution metadata (`width`, `height`).

---

## 💡 Key Design Decisions

### 1. Why FFmpeg Timestamp Seeking (`ffmpeg -ss <timestamp> -i <video> -vframes 1`)?
* **Universal Container & Codec Support**: Handles MP4, MKV, WebM, H.264, H.265/HEVC, VP9, and AV1 natively without requiring extra custom decoders.
* **PTS Timestamp-Based Seeking**: FFmpeg seeking relies on Presentation Timestamps (PTS) embedded in the container stream, locating the exact presentation frame corresponding to the dialogue timestamp.
* **CFR & VFR Compatible**: Works identically for Constant Frame Rate (CFR) and Variable Frame Rate (VFR) videos.

---

### 2. Why NOT `frame_number = timestamp × FPS`?
* **VFR (Variable Frame Rate) Drift**: In VFR videos (common in web downloads, screen recordings, HLS streams), frame intervals fluctuate dynamically. Multiplying by a constant average FPS leads to frame drift of several seconds over a 50+ minute video.
* **FPS Rational Fractions**: Many videos use non-integer rational frame rates (e.g. $24000/1001 \approx 23.9760239... \text{ FPS}$). Simple floating-point multiplication accumulates rounding errors.
* **Timestamp as Source of Truth**: Preserving raw floating-point seconds (`timestamp = 320.48`) and letting FFmpeg seek to PTS avoids calculation inaccuracies. If `frame_number` cannot be deterministically proven without a full index scan, `frame_number = None` is returned.

---

## 🚀 Pipeline Data Flow Summary

```
Matched Dialogue Query
          │
          ▼
RapidFuzz Dialogue Matcher
          │  Timestamp (e.g. 320.48s)
          ▼
FFmpeg Timestamp Seeking (`ffmpeg -ss 320.48 -i video.mp4 -vframes 1 frame.jpg`)
          │
          ▼
Frame Metadata Probe (`ffprobe -show_streams frame.jpg`)
          │
          ▼
FrameResult (frame_path="output/frames/frame_320_48.jpg", resolution=960x720)
```
