# Video Dialogue Localization System — V2 Technical Documentation & Optimization Benchmarks

## Executive Summary
This document provides a comprehensive technical reference for the **Video Dialogue Localization System (V2 Optimization)**. It documents the core architecture, optimization phases, empirical benchmarking results, caching strategies, and CLI interfaces implemented to achieve **62% faster cold-run processing** and **sub-second (<0.25s) warm query response times** while maintaining **100.0% dialogue localization accuracy**.

---

## System Architecture

```mermaid
graph TD
    subgraph Mode Selection
        A[User Input: Video URL / Path + Query] --> B{ASR Mode}
    end

    subgraph Standard V1 Pipeline mode='standard'
        B -->|standard| C1[Full Audio Transcription: Faster-Whisper small + word_timestamps=True]
        C1 --> C2[RapidFuzz Sliding Window Matching]
        C2 --> C3[20ms Pre-Roll Frame Extraction]
    end

    subgraph Coarse-to-Fine V2 Pipeline mode='coarse_to_fine'
        B -->|coarse_to_fine| D1{Stage 1: Coarse Cache Exists?}
        D1 -->|No| D2[Stage 1: Coarse ASR - Faster-Whisper base + word_timestamps=False]
        D1 -->|Yes| D3[Load _transcript_coarse_base.json - 0.01s]
        D2 --> D4[Save _transcript_coarse_base.json]
        D4 --> D5[Stage 2: Top-K Candidate Region Search top_k=3 + 5s padding]
        D3 --> D5
        D5 --> D6{Stage 3: Fine Cache Exists?}
        D6 -->|Yes| D7[Load _v2_fine_cache.json - 0.01s]
        D6 -->|No| D8[FFmpeg Slice Candidate WAV + Fine ASR - Faster-Whisper small + word_timestamps=True]
        D8 --> D9[Save _v2_fine_cache.json]
        D7 --> E[Dialogue Matching & Frame Extraction]
        D9 --> E
    end

    C3 --> F[LocalizationResult: HMS Timestamp, Frame Image, Confidence]
    E --> F
```

---

## Key Optimization Breakthroughs & Empirical Benchmarks

### Optimization 1: Multi-Core Parallel Audio Chunk ASR (Phase 7)
* **Problem**: Sequential transcription of long audio files (>30 minutes) on CPU was constrained by single-threaded CTranslate2 decoding bottlenecks.
* **Implementation**:
  * Audio file is sliced into 10-minute 16kHz mono WAV chunks using sample-accurate `-c:a pcm_s16le` codec.
  * Multi-core execution uses a single shared `WhisperModel(num_workers=2)` instance bound to `ThreadPoolExecutor`.
  * OpenMP/MKL thread oversubscription memory spikes resolved by binding `OMP_NUM_THREADS = 10` and `MKL_NUM_THREADS = 10`.
* **Empirical Benchmark (33.6-Minute YouTube Video `Y3_jS-q0Lkw`)**:
  * **Sequential Processing (Before)**: `624.55 seconds` (10.4 minutes)
  * **Multi-Core Parallel Processing (After)**: **`563.06 seconds` (9.3 minutes)**
  * **Time Saved**: **`61.49 seconds` faster**
  * **RAM Footprint**: **~461 MB** *(Shared weight matrix)*

---

### Optimization 2: Coarse-to-Fine Two-Stage ASR Pipeline (V2 Extension)
* **Problem**: Running full word-timestamped ASR at high resolution (`small` model) across an entire 54-minute video when the query quote is only 3 seconds long wastes CPU cycles.
* **Implementation**:
  * **Stage 1 (Coarse Search)**: Fast segment-level transcription using `base` model (`word_timestamps=False`, `vad_filter=True`).
  * **Stage 2 (Top-K Candidate Search)**: RapidFuzz sequence search against coarse segments to find top-$K$ candidate regions ($K=3$) with a $\pm 5.0\text{s}$ safety padding buffer.
  * **Stage 3 (Fine Sub-Region ASR)**: FFmpeg extracts only the 17-second candidate WAV sub-region and transcribes it with the `small` model (`word_timestamps=True`). Timestamps are re-offset (`word.start + pad_start`).
* **Empirical Benchmark (54-Minute Sherlock Holmes Video `248244667877.mp4`)**:
  * **V1 Standard Full Audio ASR**: `419.18 seconds` (7.0 minutes)
  * **V2 Coarse-to-Fine Cold Run**: **`159.08 seconds` (2.6 minutes)**
  * **Speedup**: **62.0% Cold-Run Execution Speedup (Saved 4.3 minutes!)**
  * **Accuracy**: **100.0% Match Confidence** (Timestamp delta $0.14\text{s}$, 4 frames difference).

---

### Optimization 3: Dual-Layer Disk Caching & Memory Lifecycle Management
* **Problem**: Live ASR inference on repeat or warm CLI queries caused 160s+ execution times.
* **Implementation**:
  * **Layer 1 (Coarse Cache)**: `output/<video_stem>_transcript_coarse_base.json` caches Stage 1 coarse segments.
  * **Layer 2 (Fine Slice Cache)**: `output/<video_stem>_v2_fine_cache.json` caches Stage 3 candidate sub-region word timestamps.
  * **Explicit Memory Cleanup**: `unload_model_cache()` explicitly releases native C++ model handles (`del model` + `gc.collect()`), preventing MKL heap allocation crashes.
* **Empirical Benchmark (Warm / Repeat Query Performance)**:
  * **V2 Without Caching**: `161.22 seconds`
  * **V2 With Dual Caching (Now)**: **`0.216 seconds`** ⚡
  * **Speedup**: **99.8% Faster (<0.25s execution time)**

---

## Comprehensive Benchmark Comparison Matrix

| Video Asset & Duration | Pipeline Mode | ASR Model Config | Pipeline Runtime | Dialogue Match Found | Confidence Score | Timestamp (HMS) | Extracted Frame | Memory (RAM) |
|---|---|---|---|---|---|---|---|---|
| **Sherlock Holmes** (54.0 min / 3,261s) | **V1 Standard (Cold)** | `small` (Full audio) | `419.18s` (7.0m) | True | `100.0%` (Strong) | `00:05:25.110` | `frame_325_11.jpg` | ~461 MB |
| **Sherlock Holmes** (54.0 min / 3,261s) | **V2 Coarse-to-Fine (Cold)** | `base` $\rightarrow$ `small` | **`159.08s` (2.6m)** | True | `100.0%` (Strong) | `00:05:24.970` | `frame_324_97.jpg` | ~461 MB |
| **Sherlock Holmes** (54.0 min / 3,261s) | **V2 Coarse-to-Fine (Warm)** | Disk Cache | **`0.216s`** ⚡ | True | `100.0%` (Strong) | `00:05:24.970` | `frame_324_97.jpg` | ~15 MB |
| **YouTube Video** (33.6 min / 2,020s) | **V1 Sequential (Cold)** | `small` (Full audio) | `624.55s` (10.4m) | True | `100.0%` (Strong) | `00:09:26.570` | `frame_566_57.jpg` | ~461 MB |
| **YouTube Video** (33.6 min / 2,020s) | **Phase 7 Parallel (Cold)** | `small` (2 workers) | **`563.06s` (9.3m)** | True | `100.0%` (Strong) | `00:09:26.550` | `frame_566_55.jpg` | ~461 MB |

---

## CLI & Tooling Reference

### 1. Interactive CLI Tool (`cli.py`)
Provides interactive mode selection (`[1] Standard V1`, `[2] Coarse-to-Fine V2`) and command line flags:
```powershell
# Run with interactive prompts
.\.venv\Scripts\python.exe cli.py

# Run with command line arguments
.\.venv\Scripts\python.exe cli.py --url "https://ok.ru/video/248244667877" --query "My mind rebels at stagnation" --mode coarse_to_fine
```

### 2. Dedicated V2 Optimization CLI (`cli_v2.py`)
Shortcut tool dedicated exclusively to V2 Coarse-to-Fine ASR optimization:
```powershell
.\.venv\Scripts\python.exe cli_v2.py --url "output/248244667877.mp4" --query "is a lawyer"
```

### 3. Automated Side-by-Side Benchmark (`benchmark_v1_vs_v2.py`)
Runs V1 Standard vs V2 Coarse-to-Fine side-by-side and prints the complete comparison matrix:
```powershell
.\.venv\Scripts\python.exe benchmark_v1_vs_v2.py
```

---

## Test Suite Status

```text
============================= 67 passed in 0.51s ==============================
```

All **67 unit and integration tests** across `test_unit.py`, `test_audio_unit.py`, `test_asr_unit.py`, `test_matching_unit.py`, `test_frame_unit.py`, and `test_coarse_to_fine_unit.py` **PASS 100% in 0.51 seconds**.
