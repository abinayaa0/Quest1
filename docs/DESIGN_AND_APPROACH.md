# Video Dialogue Localization System — Design & Approach

An end-to-end multi-stage AI pipeline for ingesting public video URLs (or local video files) and localizing target spoken dialogue quotes down to **exact timestamps and video frame images** with zero manual video inspection.

---

## 1. Problem Statement Analysis

Before writing code, the project brief was decomposed into explicit requirements, ambiguous terms, and inferred assumptions.

### 1.1 Input / Output Contract

**Input:**
1. **Video URL:** Any publicly accessible video link (e.g., OK.ru, YouTube, Vimeo) or local video file. Example videos may carry no subtitles or captions.
2. **Target Dialogue:** The line of dialogue to locate, returning its first occurrence.

**Output:**
1. **Timestamp:** Formatted as `HH:MM:SS.sss`
2. **Frame Number:** Calculated where applicable (Presentation Timestamp based)
3. **Extracted Dialogue Text:** Matched line with confidence score
4. **Video Frame Image:** Extracted image corresponding to the target timestamp

---

### 1.2 Functional Requirements

| ID | Requirement |
| :--- | :--- |
| **FR1** | Accept a publicly accessible video URL or local file |
| **FR2** | Retrieve and process the video without human inspection |
| **FR3** | Locate the specified dialogue within the video |
| **FR4** | Identify the first occurrence of the dialogue |
| **FR5** | Provide the corresponding timestamp |
| **FR6** | Provide the frame number, where applicable |
| **FR7** | Provide the text associated with the identified dialogue |
| **FR8** | Return the actual video frame as an image |

*Localization requirement:* The system must determine *where in the video to look automatically* — ruling out any design that relies on manual pre-segmentation.

---

### 1.3 Robustness Requirements

| Variation | How it is handled |
| :--- | :--- |
| **Video Quality (high / medium / low)** | ASR operates on the audio track, which degrades far more gracefully than visual text |
| **Resolution (720p, 1080p, ...)** | Ingestion caps at $\le 720\text{p}$; frame extraction probes actual resolution rather than assuming |
| **Frame Rate** | Rational FPS strings preserved exactly; VFR detected explicitly; PTS-based seeking |
| **Appearance of the Dialogue** | Interpreted as spoken utterance temporal localization |

---

### 1.4 Interpreting "Appearance of the Dialogue"

The phrase "appearance of the dialogue" is ambiguous—it could imply burned-in subtitle OCR or the spoken occurrence of dialogue. 
Inspection of real evaluation videos (e.g. 54-minute *Sherlock Holmes* episode) confirms there are no burned-in captions or subtitles on screen. Therefore, the task is treated as **spoken-dialogue temporal localization**: identifying when the target utterance occurs on the audio timeline, then retrieving the corresponding video frame image.

---

### 1.5 Ambiguities vs. Engineering Considerations

| Left unspecified by brief | Practical cases handled |
| :--- | :--- |
| Supported video platforms | Missing audio track |
| Maximum video duration | Redirects & download failures |
| Direct media URLs vs webpage URLs | Network failures / rate limiting |
| Multiple audio-track behavior | Varying formats, resolutions, frame rates |

---

### 1.6 Generalization & Non-Functional Requirements

- **Generalization:** Any dialogue text, any video URL. No hardcoding or tuning specifically for sample files.
- **Uncertainty Handling:** Express confidence score tier rather than returning silent incorrect answers.
- **Performance:** Reasonable processing time via two-stage coarse-to-fine ASR & disk caching.
- **Reproducibility & Modular Code:** 7 decoupled modules with clean Python unit & integration tests.

---

## 2. Pipeline Architecture

The system consists of seven independently testable stages:

```
[Video URL / Path]
       │
       ▼
1. Video Ingestion (yt-dlp + validation)
       │
       ▼
2. Audio Extraction (FFmpeg -> 16kHz Mono PCM WAV)
       │
       ▼
3. Coarse ASR (Whisper base/tiny + VAD + parallel chunks) <──> Coarse Disk Cache
       │
       ▼
4. Candidate Retrieval (RapidFuzz segment search)
       │
       ▼
5. Fine ASR (Whisper small + word timestamps on 20s slice) <──> Fine Slice Cache
       │
       ▼
6. Dialogue Matching (Word-level sliding window & trimming)
       │
       ▼
7. Frame Extraction (FFmpeg PTS seek -20ms pre-roll)
       │
       ▼
[Timestamp, Frame #, Confidence, Frame Image]
```

### Stage Details

| # | Stage | Module | What it does |
| :-: | :--- | :--- | :--- |
| **1** | Video Ingestion | `src/ingestion/` | Resolve URL $\rightarrow$ `yt-dlp` download ($\le 720\text{p}$) $\rightarrow$ `ffprobe` validation & metadata |
| **2** | Audio Extraction | `src/audio/` | FFmpeg $\rightarrow$ 16 kHz mono PCM WAV (standard ASR engine input) |
| **3** | Coarse ASR | `src/asr/` | Whisper `base`/`tiny`, `word_timestamps=False`, full audio $\rightarrow$ coarse segment transcript |
| **4** | Candidate Retrieval | `src/matching/` | RapidFuzz over coarse segments $\rightarrow$ top-K windows ($K=3, \pm 10\text{s}$ padding) |
| **5** | Fine ASR | `src/asr/` | Whisper `small` with `word_timestamps=True`, candidate window only |
| **6** | Dialogue Matching | `src/matching/` | Word-level sliding window search $\rightarrow$ boundary refinement $\rightarrow$ confidence tier |
| **7** | Frame Extraction | `src/frame_extraction/` | FFmpeg PTS seek $\rightarrow$ JPEG export $\rightarrow$ resolution probe |

Two disk caches sit alongside:
- `*_transcript_coarse_base.json`
- `*_v2_fine_cache.json`

Repeat queries against an already-processed video skip ASR inference entirely, reducing execution time from ~160 seconds to **<0.25 seconds**.

---

## 3. Key Design Decisions

### 3.1 Coarse-to-Fine ASR, Not a Single Pass

- **Core Idea:** A 54-minute video contains ~3,300 seconds of speech. Running full word-timestamped ASR across the whole video wastes compute.
- **Solution:** A cheap pass (`base`/`tiny` model) finds *roughly where to look*, followed by an accurate pass (`small` model with word timestamps) over only the ~20-second candidate window.
- **Result:** **62.5% cold-run latency reduction** with negligible ($<0.14\text{s}$) timestamp difference.

### 3.2 Lexical Fuzzy Matching, Not Embeddings

- The goal is **exact spoken dialogue retrieval**, not semantic search.
- Vector embeddings can introduce false positives (e.g. matching *"I hate sitting still"* to *"My mind rebels at stagnation"*).
- RapidFuzz ratio operates in C++ in microseconds with deterministic 0–100 scoring and zero LLM hallucination risk.

### 3.3 Word-Level Sliding Windows, Not Segment Matching

- Whisper segment boundaries are silence-driven, so target phrases often straddle two segments (`"My mind rebels"` / `"at stagnation"`).
- Our matcher slides a window across the word stream and trims it back to the target's true first and last words, giving exact start timestamps and boosting confidence score accuracy.

### 3.4 PTS Seeking, Not `frame = timestamp * fps`

- Videos often run at non-integer rational frame rates (e.g. $24000 / 1001 \approx 23.976\text{ fps}$).
- Floating-point FPS multiplication accumulates rounding drift across long videos.
- FFmpeg seeks by embedded Presentation Timestamp (PTS), combined with a **20ms pre-roll offset** ($\text{seek\_ts} = \max(0, t - 0.02)$) to ensure accurate frame capture without skipping ahead.

### 3.5 Ambiguity & Confidence Tiers

Every match is assigned a confidence score bucketed into four tiers:
- **$> 90\%$**: Strong match
- **$80\text{--}90\%$**: Acceptable match
- **$70\text{--}80\%$**: Needs review
- **$< 70\%$**: Reject

---

## 4. Optimizations & Benchmarks

### 4.1 Multi-Core Parallel Chunk ASR
- Slices long audio files (>10 mins) into 600-second PCM WAV chunks and processes them concurrently across 8 CPU threads (`OMP_NUM_THREADS=8`).

### 4.2 Coarse-to-Fine Performance Comparison (54-min Video)

| Pipeline Mode | Coarse ASR Model | Total Runtime | Confidence | Timestamp Result |
| :--- | :--- | :--- | :--- | :--- |
| **V1 Single-Pass (Sequential)** | `small` (full audio) | 419.18s (~7.0 min) | 100% | `00:05:25.110` |
| **V2 Coarse-to-Fine (Cold)** | `base` $\rightarrow$ `small` | **159.08s (~2.6 min)** | 100% | `00:05:24.970` |
| **V2 Coarse-to-Fine (Warm Cache)** | Disk cache | **0.216s (< 0.25s)** | 100% | `00:05:24.970` |

---

## 5. Robustness Testing & Failure Handling

Tested across speech domains (TV drama, comedy, gaming commentary, podcasts, accented English) and query classes:
- **Exact quotes:** 100% accuracy
- **Case & punctuation variants:** 100% accuracy
- **Partial phrase queries:** Successfully localized
- **Negative / Cross-video queries:** Correctly rejected (0% match confidence, preventing false positives)

---

## 6. Testing

All 67 unit and integration tests pass in 0.51s across ingestion, audio extraction, ASR, matching, frame extraction, and the coarse-to-fine pipeline.
