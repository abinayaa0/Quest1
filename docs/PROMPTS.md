# Coding Execution Prompts & Development History

This document consolidates all implementation prompts, research queries, and development-phase prompts used for building the Video Dialogue Localization System.

---

## 📌 Coding Execution Prompts (Consolidated)

> **Note:** This section consolidates the implementation prompts and development-phase prompts used/planned for generating code, organized in development order.

### Phase 0 — Project Setup and Codebase Understanding

**Prompt:**
You are working on a Video Dialogue Localization System.
Understand the existing Python codebase before making changes.

**Goal:**
Given a video URL/file and a target dialogue phrase, locate the exact timestamp where the dialogue occurs and extract the corresponding video frame.

**Existing pipeline:**
Video -> Download/Input handling -> FFmpeg audio extraction -> Faster-Whisper ASR -> Dialogue matching -> Frame extraction

Do not rewrite existing working logic.
First analyze:
- project structure
- current pipeline flow
- ASR module
- matching module
- frame extraction module
- output format

Provide a summary before making modifications.

---

### Phase 1 — V1 Pipeline Implementation

**Prompt:**
Implement the baseline V1 video dialogue localization pipeline.

**Requirements:**
- **Input:** Video URL or local video path, Target dialogue phrase
- **Pipeline:**
  1. Download video if URL is provided.
  2. Extract audio using FFmpeg (mono audio, 16kHz sampling rate, PCM WAV output).
  3. Run Faster-Whisper small model (`word_timestamps=True`, `vad_filter=True`).
  4. Match target dialogue against transcript.
  5. Extract the exact frame corresponding to the matched timestamp.
- **Output:** Return matched dialogue, timestamp seconds, timestamp HMS, frame number, extracted frame image, confidence score.

Keep modules separated and maintainable.

---

### Phase 2 — Matching and Frame Extraction

**Prompt:**
Implement dialogue matching and timestamp localization.

**Requirements:**
- Use RapidFuzz based matching.
- Normalize text before matching.
- Preserve word timestamps.
- Convert matched word timestamps into the exact dialogue timestamp.

**Frame extraction:**
- Use FFmpeg for frame extraction. Do not use OpenCV.
- Command should support accurate timestamp seeking: `ffmpeg -ss <timestamp> -vframes 1`

---

### Phase 3 — V2 Coarse-to-Fine ASR Extension

**Prompt:**
Implement an optional Coarse-to-Fine ASR pipeline as V2.

**Important:** Do not modify or break V1. V1 must remain available as `mode="standard"`. Add `mode="coarse_to_fine"`.

**Architecture:**
- **Stage 1 (Coarse transcription):** Fast coarse transcription using Whisper tiny/base, `word_timestamps=False`, `vad_filter=True`. Output: segment timestamps only.
- **Stage 2 (Candidate retrieval):** Use existing RapidFuzz matching, search coarse transcript segments, identify candidate time window, add timestamp padding.
- **Stage 3 (Fine transcription):** Only process candidate audio region using Faster Whisper small, `word_timestamps=True`.

Output format must remain compatible with V1.

---

### Phase 4 — CPU Optimization

**Prompt:**
Optimize CPU inference performance without changing pipeline behavior.

**Requirements:**
- Enable multi-threading where safe.
- Benchmark different CPU thread counts.
- Do not reduce transcription accuracy.
- Keep model outputs compatible.

**Measure:** Audio extraction time, coarse ASR time, fine ASR time, matching time, frame extraction time, total execution time.

---

### Phase 5 — Model Benchmarking

**Prompt:**
Create a benchmark comparing coarse ASR models.

**Compare:**
1. Whisper tiny as coarse model
2. Whisper base as coarse model

Use the same videos and queries.

**Measure:**
- **Cold run:** Download, extraction, coarse ASR, candidate retrieval, fine ASR, total time.
- **Warm run:** Cached transcript lookup, candidate retrieval, fine ASR latency.
- **Evaluate:** Timestamp accuracy, confidence score, successful localization, failure cases.

---

### Phase 6 — Caching System

**Prompt:**
Implement caching for repeated video queries.

**Goal:** Avoid rerunning expensive ASR stages.

**Cache:** Downloaded videos, extracted audio, coarse transcripts, fine transcript regions, localization results.

**Warm query flow:**
1. Load cached transcript
2. Retrieve candidate
3. Run fine ASR only if required
4. Return result quickly

Do not change V1 behavior.

---

### Phase 7 — Profiling and Logging

**Prompt:**
Add detailed execution profiling.

**Record:** Video download time, audio extraction time, coarse ASR time, candidate retrieval time, fine ASR time, matching time, frame extraction time, total runtime.

Store execution history including model used, query, video information, timestamp result, and confidence score.

---

### Phase 8 — Failure Handling

**Prompt:**
Improve robustness.

**Handle:**
1. **Query not found:** Return `{"found": false, "reason": "dialogue not detected"}`
2. **Multiple occurrences:** Define deterministic behavior (first occurrence or highest confidence match). Document the chosen rule.
3. **Invalid videos, invalid URLs, empty queries, corrupted files.**

---

### Phase 9 — Evaluation Framework

**Prompt:**
Create a structured evaluation framework.

**Test categories:** Exact phrases, long sentences, short phrases, proper nouns, ASR error cases, noisy audio, multiple occurrences, negative queries, empty queries.

**Record:** Found/not found, timestamp error, confidence, runtime, extracted frame correctness.

---

### Phase 10 — Streamlit UI

**Prompt:**
Create a Streamlit UI for the Video Dialogue Localization System.

**Inputs:** Video URL, Video file upload, Dialogue query.

**Display:** Timestamp HMS, timestamp seconds, frame number, extracted dialogue, confidence score, frame image, pipeline execution time.

**Add:** Processing status, history view, download result/history files.

---

### Phase 11 — FastAPI Wrapper (Optional)

**Prompt:**
Create a thin FastAPI layer around the existing pipeline without modifying pipeline logic.

**Endpoints:**
- `GET /health`
- `POST /localize` (accepts `{video_source, dialogue_query}`, returns structured localization response).

Keep AI logic inside existing pipeline modules.

---

### Phase 12 — Deployment Preparation

**Prompt:**
Prepare the application for deployment.

**Requirements:**
- Dockerize application.
- Include FFmpeg dependency.
- Include Python dependencies.
- Maintain Faster-Whisper CPU compatibility.
- Provide startup instructions.

**Deployment goal:** User opens application, uploads/provides video, enters dialogue, receives frame result.

---

### Final Development Goal

A complete system:
```
User -> Streamlit UI -> Video Dialogue Localization Pipeline -> FFmpeg -> Coarse Whisper -> Candidate Retrieval -> Fine Whisper -> RapidFuzz Matching -> FFmpeg Frame Extraction -> Timestamp + Frame Output
```

Priorities:
1. Correct localization
2. Reliable failure handling
3. Reproducible evaluation
4. Practical CPU performance
5. Deployable architecture

---

## 🔍 Previous Research & Development Prompts

The following prompts record the sequential exploration, requirement analysis, and architecture design discussions during early development:

1. *"okay so frist divide this into different phases like this need not be an AI problem statement it could be anthing"*
2. *"This is the given problem statement: what is an on-screen dialogue"*
3. *"from what i observer from the video given: I think on screen dialogue just means that the character speaks it on screen cuz i dont find much textual displayed and there is no caption available on the screen."*
4. *"okay lets push video analysis to version 2 now."*
5. *"Find out the models/open weights / repositories and everything else that do this exact problem statement and related to this. the best models that perform this as the pipeline that has been designed for this. do a full sweep research. we can use AI/ML tools, libraries, APIs, or locally hosted models as part of your solution. proven working solutions. dont focus on OCR models as we do not need it right now. target solving the core PS"*
6. *"we dont need diarization we just need to recognize the time stamps and frames"*
7. *"forced alignment?"*
8. *"soo this problem statement is just essentially audio transcription and matching with that right"*
9. *"we dont need to explicitly map the frames of the video with the audio timestamp wise we just have to maintain the audio timestamp and get frames by the FPS we are using."*
10. *"google search kind of implements this how does it do that."*
11. *"what is the video understanding that we want here?"*
12. *"V1 (Baseline): Download/Extract audio -> Transcribe with Whisper -> Search Text with RapidFuzz -> Map to Frame with FFmpeg PTS seek."*
13. *"are there any existing github repos / projects that do this pipeline really well"*
14. *"dont we need to also determine how from a URL we go to the video and audio and then convert that how would we do that"*
15. *"see im not going to pick one particular model for one thing unless there is some solid unit testing that has happened right"*
16. *"Convert to writing block"*
17. *"soo in the V1 there is no use of video?"*
18. *"125.2 sec × 24 FPS → frame ≈ 3005 → extract frame. is this calculation accurate?"*
19. *"Okay i want to do this phase wise cuz its all soo confusing"*
20. *"but we need to document something called as an approach. Like how we are coming up with the solution and how we are approaching the problem statement. what architecture/pipeline we are using here and everything. soo thats really important so how do I do that"*
21. *"v2 is not required imo"*
22. *"give the above answer as a PDF document"*
23. *"how did we come up with this architecture?"*
24. *"let us structure this properly, i want you to scope out the requirements of this problem statement, as in what the input is, what the output is, etc."*
25. *"i dont understand the 'where applicable'"*
26. *"Appearance of the dialogue. The PS says: 'the appearance of the dialogue'. Given our current interpretation that this is spoken dialogue, we should be careful here: the wording is somewhat ambiguous. The PS does not explicitly define what 'appearance of the dialogue' means. So we should not invent a specific requirement such as subtitle appearance or visual appearance. We can document this as an ambiguity to investigate. what do you think it means?"*
27. *"is there any architecture that is soo robust it could do both properly?"*
28. *"but 'how will it determine where to look in the video' is a clue by itself right"*
