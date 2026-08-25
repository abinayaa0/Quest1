# Phase 5 — Dialogue Matching Architecture & Design Rationale

---

## 📌 Overview

The **Dialogue Matching Module** (`src/matching/`) maps target spoken dialogue text queries (e.g., `"My mind rebels at stagnation"`) to precise video timestamps ($\text{start\_time}, \text{end\_time}$) using deterministic fuzzy matching over word-level sliding windows.

---

## 💡 Key Design Decisions

### 1. Why RapidFuzz?
* **Lightweight & High-Speed**: C++ implementation (`rapidfuzz`) executing sequence alignment algorithms in microseconds without heavy neural network runtime dependencies.
* **Handles ASR Errors**: Automatic Speech Recognition models frequently make small word substitutions (e.g. Target: `"my mind rebels at stagnation"` $\rightarrow$ ASR: `"my mind rebels its stagnation"`). RapidFuzz fuzzy ratio scoring maintains high similarity ($>85\%$) despite minor phonetic or word substitution errors.
* **Explainable & Deterministic**: Produces exact percentage similarity scores ($0.0 - 100.0\%$) with zero non-deterministic LLM hallucination risk.

---

### 2. Why Word-Level Sliding Windows?
* **Overcomes Segment Boundary Splits**: ASR engines divide transcripts into arbitrary speech segments based on silences. Dialogue often spans across segment boundaries (e.g. Segment 1: `"My mind rebels"`, Segment 2: `"at stagnation"`). Segment-only matching fails in these boundary cases.
* **Precise Sub-Second Localization**: Word-level timestamps allow computing exact start and end timestamps ($\text{words}[0].\text{start} \dots \text{words}[-1].\text{end}$) for frame extraction in Phase 6.

---

### 3. Why NOT Embeddings or Semantic Vector Search?
* **Exact Dialogue Retrieval vs. Semantic Search**: The task is finding where a *specific spoken line of dialogue* occurred in a video. Semantic vector search (e.g. BERT/SentenceTransformers) matches semantic concepts rather than exact words, introducing false positives (e.g. matching `"I hate sitting still"` to `"My mind rebels at stagnation"`).
* **Zero Overhead**: No heavy PyTorch model weights or vector database dependencies required for basic text localization.

---

### 4. Scoring Metric Selection: `rapidfuzz.fuzz.ratio`
* **Ratio vs Token Set Ratio**: `fuzz.ratio` enforces exact word order alignment across the sliding window, which prevents out-of-order word matching false positives. `fuzz.token_set_ratio` can be toggled via parameter when word order in ASR is scrambled.

---

## 🚀 Future Enhancements & Upgrades (Post-V1)

1. **Phonetic Matching (Double Metaphone / Soundex)**: For heavy accent or named entity misspellings (e.g. `"Irene"` vs `"Irreina"`).
2. **Hybrid Embedding Reranking**: Re-rank top-K fuzzy candidate windows using lightweight cross-encoder embeddings for noisy domain-specific transcripts.
3. **Forced Alignment Models**: Use CTC forced aligners (e.g. PyTorch `torchaudio.pipelines.MMS_FA`) to refine sub-word acoustic frame boundaries down to millisecond precision.
