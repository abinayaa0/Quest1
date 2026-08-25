"""Dialogue matching implementation using RapidFuzz and word-level sliding windows."""

import logging
from typing import Any, List, Optional, Union

from rapidfuzz import fuzz

from .errors import EmptyQueryError, EmptyTranscriptError, NoWordsError
from .models import MatchResult, WordTimestamp, WordWindow
from .normalization import normalize_text

logger = logging.getLogger(__name__)


def extract_words_from_transcript(transcript: Any) -> List[WordTimestamp]:
    """
    Extract a flat chronological list of WordTimestamp objects from various transcript formats.
    Supports TranscriptionResult dataclass, dictionary, or direct list of words.
    """
    if not transcript:
        raise EmptyTranscriptError("Transcript input is empty or None")

    words: List[WordTimestamp] = []

    # Case 1: TranscriptionResult dataclass from asr module
    if hasattr(transcript, "segments"):
        for seg in transcript.segments:
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    word_str = str(getattr(w, "word", "")).strip()
                    if word_str:
                        words.append(
                            WordTimestamp(
                                word=word_str,
                                start=float(getattr(w, "start", 0.0)),
                                end=float(getattr(w, "end", 0.0)),
                                probability=float(getattr(w, "probability", 1.0)),
                            )
                        )

    # Case 2: Dictionary format (e.g. loaded from JSON)
    elif isinstance(transcript, dict):
        segments = transcript.get("segments", [])
        for seg in segments:
            seg_words = seg.get("words", [])
            for w in seg_words:
                word_str = str(w.get("word", "")).strip()
                if word_str:
                    words.append(
                        WordTimestamp(
                            word=word_str,
                            start=float(w.get("start", 0.0)),
                            end=float(w.get("end", 0.0)),
                            probability=float(w.get("probability", 1.0)),
                        )
                    )

    # Case 3: List of WordTimestamp objects or dicts directly
    elif isinstance(transcript, list):
        for item in transcript:
            if isinstance(item, WordTimestamp):
                words.append(item)
            elif isinstance(item, dict):
                word_str = str(item.get("word", "")).strip()
                if word_str:
                    words.append(
                        WordTimestamp(
                            word=word_str,
                            start=float(item.get("start", 0.0)),
                            end=float(item.get("end", 0.0)),
                            probability=float(item.get("probability", 1.0)),
                        )
                    )
            elif hasattr(item, "word"):
                words.append(
                    WordTimestamp(
                        word=str(getattr(item, "word", "")).strip(),
                        start=float(getattr(item, "start", 0.0)),
                        end=float(getattr(item, "end", 0.0)),
                    )
                )

    if not words:
        raise NoWordsError("No word-level timestamps found in transcript")

    # Sort strictly by start timestamp chronologically
    words.sort(key=lambda w: w.start)
    return words


def generate_windows(
    words: List[WordTimestamp],
    target_length: int,
    length_variance: int = 2,
) -> List[WordWindow]:
    """
    Generate sliding windows of consecutive words around target length.

    Args:
        words: Chronologically sorted list of WordTimestamp objects.
        target_length: Number of words in target dialogue query.
        length_variance: Allowed window size variance (default +-2 words).

    Returns:
        List of WordWindow objects ordered chronologically.
    """
    if not words or target_length <= 0:
        return []

    min_win = max(1, target_length - length_variance)
    max_win = target_length + length_variance

    windows: List[WordWindow] = []

    # Generate windows starting at each word index
    for i in range(len(words)):
        for win_size in range(min_win, max_win + 1):
            if i + win_size <= len(words):
                win_words = words[i : i + win_size]
                raw_text = " ".join(w.word for w in win_words)
                norm_text = normalize_text(raw_text)

                if norm_text:
                    windows.append(
                        WordWindow(
                            words=win_words,
                            raw_text=raw_text,
                            normalized_text=norm_text,
                            start_time=win_words[0].start,
                            end_time=win_words[-1].end,
                        )
                    )

    # Sort strictly by window start time chronologically
    windows.sort(key=lambda win: (win.start_time, win.end_time))
    return windows


def match_dialogue(
    target_text: str,
    transcript: Any,
    confidence_threshold: float = 80.0,
    length_variance: int = 2,
    scorer_name: str = "ratio",
) -> MatchResult:
    """
    Locate the FIRST occurrence of a target dialogue in transcript using RapidFuzz sliding windows.

    Args:
        target_text: The target spoken dialogue string to find.
        transcript: TranscriptionResult, transcript dict, or list of WordTimestamp objects.
        confidence_threshold: Minimum RapidFuzz similarity score (0-100) to accept a match (default 80.0).
        length_variance: Allowed word count variation (+- N words around target length).
        scorer_name: RapidFuzz similarity metric ('ratio' or 'token_set_ratio').

    Returns:
        MatchResult object with match status, timestamps, and confidence score.

    Raises:
        EmptyQueryError: Target query is empty or whitespace-only.
        EmptyTranscriptError: Transcript is empty or None.
        NoWordsError: No word timestamps exist in transcript.
    """
    norm_target = normalize_text(target_text)
    if not norm_target:
        raise EmptyQueryError("Target dialogue query text is empty")

    words = extract_words_from_transcript(transcript)
    target_word_count = len(norm_target.split())

    windows = generate_windows(
        words=words,
        target_length=target_word_count,
        length_variance=length_variance,
    )

    if not windows:
        return MatchResult(match_found=False, confidence=0.0)

    # Select RapidFuzz scoring function
    scorer_fn = fuzz.ratio if scorer_name == "ratio" else fuzz.token_set_ratio

    best_score_overall = 0.0
    best_window_overall: Optional[WordWindow] = None

    # CHRONOLOGICAL SEARCH: Evaluate windows in strict chronological order
    for win in windows:
        score = float(scorer_fn(norm_target, win.normalized_text))

        if score > best_score_overall:
            best_score_overall = score
            best_window_overall = win

        # FIRST OCCURRENCE RULE: Immediately return the first window meeting/exceeding threshold
        if score >= confidence_threshold:
            logger.info(
                f"Match found chronologically at [{win.start_time:.2f}s - {win.end_time:.2f}s] "
                f"with confidence {score:.1f}% >= threshold {confidence_threshold}%"
            )
            return MatchResult(
                match_found=True,
                matched_text=win.normalized_text,
                start_time=win.start_time,
                end_time=win.end_time,
                confidence=score,
                matched_window_raw_text=win.raw_text,
            )

    # No window met confidence_threshold
    logger.info(
        f"No match exceeded threshold {confidence_threshold}%. Best score was {best_score_overall:.1f}%"
    )
    return MatchResult(
        match_found=False,
        confidence=best_score_overall,
        matched_text=best_window_overall.normalized_text if best_window_overall else None,
        start_time=best_window_overall.start_time if best_window_overall else None,
        end_time=best_window_overall.end_time if best_window_overall else None,
        matched_window_raw_text=best_window_overall.raw_text if best_window_overall else None,
    )
