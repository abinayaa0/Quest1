"""Unit tests for Phase 5 Dialogue Matching module."""

import pytest

from matching import (
    EmptyQueryError,
    EmptyTranscriptError,
    MatchResult,
    NoWordsError,
    WordTimestamp,
    generate_windows,
    match_dialogue,
    normalize_text,
)


class TestNormalization:
    def test_lowercase(self):
        assert normalize_text("MY MIND REBELS") == "my mind rebels"

    def test_punctuation_removal(self):
        assert normalize_text("My mind, rebels at stagnation!") == "my mind rebels at stagnation"

    def test_whitespace_normalization(self):
        assert normalize_text("  hello \t  world \n ") == "hello world"

    def test_empty_string(self):
        assert normalize_text("") == ""


class TestWindows:
    def test_generate_windows_timestamps(self):
        words = [
            WordTimestamp("My", 325.11, 325.30),
            WordTimestamp("mind", 325.31, 325.70),
            WordTimestamp("rebels", 325.71, 326.10),
            WordTimestamp("its", 326.11, 326.50),
            WordTimestamp("stagnation", 326.51, 327.69),
        ]
        # Target length 5 words, variance 2 -> windows of lengths 3, 4, 5
        windows = generate_windows(words, target_length=5, length_variance=2)
        assert len(windows) > 0

        # Check 5-word window timestamps
        win_5 = [w for w in windows if len(w.words) == 5][0]
        assert win_5.start_time == 325.11
        assert win_5.end_time == 327.69
        assert win_5.normalized_text == "my mind rebels its stagnation"


class TestMatcherUnit:
    def test_exact_match(self):
        transcript = [
            WordTimestamp("hello", 1.0, 1.5),
            WordTimestamp("world", 1.6, 2.0),
        ]
        res = match_dialogue("hello world", transcript)
        assert res.match_found is True
        assert res.start_time == 1.0
        assert res.end_time == 2.0
        assert res.confidence == 100.0

    def test_asr_error_matching(self):
        """Test ASR typo/substitution: target 'at' vs ASR 'its'."""
        words = [
            WordTimestamp("My", 325.11, 325.30),
            WordTimestamp("mind", 325.31, 325.70),
            WordTimestamp("rebels", 325.71, 326.10),
            WordTimestamp("its", 326.11, 326.50),
            WordTimestamp("stagnation", 326.51, 327.69),
        ]
        res = match_dialogue("My mind rebels at stagnation", words, confidence_threshold=80.0)
        assert res.match_found is True
        assert res.start_time == 325.11
        assert res.end_time == 327.69
        assert res.confidence > 85.0

    def test_boundary_split_dialogue(self):
        """Test dialogue split across two segment boundaries."""
        dict_transcript = {
            "segments": [
                {
                    "text": "My mind rebels",
                    "words": [
                        {"word": "My", "start": 10.0, "end": 10.5},
                        {"word": "mind", "start": 10.6, "end": 11.0},
                        {"word": "rebels", "start": 11.1, "end": 11.5},
                    ],
                },
                {
                    "text": "at stagnation",
                    "words": [
                        {"word": "at", "start": 11.6, "end": 12.0},
                        {"word": "stagnation", "start": 12.1, "end": 13.0},
                    ],
                },
            ]
        }
        res = match_dialogue("My mind rebels at stagnation", dict_transcript)
        assert res.match_found is True
        assert res.start_time == 10.0
        assert res.end_time == 13.0
        assert res.confidence == 100.0

    def test_first_occurrence_rule(self):
        """Verify first occurrence is returned when dialogue is repeated later."""
        words = [
            # First occurrence at 10s
            WordTimestamp("good", 10.0, 10.5),
            WordTimestamp("morning", 10.6, 11.0),
            # Distractor filler
            WordTimestamp("doctor", 100.0, 100.5),
            WordTimestamp("watson", 100.6, 101.0),
            # Second occurrence at 500s
            WordTimestamp("good", 500.0, 500.5),
            WordTimestamp("morning", 500.6, 501.0),
        ]
        res = match_dialogue("Good morning", words)
        assert res.match_found is True
        assert res.start_time == 10.0  # Must return 10.0s (first occurrence), not 500.0s!
        assert res.end_time == 11.0

    def test_match_quality_classification(self):
        m1 = MatchResult(match_found=True, confidence=95.0)
        assert m1.match_quality == "Strong match"

        m2 = MatchResult(match_found=True, confidence=85.0)
        assert m2.match_quality == "Acceptable"

        m3 = MatchResult(match_found=True, confidence=75.0)
        assert m3.match_quality == "Needs review"

        m4 = MatchResult(match_found=False, confidence=65.0)
        assert m4.match_quality == "Reject"

    def test_empty_query_error(self):
        words = [WordTimestamp("test", 1.0, 2.0)]
        with pytest.raises(EmptyQueryError):
            match_dialogue("   ", words)

    def test_empty_transcript_error(self):
        with pytest.raises(EmptyTranscriptError):
            match_dialogue("hello world", [])
