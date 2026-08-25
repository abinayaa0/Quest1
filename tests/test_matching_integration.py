"""Integration tests for Phase 5 Dialogue Matching using the most recent small model transcript."""

import json
from pathlib import Path

import pytest

from matching import match_dialogue

pytestmark = pytest.mark.integration


class TestMatchingIntegration:
    # Use the most recent model transcript tested (small model)
    TRANSCRIPT_PATH = Path("output/248244667877_transcript_small.json")
    FALLBACK_PATH = Path("output/248244667877_transcript_base.json")

    @pytest.fixture
    def transcript_data(self):
        target_file = self.TRANSCRIPT_PATH if self.TRANSCRIPT_PATH.exists() else self.FALLBACK_PATH
        if not target_file.exists():
            pytest.skip(f"Transcript file not found at {target_file}")
        with open(target_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_sherlock_stagnation_quote(self, transcript_data):
        target_query = "My mind rebels at stagnation"
        print(f"\nSearching for target dialogue: '{target_query}'...")

        result = match_dialogue(target_query, transcript_data, confidence_threshold=80.0)

        assert result.match_found is True
        assert result.start_time is not None
        assert result.end_time is not None

        print(f"Match Results:")
        print(f"  Match Found:    {result.match_found}")
        print(f"  Matched Text:   '{result.matched_text}'")
        print(f"  Raw ASR Text:   '{result.matched_window_raw_text}'")
        print(f"  Start Time:     {result.start_time:.2f}s")
        print(f"  End Time:       {result.end_time:.2f}s")
        print(f"  Confidence:     {result.confidence:.1f}%")

        # Verify timestamp matches Sherlock's dialogue (~325s)
        assert 320.0 <= result.start_time <= 330.0

    def test_the_woman_quote(self, transcript_data):
        target_query = "To Sherlock Holmes she was always the woman"
        result = match_dialogue(target_query, transcript_data, confidence_threshold=80.0)

        assert result.match_found is True
        assert result.start_time is not None
        
        print(f"\nMatch Results:")
        print(f"  Match Found:    {result.match_found}")
        print(f"  Matched Text:   '{result.matched_text}'")
        print(f"  Raw ASR Text:   '{result.matched_window_raw_text}'")
        print(f"  Start Time:     {result.start_time:.2f}s")
        print(f"  End Time:       {result.end_time:.2f}s")
        print(f"  Confidence:     {result.confidence:.1f}%")

        # Verify timestamp matches opening narration (~117s -> 141s)
        assert 115.0 <= result.start_time <= 145.0
