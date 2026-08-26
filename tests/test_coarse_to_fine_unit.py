"""
Unit tests for V2 Coarse-to-Fine ASR Pipeline
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from asr import transcribe_audio
from asr.coarse_to_fine import detect_candidate_region, transcribe_audio_coarse_to_fine
from asr.errors import AudioNotFoundError, TranscriptionError
from asr.models import TranscriptSegment, WordTimestamp


class TestCandidateRegionDetection:
    """Unit tests for Stage 2 candidate region detection and +-5s padding."""

    def test_candidate_detection_exact_phrase(self):
        coarse_segs = [
            TranscriptSegment(text="hello and welcome back", start=0.0, end=10.0, words=[]),
            TranscriptSegment(text="in time my mind rebels at stagnation", start=320.0, end=327.0, words=[]),
            TranscriptSegment(text="goodbye for now", start=500.0, end=510.0, words=[]),
        ]
        pad_start, pad_end, score = detect_candidate_region(
            target_query="My mind rebels at stagnation",
            coarse_segments=coarse_segs,
            total_duration=600.0,
            padding_seconds=5.0,
        )

        assert pad_start == 315.0  # 320 - 5 = 315s
        assert pad_end == 332.0    # 327 + 5 = 332s
        assert score > 80.0

    def test_candidate_detection_padding_bounds_clamped_at_zero(self):
        coarse_segs = [
            TranscriptSegment(text="my mind rebels at stagnation", start=2.0, end=7.0, words=[]),
        ]
        pad_start, pad_end, _ = detect_candidate_region(
            target_query="My mind rebels at stagnation",
            coarse_segments=coarse_segs,
            total_duration=100.0,
            padding_seconds=5.0,
        )

        assert pad_start == 0.0  # max(0.0, 2 - 5) = 0.0s
        assert pad_end == 12.0   # min(100.0, 7 + 5) = 12.0s

    def test_candidate_detection_no_match_raises_error(self):
        coarse_segs = [
            TranscriptSegment(text="good morning everyone", start=0.0, end=10.0, words=[]),
        ]
        with pytest.raises(TranscriptionError, match="failed to locate candidate region"):
            detect_candidate_region(
                target_query="completely unrelated text query",
                coarse_segments=coarse_segs,
                total_duration=100.0,
                confidence_threshold=80.0,
            )


class TestCoarseToFineASR:
    """Unit tests for Stage 1 coarse & Stage 3 fine ASR pipeline execution."""

    def test_missing_audio_raises_error(self):
        with pytest.raises(AudioNotFoundError):
            transcribe_audio_coarse_to_fine(
                audio_path="nonexistent_file.wav",
                target_query="test query",
            )

    @patch("asr.coarse_to_fine.subprocess.run")
    @patch("asr.coarse_to_fine.get_whisper_model")
    @patch("asr.coarse_to_fine.shutil.which", return_value=True)
    def test_coarse_to_fine_pipeline_flow(self, mock_which, mock_get_model, mock_sub_run):
        # Setup mock coarse model
        coarse_seg = MagicMock(text="My mind rebels at stagnation", start=320.0, end=327.0, words=[])
        coarse_info = MagicMock(language="en", language_probability=0.98, duration=600.0)
        coarse_model_mock = MagicMock()
        coarse_model_mock.transcribe.return_value = ([coarse_seg], coarse_info)

        # Setup mock fine model
        fine_word1 = MagicMock(word="My", start=1.2, end=1.4, probability=0.99)
        fine_word2 = MagicMock(word="mind", start=1.5, end=1.8, probability=0.99)
        fine_seg = MagicMock(text="My mind", start=1.2, end=1.8, words=[fine_word1, fine_word2])
        fine_info = MagicMock(language="en", language_probability=0.98, duration=17.0)
        fine_model_mock = MagicMock()
        fine_model_mock.transcribe.return_value = ([fine_seg], fine_info)

        def mock_model_side_effect(model_size, **kwargs):
            if model_size == "base":
                return coarse_model_mock
            return fine_model_mock

        mock_get_model.side_effect = mock_model_side_effect

        with tempfile.TemporaryDirectory() as td:
            audio_wav = Path(td) / "audio.wav"
            audio_wav.write_bytes(b"dummy wav data")

            res = transcribe_audio_coarse_to_fine(
                audio_path=audio_wav,
                target_query="My mind rebels at stagnation",
                coarse_model_size="base",
                fine_model_size="small",
            )

            # Verify Stage 1 coarse model called with word_timestamps=False
            coarse_model_mock.transcribe.assert_called_once()
            coarse_call_kwargs = coarse_model_mock.transcribe.call_args[1]
            assert coarse_call_kwargs["word_timestamps"] is False
            assert coarse_call_kwargs["vad_filter"] is True

            # Verify Stage 3 fine model called with word_timestamps=True
            fine_model_mock.transcribe.assert_called_once()
            fine_call_kwargs = fine_model_mock.transcribe.call_args[1]
            assert fine_call_kwargs["word_timestamps"] is True
            assert fine_call_kwargs["vad_filter"] is True

            # Verify fine segment timestamps re-offset by pad_start (315.0s)
            assert res.segments[0].start == round(1.2 + 315.0, 3)  # 316.2s
            assert res.segments[0].words[0].start == round(1.2 + 315.0, 3)  # 316.2s
            assert res.model_name == "base->small"

    def test_transcribe_audio_standard_vs_coarse_to_fine_routing(self):
        with tempfile.TemporaryDirectory() as td:
            audio_wav = Path(td) / "audio.wav"
            audio_wav.write_bytes(b"dummy wav data")

            with patch("asr.coarse_to_fine.transcribe_audio_coarse_to_fine") as mock_v2:
                mock_v2.return_value = MagicMock()
                transcribe_audio(
                    audio_path=audio_wav,
                    mode="v2",
                    target_query="My mind rebels at stagnation",
                )
                mock_v2.assert_called_once()

            with patch("asr.transcribe_audio_file") as mock_v1:
                mock_v1.return_value = MagicMock()
                transcribe_audio(
                    audio_path=audio_wav,
                    mode="standard",
                )
                mock_v1.assert_called_once()
