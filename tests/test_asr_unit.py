"""Unit tests for Phase 4 ASR Speech Recognition. All model inference is mocked."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from asr import transcribe_audio
from asr.models import WordTimestamp, TranscriptSegment, TranscriptionResult
from asr.errors import ASRError, AudioNotFoundError, ModelLoadError, TranscriptionError
from asr.transcriber import transcribe_audio_file


class TestASRModels:
    def test_word_timestamp(self):
        w = WordTimestamp(word="hello", start=1.2, end=1.5, probability=0.98)
        assert w.word == "hello"
        assert w.start == 1.2
        assert w.end == 1.5
        assert w.probability == 0.98

    def test_transcript_segment(self):
        w1 = WordTimestamp(word="hello", start=1.2, end=1.5)
        w2 = WordTimestamp(word="world", start=1.6, end=2.0)
        seg = TranscriptSegment(text="hello world", start=1.2, end=2.0, words=[w1, w2])
        assert seg.text == "hello world"
        assert len(seg.words) == 2

    def test_transcription_result_full_text(self):
        seg1 = TranscriptSegment(text="Hello world", start=0.0, end=1.5)
        seg2 = TranscriptSegment(text="this is a test", start=1.6, end=3.0)
        res = TranscriptionResult(
            audio_path=Path("audio.wav"),
            segments=[seg1, seg2],
            language="en",
            language_probability=0.99,
            duration=3.0,
            model_name="tiny",
            transcription_duration_seconds=0.5,
        )
        assert res.full_text == "Hello world this is a test"
        assert res.language == "en"


class TestASRErrors:
    def test_nonexistent_audio(self):
        with pytest.raises(AudioNotFoundError, match="does not exist"):
            transcribe_audio_file(Path("/nonexistent/audio.wav"))

    def test_empty_audio(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            pass
        with pytest.raises(AudioNotFoundError, match="empty"):
            transcribe_audio_file(Path(f.name))

    @patch("asr.transcriber.get_whisper_model")
    def test_transcription_failure(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("GPU Out of memory")
        mock_get_model.return_value = mock_model

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            f.flush()
            with pytest.raises(TranscriptionError, match="Whisper transcription failed"):
                transcribe_audio_file(Path(f.name))


class TestASRTranscriberMocked:
    @patch("asr.transcriber.get_whisper_model")
    def test_successful_transcription(self, mock_get_model):
        # Create mock segment return from faster-whisper
        mock_word = MagicMock(word="immediately", start=122.0, end=122.4, probability=0.95)
        mock_seg = MagicMock(
            text=" I need your help immediately ",
            start=120.5,
            end=123.8,
            words=[mock_word],
        )
        mock_info = MagicMock(
            language="en",
            language_probability=0.9876,
            duration=125.0,
        )

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_seg], mock_info)
        mock_get_model.return_value = mock_model

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            f.flush()

            result = transcribe_audio(Path(f.name), model_size="tiny")

            assert result.language == "en"
            assert result.duration == 125.0
            assert len(result.segments) == 1
            assert result.segments[0].text == "I need your help immediately"
            assert result.segments[0].start == 120.5
            assert result.segments[0].end == 123.8
            assert len(result.segments[0].words) == 1
            assert result.segments[0].words[0].word == "immediately"
            assert result.segments[0].words[0].start == 122.0
            assert result.segments[0].words[0].end == 122.4
