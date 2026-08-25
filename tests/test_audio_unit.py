"""Unit tests for Phase 3 Audio Extraction. All subprocess calls are mocked."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audio import extract_audio
from audio.models import AudioMetadata, AudioResult, AudioExtractionError
from audio.probe import probe_audio_file
from audio.extractor import extract_audio_stream


SAMPLE_FFPROBE_WAV_JSON = {
    "streams": [
        {
            "codec_type": "audio",
            "codec_name": "pcm_s16le",
            "sample_rate": "16000",
            "channels": 1,
            "duration": "3261.8",
        }
    ],
    "format": {
        "duration": "3261.8",
        "format_name": "wav",
        "size": "104377644",
    },
}


class TestAudioModels:
    def test_audio_metadata(self):
        m = AudioMetadata(
            duration=3261.8,
            sample_rate=16000,
            channels=1,
            codec_name="pcm_s16le",
            file_size_bytes=104377644,
        )
        assert m.duration == 3261.8
        assert m.sample_rate == 16000
        assert m.channels == 1

    def test_audio_result(self):
        m = AudioMetadata(
            duration=60.0, sample_rate=16000, channels=1,
            codec_name="pcm_s16le", file_size_bytes=1920000,
        )
        r = AudioResult(
            audio_path=Path("audio.wav"),
            source_video_path=Path("video.mp4"),
            metadata=m,
            extraction_duration_seconds=2.5,
        )
        assert r.audio_path == Path("audio.wav")
        assert r.extraction_duration_seconds == 2.5


class TestAudioProbe:
    def test_nonexistent_file(self):
        with pytest.raises(AudioExtractionError, match="does not exist"):
            probe_audio_file(Path("/nonexistent/audio.wav"))

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            pass
        with pytest.raises(AudioExtractionError, match="empty"):
            probe_audio_file(Path(f.name))

    @patch("audio.probe.shutil.which", return_value="ffprobe")
    @patch("audio.probe.subprocess.run")
    def test_successful_probe(self, mock_run, _):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(SAMPLE_FFPROBE_WAV_JSON)
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake wav bytes")
            f.flush()
            meta = probe_audio_file(Path(f.name))

        assert meta.duration == 3261.8
        assert meta.sample_rate == 16000
        assert meta.channels == 1
        assert meta.codec_name == "pcm_s16le"


class TestAudioExtractor:
    @patch("audio.extractor.shutil.which", return_value=None)
    def test_missing_ffmpeg(self, _):
        with pytest.raises(AudioExtractionError, match="ffmpeg not found"):
            extract_audio_stream(Path("video.mp4"), Path("audio.wav"))

    @patch("audio.extractor.subprocess.run")
    @patch("audio.extractor.shutil.which", return_value="ffmpeg")
    def test_successful_extraction(self, _, mock_run):
        with tempfile.TemporaryDirectory() as td:
            video_path = Path(td) / "input.mp4"
            video_path.write_bytes(b"fake video")

            output_wav = Path(td) / "output.wav"
            
            # Simulate FFmpeg creating output file
            def side_effect(cmd, **kwargs):
                output_wav.write_bytes(b"fake wav content")
                return MagicMock(returncode=0, stdout="", stderr="")

            mock_run.side_effect = side_effect

            res = extract_audio_stream(video_path, output_wav)
            assert res == output_wav
            assert output_wav.exists()
            
            # Verify FFmpeg flags passed correctly
            cmd_args = mock_run.call_args[0][0]
            assert "-vn" in cmd_args
            assert "-ac" in cmd_args and "1" in cmd_args
            assert "-ar" in cmd_args and "16000" in cmd_args
            assert "-c:a" in cmd_args and "pcm_s16le" in cmd_args
