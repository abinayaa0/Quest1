"""Exception hierarchy for ASR Speech Recognition module."""


class ASRError(Exception):
    """Base exception for all ASR transcription failures."""
    pass


class AudioNotFoundError(ASRError):
    """Audio file missing or empty."""
    pass


class ModelLoadError(ASRError):
    """Failed to load Faster-Whisper model weights."""
    pass


class TranscriptionError(ASRError):
    """ASR transcription inference execution failed."""
    pass
