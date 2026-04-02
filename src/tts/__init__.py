"""Text-to-speech package."""

from .pyttsx3_tts import (
    Pyttsx3TTS,
    TTSConfigurationError,
    TTSDependencyError,
    TTSPlaybackError,
    TTSResult,
)

__all__ = [
    "Pyttsx3TTS",
    "TTSConfigurationError",
    "TTSDependencyError",
    "TTSPlaybackError",
    "TTSResult",
]