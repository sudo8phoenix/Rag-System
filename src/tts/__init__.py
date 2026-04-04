"""Text-to-speech package."""

from .base import (
    BaseTTSBackend,
    TTSBackendError,
    TTSConfigurationError,
    TTSDependencyError,
    TTSPlaybackError,
    TTSResult,
)
from .gtts_tts import GTTSTTS
from .kokoro_tts import KokoroTTS
from .orchestrator import TTSOrchestrator
from .pyttsx3_tts import Pyttsx3TTS

__all__ = [
    "BaseTTSBackend",
    "GTTSTTS",
    "KokoroTTS",
    "Pyttsx3TTS",
    "TTSBackendError",
    "TTSConfigurationError",
    "TTSDependencyError",
    "TTSOrchestrator",
    "TTSPlaybackError",
    "TTSResult",
]