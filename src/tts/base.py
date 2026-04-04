"""Shared TTS primitives and playback helpers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from src.config.settings import AppConfig, TTSConfig


class TTSDependencyError(RuntimeError):
    """Raised when a required TTS dependency is unavailable."""


class TTSConfigurationError(ValueError):
    """Raised when the configured TTS engine or voice cannot be used."""


class TTSBackendError(RuntimeError):
    """Raised when a TTS backend fails while generating audio."""


class TTSPlaybackError(RuntimeError):
    """Raised when audio playback fails."""


@dataclass(frozen=True)
class TTSResult:
    """Generated audio metadata for a synthesized response."""

    text: str
    audio_path: Path
    engine: str
    played: bool


def _default_mixer_module() -> Any:
    """Import pygame.mixer lazily for speaker playback."""
    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - exercised when dependency exists
        raise TTSDependencyError(
            "pygame is not installed. Add it to requirements and reinstall dependencies."
        ) from exc

    return pygame.mixer


class BaseTTSBackend(ABC):
    """Common TTS behavior shared by every backend."""

    engine_name = "tts"
    output_suffix = ".wav"

    def __init__(
        self,
        config: TTSConfig | None = None,
        *,
        mixer_module: Any | Callable[[], Any] | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config or TTSConfig()
        self._mixer_source = mixer_module or _default_mixer_module
        self.output_dir = Path(output_dir or Path("./data/tts"))
        self._mixer: Any | None = None

    @classmethod
    def from_app_config(cls, config: AppConfig, **kwargs: Any):
        return cls(config=config.tts, **kwargs)

    def _ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _default_output_path(self) -> Path:
        return self.output_dir / f"{self.engine_name}_{uuid4().hex}{self.output_suffix}"

    @abstractmethod
    def synthesize_to_file(self, text: str, output_path: str | Path | None = None) -> Path:
        """Generate speech audio for the provided text and write it to disk."""

    def load_audio(self, audio_path: str | Path) -> Any:
        """Load the platform mixer if needed and prepare playback."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if self._mixer is None:
            self._mixer = self._mixer_source() if callable(self._mixer_source) else self._mixer_source

        try:
            self._mixer.init()
        except Exception as exc:  # pragma: no cover - backend specific
            raise TTSPlaybackError(f"Unable to initialize audio playback: {exc}") from exc

        return self._mixer

    def play_audio(self, audio_path: str | Path, *, block: bool = True) -> None:
        """Play an audio file through the system speakers."""
        mixer = self.load_audio(audio_path)

        try:
            mixer.music.load(str(audio_path))
            mixer.music.play()
            if block:
                while mixer.music.get_busy():
                    time.sleep(0.05)
        except Exception as exc:  # pragma: no cover - backend specific
            raise TTSPlaybackError(f"Unable to play audio {audio_path}: {exc}") from exc

    def speak(
        self,
        text: str,
        *,
        output_path: str | Path | None = None,
        block: bool = True,
    ) -> TTSResult:
        """Synthesize text to audio and optionally play it back."""
        audio_path = self.synthesize_to_file(text, output_path=output_path)
        played = False

        if not self.config.mute:
            self.play_audio(audio_path, block=block)
            played = True

        return TTSResult(
            text=text,
            audio_path=audio_path,
            engine=self.engine_name,
            played=played,
        )