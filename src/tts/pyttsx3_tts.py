"""Local text-to-speech synthesis and playback using pyttsx3 and pygame."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from src.config.settings import AppConfig, TTSConfig


class TTSDependencyError(RuntimeError):
    """Raised when a required TTS dependency is unavailable."""


class TTSConfigurationError(ValueError):
    """Raised when the configured TTS engine or voice cannot be used."""


class TTSPlaybackError(RuntimeError):
    """Raised when audio playback fails."""


@dataclass(frozen=True)
class TTSResult:
    """Generated audio metadata for a synthesized response."""

    text: str
    audio_path: Path
    engine: str
    played: bool


def _default_engine_factory() -> Any:
    """Import pyttsx3 lazily so the module remains importable without it."""
    try:
        import pyttsx3
    except ImportError as exc:  # pragma: no cover - exercised when dependency exists
        raise TTSDependencyError(
            "pyttsx3 is not installed. Add it to requirements and reinstall dependencies."
        ) from exc

    return pyttsx3.init()


def _default_mixer_module() -> Any:
    """Import pygame.mixer lazily for speaker playback."""
    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - exercised when dependency exists
        raise TTSDependencyError(
            "pygame is not installed. Add it to requirements and reinstall dependencies."
        ) from exc

    return pygame.mixer


def _normalise_voice_name(value: str) -> str:
    return value.strip().lower()


class Pyttsx3TTS:
    """Phase 1 TTS implementation based on pyttsx3 synthesis and pygame playback."""

    def __init__(
        self,
        config: TTSConfig | None = None,
        *,
        engine_factory: Callable[[], Any] | None = None,
        mixer_module: Any | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config or TTSConfig()
        self._engine_factory = engine_factory or _default_engine_factory
        self._mixer_module = mixer_module or _default_mixer_module
        self.output_dir = Path(output_dir or Path("./data/tts"))
        self._engine: Any | None = None
        self._mixer: Any | None = None

        if self.config.engine != "pyttsx3":
            raise TTSConfigurationError(
                f"Phase 1 TTS only supports pyttsx3, not {self.config.engine!r}"
            )

    @classmethod
    def from_app_config(cls, config: AppConfig, **kwargs: Any) -> "Pyttsx3TTS":
        return cls(config=config.tts, **kwargs)

    def load_engine(self) -> Any:
        """Create and configure the speech engine once."""
        if self._engine is not None:
            return self._engine

        engine = self._engine_factory()
        self._configure_engine(engine)
        self._engine = engine
        return engine

    def _configure_engine(self, engine: Any) -> None:
        voices = []
        try:
            voices = list(engine.getProperty("voices") or [])
        except Exception:
            voices = []

        selected_voice = self._select_voice(voices)
        if selected_voice is not None:
            engine.setProperty("voice", selected_voice)

        try:
            default_rate = float(engine.getProperty("rate") or 200.0)
        except Exception:
            default_rate = 200.0

        engine.setProperty("rate", int(default_rate * float(self.config.rate)))
        engine.setProperty("volume", float(self.config.volume))

    def _select_voice(self, voices: list[Any]) -> str | None:
        requested = _normalise_voice_name(self.config.voice)
        if not voices:
            return None

        def _voice_text(voice: Any) -> str:
            parts = [
                getattr(voice, "id", ""),
                getattr(voice, "name", ""),
                getattr(voice, "gender", ""),
                getattr(voice, "languages", ""),
            ]
            return " ".join(str(part).lower() for part in parts if part)

        if requested not in {"male", "female"}:
            for voice in voices:
                if requested in _voice_text(voice):
                    return str(getattr(voice, "id", "")) or None
            return None

        for voice in voices:
            voice_text = _voice_text(voice)
            if requested in voice_text:
                return str(getattr(voice, "id", "")) or None

        for voice in voices:
            voice_name = str(getattr(voice, "name", "")).lower()
            if requested in voice_name:
                return str(getattr(voice, "id", "")) or None

        return None

    def _ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize_to_file(self, text: str, output_path: str | Path | None = None) -> Path:
        """Generate speech audio for the provided text and write it to disk."""
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        engine = self.load_engine()
        self._ensure_output_dir()

        path = Path(output_path) if output_path is not None else self.output_dir / f"tts_{uuid4().hex}.wav"
        engine.save_to_file(text, str(path))
        engine.runAndWait()
        return path

    def load_audio(self, audio_path: str | Path) -> Any:
        """Load the platform mixer if needed and prepare playback."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if self._mixer is None:
            self._mixer = self._mixer_module

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
            engine=self.config.engine,
            played=played,
        )