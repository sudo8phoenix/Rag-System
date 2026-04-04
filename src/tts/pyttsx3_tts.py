"""Local text-to-speech synthesis using pyttsx3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.config.settings import AppConfig, TTSConfig

from .base import (
    BaseTTSBackend,
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


def _default_engine_factory() -> Any:
    """Import pyttsx3 lazily so the module remains importable without it."""
    try:
        import pyttsx3
    except ImportError as exc:  # pragma: no cover - exercised when dependency exists
        raise TTSDependencyError(
            "pyttsx3 is not installed. Add it to requirements and reinstall dependencies."
        ) from exc

    return pyttsx3.init()


def _normalise_voice_name(value: str) -> str:
    return value.strip().lower()


class Pyttsx3TTS(BaseTTSBackend):
    """pyttsx3 synthesis and pygame playback backend."""

    engine_name = "pyttsx3"
    output_suffix = ".wav"

    def __init__(
        self,
        config: TTSConfig | None = None,
        *,
        engine_factory: Callable[[], Any] | None = None,
        mixer_module: Any | Callable[[], Any] | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        super().__init__(config=config, mixer_module=mixer_module, output_dir=output_dir)
        self._engine_factory = engine_factory or _default_engine_factory
        self._engine: Any | None = None

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

    def synthesize_to_file(self, text: str, output_path: str | Path | None = None) -> Path:
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        engine = self.load_engine()
        self._ensure_output_dir()

        path = Path(output_path) if output_path is not None else self._default_output_path()
        engine.save_to_file(text, str(path))
        engine.runAndWait()
        return path