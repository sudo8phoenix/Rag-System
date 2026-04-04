"""Config-driven TTS engine selection and fallback orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.config.settings import AppConfig, TTSConfig

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
from .pyttsx3_tts import Pyttsx3TTS


SupportedBackend = Callable[[], BaseTTSBackend]


class TTSOrchestrator:
    """Select a TTS backend from config and fall back when it fails."""

    SUPPORTED_ENGINES = ("pyttsx3", "gtts", "kokoro")

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        backend_factories: dict[str, SupportedBackend] | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self._backend_factories = backend_factories or {}
        self.output_dir = Path(output_dir or Path("./data/tts"))

        if self.config.tts.engine not in self.SUPPORTED_ENGINES:
            raise TTSConfigurationError(
                f"Unsupported TTS engine: {self.config.tts.engine!r}"
            )

        self._backend_cache: dict[str, BaseTTSBackend] = {}

    @classmethod
    def from_app_config(cls, config: AppConfig, **kwargs: Any) -> "TTSOrchestrator":
        return cls(config=config, **kwargs)

    def _candidate_engines(self) -> list[str]:
        primary = self.config.tts.engine
        return [primary, *[engine for engine in self.SUPPORTED_ENGINES if engine != primary]]

    def _build_backend(self, engine: str) -> BaseTTSBackend:
        if engine in self._backend_cache:
            return self._backend_cache[engine]

        if engine in self._backend_factories:
            backend = self._backend_factories[engine]()
            self._backend_cache[engine] = backend
            return backend

        backend_map: dict[str, Callable[[TTSConfig], BaseTTSBackend]] = {
            "pyttsx3": Pyttsx3TTS,
            "gtts": GTTSTTS,
            "kokoro": KokoroTTS,
        }

        backend = backend_map[engine](self.config.tts)
        self._backend_cache[engine] = backend
        return backend

    def speak(
        self,
        text: str,
        *,
        output_path: str | Path | None = None,
        block: bool = True,
    ) -> TTSResult:
        errors: list[str] = []
        for engine in self._candidate_engines():
            backend = self._build_backend(engine)
            try:
                return backend.speak(text, output_path=output_path, block=block)
            except (
                TTSDependencyError,
                TTSConfigurationError,
                TTSBackendError,
                TTSPlaybackError,
                ValueError,
                OSError,
            ) as exc:
                errors.append(f"{engine}: {exc}")

        raise TTSBackendError("All configured TTS engines failed: " + "; ".join(errors))