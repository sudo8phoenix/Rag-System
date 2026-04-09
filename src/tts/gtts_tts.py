"""Google text-to-speech backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.config.settings import AppConfig, TTSConfig

from .base import BaseTTSBackend, TTSBackendError, TTSDependencyError


def _default_gtts_factory(*args: Any, **kwargs: Any) -> Any:
    try:
        from gtts import gTTS
    except ImportError as exc:  # pragma: no cover - exercised when dependency exists
        raise TTSDependencyError(
            "gTTS is not installed. Add it to requirements and reinstall dependencies."
        ) from exc

    return gTTS(*args, **kwargs)


class GTTSTTS(BaseTTSBackend):
    """gTTS synthesis backend that writes mp3 files."""

    engine_name = "gtts"
    output_suffix = ".mp3"

    def __init__(
        self,
        config: TTSConfig | None = None,
        *,
        gtts_factory: Callable[..., Any] | None = None,
        mixer_module: Any | Callable[[], Any] | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        super().__init__(config=config, mixer_module=mixer_module, output_dir=output_dir)
        self._gtts_factory = gtts_factory or _default_gtts_factory

    @classmethod
    def from_app_config(cls, config: AppConfig, **kwargs: Any) -> "GTTSTTS":
        return cls(config=config.tts, **kwargs)

    def _resolve_language(self) -> str:
        requested = self.config.voice.strip().lower()
        if requested in {"male", "female", ""}:
            return "en"

        if len(requested) in {2, 5} and requested.replace("-", "").replace("_", "").isalpha():
            return requested

        if requested.startswith("en"):
            return "en"

        return "en"

    def synthesize_to_file(self, text: str, output_path: str | Path | None = None) -> Path:
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        self._ensure_output_dir()
        path = Path(output_path) if output_path is not None else self._default_output_path()
        slow = float(self.config.rate) < 1.0

        try:
            tts = self._gtts_factory(text=text, lang=self._resolve_language(), slow=slow)
            tts.save(str(path))
        except TTSDependencyError:
            raise
        except Exception as exc:  # pragma: no cover - backend specific
            raise TTSBackendError(f"gTTS synthesis failed: {exc}") from exc

        return path