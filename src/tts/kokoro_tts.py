"""Local kokoro text-to-speech backend."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Callable, Iterable

from src.config.settings import AppConfig, TTSConfig

from .base import BaseTTSBackend, TTSBackendError, TTSDependencyError


def _default_kokoro_factory() -> Any:
    try:
        from kokoro import KPipeline
    except ImportError as exc:  # pragma: no cover - exercised when dependency exists
        raise TTSDependencyError(
            "kokoro is not installed. Add it to requirements and reinstall dependencies."
        ) from exc

    return KPipeline


class KokoroTTS(BaseTTSBackend):
    """High-quality local kokoro synthesis backend."""

    engine_name = "kokoro"
    output_suffix = ".wav"
    sample_rate = 24000

    def __init__(
        self,
        config: TTSConfig | None = None,
        *,
        pipeline_factory: Callable[..., Any] | None = None,
        mixer_module: Any | Callable[[], Any] | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            config=config, mixer_module=mixer_module, output_dir=output_dir
        )
        self._pipeline_factory = pipeline_factory or _default_kokoro_factory
        self._pipeline: Any | None = None

    @classmethod
    def from_app_config(cls, config: AppConfig, **kwargs: Any) -> "KokoroTTS":
        return cls(config=config.tts, **kwargs)

    def _resolve_voice(self) -> str:
        requested = self.config.voice.strip().lower()
        if requested == "female":
            return "af_heart"
        if requested == "male":
            return "am_adam"
        if requested:
            return requested
        return "af_heart"

    def _resolve_lang_code(self) -> str:
        voice = self._resolve_voice()
        if "_" in voice and voice:
            return voice[0]
        return "a"

    def load_engine(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        pipeline_cls = self._pipeline_factory()
        self._pipeline = pipeline_cls(lang_code=self._resolve_lang_code())
        return self._pipeline

    def _iter_audio_samples(self, audio_chunks: Iterable[Any]) -> Iterable[float]:
        for chunk in audio_chunks:
            try:
                iterator = iter(chunk)
            except TypeError:
                yield float(chunk)
                continue

            for sample in iterator:
                yield float(sample)

    def _write_wave_file(self, path: Path, audio_chunks: list[Any]) -> None:
        frames = bytearray()
        for sample in self._iter_audio_samples(audio_chunks):
            value = max(-1.0, min(1.0, float(sample)))
            frames.extend(
                int(value * 32767).to_bytes(2, byteorder="little", signed=True)
            )

        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(bytes(frames))

    def synthesize_to_file(
        self, text: str, output_path: str | Path | None = None
    ) -> Path:
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        pipeline = self.load_engine()
        self._ensure_output_dir()

        path = (
            Path(output_path)
            if output_path is not None
            else self._default_output_path()
        )
        audio_chunks: list[Any] = []

        try:
            for _graphemes, _phonemes, audio in pipeline(
                text,
                voice=self._resolve_voice(),
                speed=float(self.config.rate),
                split_pattern=r"\n+",
            ):
                audio_chunks.append(audio)
        except TTSDependencyError:
            raise
        except Exception as exc:  # pragma: no cover - backend specific
            raise TTSBackendError(f"kokoro synthesis failed: {exc}") from exc

        if not audio_chunks:
            raise TTSBackendError("kokoro produced no audio")

        self._write_wave_file(path, audio_chunks)
        return path
