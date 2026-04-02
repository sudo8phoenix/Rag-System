"""Voice activity detection using Silero VAD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional


class VADDependencyError(RuntimeError):
    """Raised when Silero VAD dependencies are unavailable."""


@dataclass(frozen=True)
class VADSegment:
    """Speech segment boundaries in sample offsets."""

    start: int
    end: int
    confidence: Optional[float] = None


def _default_silero_loader() -> tuple[Any, Callable[..., list[dict[str, Any]]]]:
    """Load Silero VAD model and timestamp function lazily."""
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:
        raise VADDependencyError(
            "silero-vad is not installed. Add it to requirements and reinstall dependencies."
        ) from exc

    model = load_silero_vad()
    return model, get_speech_timestamps


class SileroVAD:
    """Thin wrapper around Silero VAD speech boundary detection."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        model_loader: Optional[
            Callable[[], tuple[Any, Callable[..., list[dict[str, Any]]]]]
        ] = None,
    ) -> None:
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms

        self._model_loader = model_loader or _default_silero_loader
        self._model: Any | None = None
        self._timestamp_fn: Callable[..., list[dict[str, Any]]] | None = None

    def load_model(self) -> None:
        """Load model only once and cache it for repeated inference."""
        if self._model is None or self._timestamp_fn is None:
            self._model, self._timestamp_fn = self._model_loader()

    def detect_speech(
        self,
        audio: Iterable[float],
        sample_rate: int,
    ) -> list[VADSegment]:
        """Return speech segments for the provided mono waveform."""
        if sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")

        self.load_model()
        assert self._timestamp_fn is not None

        raw_segments = self._timestamp_fn(
            audio,
            self._model,
            sampling_rate=sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,
        )

        return [
            VADSegment(
                start=int(segment["start"]),
                end=int(segment["end"]),
                confidence=(
                    float(segment["confidence"])
                    if segment.get("confidence") is not None
                    else None
                ),
            )
            for segment in raw_segments
        ]

    def has_speech(self, audio: Iterable[float], sample_rate: int) -> bool:
        """Convenience method for quick speech presence checks."""
        return len(self.detect_speech(audio, sample_rate=sample_rate)) > 0
