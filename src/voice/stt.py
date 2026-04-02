"""Speech-to-text wrapper using faster-whisper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


class STTDependencyError(RuntimeError):
    """Raised when faster-whisper dependency is unavailable."""


@dataclass(frozen=True)
class STTSegment:
    """Single transcription segment."""

    text: str
    start: float
    end: float
    avg_logprob: Optional[float] = None


@dataclass(frozen=True)
class STTResult:
    """Transcription result with aggregated text and confidence."""

    text: str
    segments: list[STTSegment]
    language: Optional[str] = None
    confidence: Optional[float] = None


def _default_model_factory(
    model_size: str,
    device: str,
    compute_type: str,
) -> Any:
    """Create WhisperModel lazily so import errors are explicit and recoverable."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise STTDependencyError(
            "faster-whisper is not installed. Add it to requirements and reinstall dependencies."
        ) from exc

    return WhisperModel(model_size, device=device, compute_type=compute_type)


class FasterWhisperSTT:
    """Thin wrapper for local speech transcription with faster-whisper."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        model_factory: Optional[Callable[[str, str, str], Any]] = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size

        self._model_factory = model_factory or _default_model_factory
        self._model: Any | None = None

    def load_model(self) -> None:
        """Load and cache Whisper model on first use."""
        if self._model is None:
            self._model = self._model_factory(
                self.model_size,
                self.device,
                self.compute_type,
            )

    def transcribe(
        self,
        audio_source: str | Path,
        language: Optional[str] = None,
        vad_filter: bool = True,
    ) -> STTResult:
        """Transcribe an audio file path. Supported formats are handled by ffmpeg."""
        self.load_model()
        assert self._model is not None

        segments, info = self._model.transcribe(
            str(audio_source),
            beam_size=self.beam_size,
            language=language,
            vad_filter=vad_filter,
        )

        segment_list = [
            STTSegment(
                text=str(segment.text).strip(),
                start=float(segment.start),
                end=float(segment.end),
                avg_logprob=getattr(segment, "avg_logprob", None),
            )
            for segment in segments
        ]

        text = " ".join(segment.text for segment in segment_list).strip()
        confidence = getattr(info, "language_probability", None)

        return STTResult(
            text=text,
            segments=segment_list,
            language=getattr(info, "language", None),
            confidence=float(confidence) if confidence is not None else None,
        )
