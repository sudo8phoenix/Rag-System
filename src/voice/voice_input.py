"""Voice input orchestrator combining mic capture, VAD, and STT."""

from __future__ import annotations

import os
import tempfile
import wave
from dataclasses import dataclass
from typing import Optional

from .mic_capture import AudioFrame, MicrophoneCapture
from .stt import FasterWhisperSTT
from .vad import SileroVAD, VADSegment


@dataclass(frozen=True)
class VoiceInputResult:
    """Output of voice capture pipeline."""

    text: str
    confidence: float
    speech_detected: bool


class VoiceInput:
    """High-level voice capture pipeline for push-to-talk flows."""

    def __init__(
        self,
        mic_capture: Optional[MicrophoneCapture] = None,
        vad: Optional[SileroVAD] = None,
        stt: Optional[FasterWhisperSTT] = None,
    ) -> None:
        self.mic_capture = mic_capture or MicrophoneCapture()
        self.vad = vad or SileroVAD()
        self.stt = stt or FasterWhisperSTT()

    def _extract_speech_samples(
        self,
        frame: AudioFrame,
        segments: list[VADSegment],
    ) -> list[float]:
        speech_samples: list[float] = []
        for segment in segments:
            start = max(0, segment.start)
            end = min(len(frame.samples), segment.end)
            if end > start:
                speech_samples.extend(frame.samples[start:end])
        return speech_samples

    def _write_temp_wav(self, samples: list[float], sample_rate: int) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="voice_input_")
        os.close(fd)

        pcm = bytearray()
        for sample in samples:
            clipped = max(-1.0, min(1.0, sample))
            int16_value = int(clipped * 32767)
            pcm.extend(int16_value.to_bytes(2, byteorder="little", signed=True))

        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(bytes(pcm))

        return path

    def capture_and_transcribe(
        self,
        duration_seconds: float = 5.0,
        push_to_talk: bool = True,
        language: Optional[str] = None,
    ) -> VoiceInputResult:
        """Capture microphone input and return transcription text with confidence."""
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if not push_to_talk:
            raise ValueError("Only push-to-talk mode is supported in Phase 1")

        frame = self.mic_capture.record(duration_seconds=duration_seconds)
        segments = self.vad.detect_speech(frame.samples, sample_rate=frame.sample_rate)

        if not segments:
            return VoiceInputResult(text="", confidence=0.0, speech_detected=False)

        speech_samples = self._extract_speech_samples(frame, segments)
        if not speech_samples:
            return VoiceInputResult(text="", confidence=0.0, speech_detected=False)

        wav_path = self._write_temp_wav(speech_samples, sample_rate=frame.sample_rate)
        try:
            stt_result = self.stt.transcribe(wav_path, language=language)
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

        return VoiceInputResult(
            text=stt_result.text,
            confidence=float(stt_result.confidence or 0.0),
            speech_detected=True,
        )
