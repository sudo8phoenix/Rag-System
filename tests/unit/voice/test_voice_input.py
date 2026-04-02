from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.voice.vad import VADSegment
from src.voice.voice_input import VoiceInput


@dataclass
class _FakeAudioFrame:
    samples: list[float]
    sample_rate: int


@dataclass
class _FakeSTTResult:
    text: str
    confidence: float


class _FakeMicCapture:
    def record(self, duration_seconds: float):
        assert duration_seconds > 0
        # 400 samples where 100..200 is speech in tests
        return _FakeAudioFrame(samples=[0.0] * 400, sample_rate=16000)


class _FakeVADSpeech:
    def detect_speech(self, _samples, sample_rate: int):
        assert sample_rate == 16000
        return [VADSegment(start=100, end=200)]


class _FakeVADNoSpeech:
    def detect_speech(self, _samples, sample_rate: int):
        assert sample_rate == 16000
        return []


class _FakeSTT:
    def __init__(self):
        self.called_with: Path | None = None

    def transcribe(self, audio_source: str, language=None):
        self.called_with = Path(audio_source)
        assert self.called_with.suffix == ".wav"
        return _FakeSTTResult(text="transcribed text", confidence=0.77)


def test_capture_and_transcribe_happy_path() -> None:
    mic = _FakeMicCapture()
    vad = _FakeVADSpeech()
    stt = _FakeSTT()
    voice_input = VoiceInput(mic_capture=mic, vad=vad, stt=stt)

    result = voice_input.capture_and_transcribe(duration_seconds=1.0)

    assert result.speech_detected is True
    assert result.text == "transcribed text"
    assert result.confidence == pytest.approx(0.77)


def test_capture_and_transcribe_no_speech_returns_empty_text() -> None:
    voice_input = VoiceInput(
        mic_capture=_FakeMicCapture(),
        vad=_FakeVADNoSpeech(),
        stt=_FakeSTT(),
    )

    result = voice_input.capture_and_transcribe(duration_seconds=1.0)

    assert result.speech_detected is False
    assert result.text == ""
    assert result.confidence == pytest.approx(0.0)


def test_capture_and_transcribe_rejects_non_push_to_talk() -> None:
    voice_input = VoiceInput(
        mic_capture=_FakeMicCapture(),
        vad=_FakeVADNoSpeech(),
        stt=_FakeSTT(),
    )

    with pytest.raises(ValueError, match="push-to-talk"):
        voice_input.capture_and_transcribe(duration_seconds=1.0, push_to_talk=False)
