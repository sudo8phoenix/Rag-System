from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.voice.stt import FasterWhisperSTT, STTDependencyError


@dataclass
class _FakeSegment:
    text: str
    start: float
    end: float
    avg_logprob: float = -0.12


@dataclass
class _FakeInfo:
    language: str = "en"
    language_probability: float = 0.88


class _FakeWhisperModel:
    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, audio_source: str, **kwargs):
        self.calls.append((audio_source, kwargs))
        return [
            _FakeSegment("hello", 0.0, 0.5),
            _FakeSegment("world", 0.5, 1.0),
        ], _FakeInfo()


def test_transcribe_returns_expected_result() -> None:
    fake_model = _FakeWhisperModel()

    def fake_factory(_model_size: str, _device: str, _compute_type: str):
        return fake_model

    stt = FasterWhisperSTT(model_factory=fake_factory)
    result = stt.transcribe("tests/data/sample.wav", language="en")

    assert result.text == "hello world"
    assert result.language == "en"
    assert result.confidence == pytest.approx(0.88)
    assert len(result.segments) == 2
    assert fake_model.calls[0][0] == "tests/data/sample.wav"
    assert fake_model.calls[0][1]["language"] == "en"


def test_load_model_dependency_error() -> None:
    def bad_factory(_model_size: str, _device: str, _compute_type: str):
        raise STTDependencyError("missing")

    stt = FasterWhisperSTT(model_factory=bad_factory)

    with pytest.raises(STTDependencyError):
        stt.load_model()
