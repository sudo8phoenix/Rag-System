from __future__ import annotations

from typing import Any

import pytest

from src.voice.vad import SileroVAD, VADDependencyError


def test_detect_speech_returns_segment_objects() -> None:
    calls: dict[str, Any] = {}

    def fake_timestamp_fn(audio, model, **kwargs):
        calls["audio"] = list(audio)
        calls["model"] = model
        calls["kwargs"] = kwargs
        return [{"start": 10, "end": 40, "confidence": 0.91}]

    def fake_loader():
        return object(), fake_timestamp_fn

    vad = SileroVAD(model_loader=fake_loader)
    segments = vad.detect_speech([0.0, 0.1, 0.2], sample_rate=16000)

    assert len(segments) == 1
    assert segments[0].start == 10
    assert segments[0].end == 40
    assert segments[0].confidence == pytest.approx(0.91)

    assert calls["kwargs"]["sampling_rate"] == 16000
    assert calls["kwargs"]["threshold"] == 0.5


def test_detect_speech_invalid_sample_rate() -> None:
    def fake_loader():
        return object(), lambda *_args, **_kwargs: []

    vad = SileroVAD(model_loader=fake_loader)

    with pytest.raises(ValueError, match="sample_rate"):
        vad.detect_speech([0.0], sample_rate=0)


def test_missing_dependency_raises_custom_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.voice import vad as vad_module

    def raise_loader():
        raise VADDependencyError("missing")

    monkeypatch.setattr(vad_module, "_default_silero_loader", raise_loader)

    vad = SileroVAD(model_loader=vad_module._default_silero_loader)
    with pytest.raises(VADDependencyError):
        vad.load_model()
