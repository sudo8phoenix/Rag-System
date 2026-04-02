from __future__ import annotations

from typing import Any

import pytest

from src.voice.mic_capture import MicrophoneCapture


class _FakeInputStream:
    def __init__(self, callback, blocksize: int, parent: Any, **_kwargs):
        self._callback = callback
        self._blocksize = blocksize
        self._parent = parent

    def __enter__(self):
        # Simulate 3 streaming chunks immediately.
        for _ in range(3):
            self._callback([0.1] * self._blocksize, self._blocksize, None, None)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeSoundDevice:
    def __init__(self) -> None:
        self.wait_called = False
        self.sleep_calls = 0

    def query_devices(self):
        return [
            {
                "name": "Mic-1",
                "max_input_channels": 2,
                "default_samplerate": 16000,
            }
        ]

    def rec(self, frames, samplerate, channels, dtype):
        assert frames > 0
        assert samplerate == 16000
        assert channels == 1
        assert dtype == "float32"
        return [0.2] * frames

    def wait(self):
        self.wait_called = True

    def InputStream(self, **kwargs):
        return _FakeInputStream(parent=self, **kwargs)

    def sleep(self, _ms: int):
        self.sleep_calls += 1


def test_record_returns_audio_frame() -> None:
    fake_sd = _FakeSoundDevice()
    mic = MicrophoneCapture(sounddevice_module=fake_sd)

    frame = mic.record(duration_seconds=0.1)

    assert frame.sample_rate == 16000
    assert len(frame.samples) == 1600
    assert fake_sd.wait_called is True


def test_stream_chunks_returns_iterable_chunks() -> None:
    fake_sd = _FakeSoundDevice()
    mic = MicrophoneCapture(sounddevice_module=fake_sd, chunk_size=32)

    chunks = list(mic.stream_chunks(duration_seconds=0.05))

    assert len(chunks) >= 1
    assert all(len(chunk.samples) == 32 for chunk in chunks)


def test_list_input_devices_from_sounddevice() -> None:
    fake_sd = _FakeSoundDevice()
    mic = MicrophoneCapture(sounddevice_module=fake_sd)

    devices = mic.list_input_devices()

    assert len(devices) == 1
    assert devices[0]["name"] == "Mic-1"


def test_record_rejects_non_positive_duration() -> None:
    mic = MicrophoneCapture(sounddevice_module=_FakeSoundDevice())

    with pytest.raises(ValueError, match="duration_seconds"):
        mic.record(0)
