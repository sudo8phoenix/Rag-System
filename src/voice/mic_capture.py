"""Microphone capture utilities for real-time and fixed-duration recording."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Iterator, Optional


class MicCaptureDependencyError(RuntimeError):
    """Raised when required microphone backend dependencies are missing."""


@dataclass(frozen=True)
class AudioFrame:
    """Container for mono PCM samples and sampling metadata."""

    samples: list[float]
    sample_rate: int


class MicrophoneCapture:
    """Capture audio from microphone via sounddevice, with pyaudio discovery fallback."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        dtype: str = "float32",
        sounddevice_module: Any | None = None,
        pyaudio_module: Any | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.dtype = dtype

        self._sounddevice = sounddevice_module
        self._pyaudio = pyaudio_module

    def _ensure_sounddevice(self) -> Any:
        if self._sounddevice is not None:
            return self._sounddevice
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise MicCaptureDependencyError(
                "sounddevice is not installed. Add it to requirements and reinstall dependencies."
            ) from exc
        self._sounddevice = sd
        return sd

    def _ensure_pyaudio(self) -> Any:
        if self._pyaudio is not None:
            return self._pyaudio
        try:
            import pyaudio
        except ImportError as exc:
            raise MicCaptureDependencyError(
                "pyaudio is not installed. Add it to requirements and reinstall dependencies."
            ) from exc
        self._pyaudio = pyaudio
        return pyaudio

    def _to_list(self, data: Any) -> list[float]:
        if hasattr(data, "flatten"):
            return [float(x) for x in data.flatten().tolist()]
        if isinstance(data, list):
            return [float(x) for x in data]
        return [float(x) for x in data]

    def list_input_devices(self) -> list[dict[str, Any]]:
        """List available input devices from sounddevice or pyaudio."""
        try:
            sd = self._ensure_sounddevice()
            devices = sd.query_devices()
            return [
                {
                    "name": d.get("name", "unknown"),
                    "max_input_channels": int(d.get("max_input_channels", 0)),
                    "default_samplerate": float(d.get("default_samplerate", 0)),
                }
                for d in devices
                if int(d.get("max_input_channels", 0)) > 0
            ]
        except MicCaptureDependencyError:
            pyaudio = self._ensure_pyaudio()
            pa = pyaudio.PyAudio()
            devices: list[dict[str, Any]] = []
            for index in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(index)
                if int(info.get("maxInputChannels", 0)) > 0:
                    devices.append(
                        {
                            "name": info.get("name", "unknown"),
                            "max_input_channels": int(info.get("maxInputChannels", 0)),
                            "default_samplerate": float(info.get("defaultSampleRate", 0)),
                        }
                    )
            pa.terminate()
            return devices

    def record(self, duration_seconds: float) -> AudioFrame:
        """Record fixed-duration audio from the default input device."""
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")

        sd = self._ensure_sounddevice()
        frames = int(duration_seconds * self.sample_rate)
        data = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
        )
        sd.wait()
        return AudioFrame(samples=self._to_list(data), sample_rate=self.sample_rate)

    def stream_chunks(self, duration_seconds: float) -> Iterator[AudioFrame]:
        """Yield real-time audio chunks from microphone until duration is reached."""
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")

        sd = self._ensure_sounddevice()
        collected: Deque[list[float]] = deque()

        def callback(indata, _frames, _time, status):
            if status:
                # Status is captured by caller logs if needed.
                pass
            collected.append(self._to_list(indata))

        max_chunks = max(1, int((duration_seconds * self.sample_rate) / self.chunk_size))
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.chunk_size,
            callback=callback,
        ):
            while len(collected) < max_chunks:
                sd.sleep(10)

        while collected:
            yield AudioFrame(samples=collected.popleft(), sample_rate=self.sample_rate)
