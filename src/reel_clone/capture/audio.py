from __future__ import annotations

import threading
from collections import deque
import numpy as np
from numpy.typing import NDArray


class WasapiLoopbackRecorder:
    """
    Non-blocking system audio loopback recorder for Windows (PyAudioWPatch).
    Maintains a rolling ring buffer resampled to 16 kHz mono float32.
    """

    TARGET_SR: int = 16000

    def __init__(self, window_sec: float = 0.5) -> None:
        self.window_sec = window_sec
        self.target_samples = int(self.TARGET_SR * self.window_sec)
        self._buffer: deque[float] = deque(maxlen=self.target_samples * 4)
        self._lock = threading.Lock()
        self._pa = None
        self._stream = None
        self._device_sr: int = self.TARGET_SR
        self._channels: int = 1
        self._running: bool = False

    def start(self) -> None:
        """Attempts to open default Windows WASAPI loopback device; falls back to silent mock if unavailable."""
        if self._running:
            return
        try:
            import pyaudiowpatch as pyaudio  # type: ignore[import-untyped]

            self._pa = pyaudio.PyAudio()
            wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = self._pa.get_device_info_by_index(
                wasapi_info["defaultOutputDevice"]
            )

            if not default_speakers.get("isLoopbackDevice", False):
                for loopback in self._pa.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        default_speakers = loopback
                        break

            self._device_sr = int(default_speakers["defaultSampleRate"])
            self._channels = int(default_speakers["maxInputChannels"])

            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self._channels,
                rate=self._device_sr,
                frames_per_buffer=1024,
                input=True,
                input_device_index=default_speakers["index"],
                stream_callback=self._audio_callback,
            )
            self._stream.start_stream()
            self._running = True
        except Exception:
            # Fallback mode (e.g., CI test environment or no loopback audio device active)
            self._running = True

    def _audio_callback(self, in_data: bytes, frame_count: int, time_info: dict, status: int) -> tuple[None, int]:
        pcm16 = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        if self._channels > 1:
            pcm16 = pcm16.reshape(-1, self._channels).mean(axis=1)

        # Lightweight linear decimation/interpolation to 16 kHz
        if self._device_sr != self.TARGET_SR and len(pcm16) > 1:
            num_out = max(1, int(len(pcm16) * self.TARGET_SR / self._device_sr))
            x_old = np.linspace(0.0, 1.0, num=len(pcm16), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=num_out, endpoint=False)
            pcm16 = np.interp(x_new, x_old, pcm16).astype(np.float32)

        with self._lock:
            self._buffer.extend(pcm16.tolist())

        return (None, 0)  # paContinue == 0

    def get_latest_window(self) -> NDArray[np.float32]:
        """Returns exactly `target_samples` (e.g. 8000 samples for 0.5s at 16kHz) as float32."""
        with self._lock:
            data = list(self._buffer)

        if len(data) >= self.target_samples:
            window = np.array(data[-self.target_samples :], dtype=np.float32)
        else:
            pad_len = self.target_samples - len(data)
            window = np.pad(
                np.array(data, dtype=np.float32),
                (pad_len, 0),
                mode="constant",
                constant_values=0.0,
            )
        return window

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None