"""Optional voice-to-prompt helper for live robot inference.

This module is intentionally optional: importing it should not require audio or
speech-to-text packages. Dependencies are checked only when voice input is
started.
"""

from __future__ import annotations

import queue
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class VoiceEvent:
    """Result emitted by the background voice worker."""

    kind: str
    text: str = ""


class VoicePromptListener:
    """Push-to-talk recorder + background STT transcriber.

    The UI thread calls :meth:`toggle_recording` when the operator presses the
    voice key. Recording starts/stops immediately; transcription runs on a
    daemon thread and calls ``on_text`` when text is ready.
    """

    def __init__(
        self,
        on_text: Callable[[str], None],
        model_name: str = "base.en",
        sample_rate: int = 44100,
        device: int | str | None = None,
        backend: str = "auto",
    ):
        self.on_text = on_text
        self.model_name = model_name
        self.sample_rate = int(sample_rate)
        self.device = device
        self.backend = backend

        self._sd = None
        self._stream = None
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._recording = False
        self._closed = False
        self._events: queue.Queue[VoiceEvent] = queue.Queue()
        self._model = None
        self._model_backend: str | None = None

    @property
    def recording(self) -> bool:
        return self._recording

    def poll_event(self) -> VoiceEvent | None:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def start_recording(self):
        if self._recording:
            return
        self._ensure_sounddevice()
        with self._lock:
            self._frames = []
            self._recording = True

        def callback(indata, frames, time, status):  # noqa: ARG001
            if status:
                self._events.put(VoiceEvent("warn", str(status)))
            with self._lock:
                if self._recording:
                    self._frames.append(indata.copy())

        self._stream = self._sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=callback,
        )
        self._stream.start()
        self._events.put(VoiceEvent("recording_started"))

    def stop_recording(self):
        if not self._recording:
            return
        with self._lock:
            self._recording = False
            frames = list(self._frames)
            self._frames = []

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not frames:
            self._events.put(VoiceEvent("warn", "No audio captured"))
            return

        audio = np.concatenate(frames, axis=0).reshape(-1)
        threading.Thread(
            target=self._transcribe_worker,
            args=(audio,),
            name="VoicePromptTranscriber",
            daemon=True,
        ).start()
        self._events.put(VoiceEvent("recording_stopped"))

    def toggle_recording(self):
        if self._recording:
            self.stop_recording()
        else:
            self.start_recording()

    def close(self):
        self._closed = True
        if self._recording:
            self.stop_recording()
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _ensure_sounddevice(self):
        if self._sd is not None:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Voice input requires sounddevice. Install with: pip install sounddevice"
            ) from exc
        self._sd = sd

    def _load_model(self):
        if self._model is not None:
            return self._model_backend, self._model

        if self.backend in {"auto", "faster-whisper"}:
            try:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(self.model_name, device="auto", compute_type="auto")
                self._model_backend = "faster-whisper"
                return self._model_backend, self._model
            except ImportError:
                if self.backend == "faster-whisper":
                    raise RuntimeError(
                        "Voice input backend faster-whisper is not installed. "
                        "Install with: pip install faster-whisper"
                    )

        if self.backend in {"auto", "whisper"}:
            try:
                import whisper

                self._model = whisper.load_model(self.model_name)
                self._model_backend = "whisper"
                return self._model_backend, self._model
            except ImportError as exc:
                raise RuntimeError(
                    "Voice input requires faster-whisper or openai-whisper. Install one of: "
                    "pip install faster-whisper  OR  pip install openai-whisper"
                ) from exc

        raise RuntimeError(f"Unsupported voice backend: {self.backend}")

    def _transcribe_worker(self, audio: np.ndarray):
        try:
            self._events.put(VoiceEvent("transcribing"))
            wav_path = self._write_wav(audio)
            backend, model = self._load_model()
            if backend == "faster-whisper":
                segments, _info = model.transcribe(str(wav_path), beam_size=1, vad_filter=True)
                text = " ".join(seg.text.strip() for seg in segments).strip()
            else:
                result = model.transcribe(str(wav_path), fp16=False)
                text = str(result.get("text", "")).strip()

            if text:
                self._events.put(VoiceEvent("text", text))
                self.on_text(text)
            else:
                self._events.put(VoiceEvent("warn", "No speech recognized"))
        except Exception as exc:  # keep robot loop alive
            self._events.put(VoiceEvent("error", str(exc)))

    def _write_wav(self, audio: np.ndarray) -> Path:
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(prefix="voice_prompt_", suffix=".wav", delete=False)
        tmp.close()
        path = Path(tmp.name)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        return path
