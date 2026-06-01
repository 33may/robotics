#!/usr/bin/env python3
"""Robot-free voice prompt test.

Examples:
    conda run -n lerobot python scripts/test_voice_prompt.py --list-devices
    conda run -n lerobot python scripts/test_voice_prompt.py --duration 4 --model tiny.en
    conda run -n lerobot python scripts/test_voice_prompt.py --device 0 --duration 4

Press-to-record mode:
    conda run -n lerobot python scripts/test_voice_prompt.py --model tiny.en
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from repo root without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def list_devices():
    import sounddevice as sd

    print(sd.query_devices())


def main():
    parser = argparse.ArgumentParser(description="Test microphone capture + speech-to-text prompt recognition.")
    parser.add_argument("--list-devices", action="store_true", help="List sounddevice input/output devices and exit")
    parser.add_argument("--device", default=None, help="Input device index/name, e.g. 0 or 'pipewire'")
    parser.add_argument("--duration", type=float, default=0.0, help="Record fixed seconds. If 0, press Enter to stop.")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--backend", default="auto", choices=["auto", "faster-whisper", "whisper"])
    parser.add_argument("--model", default="tiny.en", help="STT model name/path. tiny.en is fastest for testing.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Seconds to wait for transcription")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    from vbti.logic.inference.voice_prompt import VoicePromptListener

    recognized: list[str] = []

    def on_text(text: str):
        recognized.append(text)

    device = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    listener = VoicePromptListener(
        on_text=on_text,
        model_name=args.model,
        sample_rate=args.sample_rate,
        device=device,
        backend=args.backend,
    )

    print("Voice test ready.")
    print(f"backend={args.backend}, model={args.model}, sample_rate={args.sample_rate}, device={device}")
    print("Say something like: 'red cup' or 'black cup'.")

    try:
        if args.duration > 0:
            print(f"Recording for {args.duration:.1f}s...")
            listener.start_recording()
            time.sleep(args.duration)
            listener.stop_recording()
        else:
            input("Press Enter to START recording...")
            listener.start_recording()
            input("Recording. Press Enter to STOP and transcribe...")
            listener.stop_recording()

        print("Waiting for transcription...")
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            ev = listener.poll_event()
            if ev is not None:
                if ev.kind == "text":
                    print(f"TRANSCRIPT: {ev.text}")
                    return
                if ev.kind in {"warn", "error"}:
                    print(f"{ev.kind.upper()}: {ev.text}")
                else:
                    print(f"[{ev.kind}]")
            if recognized:
                print(f"TRANSCRIPT: {recognized[-1]}")
                return
            time.sleep(0.05)
        print("Timed out waiting for transcription.")
    finally:
        listener.close()


if __name__ == "__main__":
    main()
