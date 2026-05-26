"""
VoiceTranscriber — live microphone capture and speech-to-text for technician input.

Captures audio from the default microphone in fixed-duration chunks, sends each
chunk to the ElevenLabs Scribe v2 API, and accumulates the returned text segments
into a full session transcript. Designed for hands-free technician note entry.
"""

import io
import os
import threading
import wave

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()


class VoiceTranscriber:
    """
    Records live microphone audio in chunks and transcribes each chunk via
    the ElevenLabs speech-to-text API, printing segments in real time.
    """

    def __init__(self) -> None:
        """Load credentials, configure audio parameters, and initialise state."""
        self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        self.sample_rate: int = 16000
        self.channels: int = 1
        self.chunk_duration_seconds: int = 5
        self.recording: bool = False
        self.transcript_segments: list[str] = []

        self._thread: threading.Thread | None = None
        self._audio_buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────────

    def start(self, prompt_label: str = "Technician Notes") -> None:
        """
        Begin recording from the default microphone.

        Prints a prompt, then launches a background thread that records audio
        in chunks of self.chunk_duration_seconds seconds, transcribes each
        chunk via ElevenLabs, and prints the result as it arrives.

        Args:
            prompt_label: Label shown in the console prompt.

        Raises:
            RuntimeError: If no microphone device is available.
        """
        self._check_microphone()

        print(f"\n🎙  {prompt_label} — speak now. Press ENTER to stop.")
        self.recording = True
        self.transcript_segments = []

        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop(self) -> str:
        """
        Stop recording and return the full session transcript.

        Signals the recording loop to stop, waits for the background thread
        to finish processing any remaining audio, then returns all transcribed
        segments joined into a single string.

        Returns:
            Full transcript as a single whitespace-joined string.
        """
        self.recording = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        return " ".join(self.transcript_segments).strip()

    def capture(self, prompt_label: str = "Technician Notes") -> str:
        """
        Record until the user presses ENTER, then return the full transcript.

        Convenience method: calls start(), blocks the main thread waiting for
        ENTER, then calls stop() and returns the result. This is the primary
        entry point used by ro_agent.py in interactive mode.

        Args:
            prompt_label: Label shown in the console prompt.

        Returns:
            Full transcript as a single whitespace-joined string.
        """
        self.start(prompt_label)
        input()
        return self.stop()

    def transcribe_file(self, audio_file_path: str) -> str:
        """
        Transcribe an existing audio file and return the text.

        Sends the file directly to ElevenLabs speech-to-text without any
        microphone capture. Useful for testing without live microphone input.

        Args:
            audio_file_path: Path to a .mp3, .wav, or .m4a file.

        Returns:
            Transcribed text string, or empty string on error.
        """
        try:
            with open(audio_file_path, "rb") as f:
                file_bytes = f.read()
            transcript = self.client.speech_to_text.convert(
                file=io.BytesIO(file_bytes),
                model_id="scribe_v2",
                tag_audio_events=True,
                language_code="eng",
                diarize=False,
            )
            return transcript.text
        except Exception as exc:
            print(f"  [ElevenLabs error] {exc}")
            return ""

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers

    def _check_microphone(self) -> None:
        """Raise RuntimeError if no input device is accessible."""
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d["max_input_channels"] > 0]
            if not input_devices:
                raise RuntimeError("Microphone not found. Check audio device and try again.")
        except sd.PortAudioError as exc:
            raise RuntimeError("Microphone not found. Check audio device and try again.") from exc

    def _record_loop(self) -> None:
        """
        Background thread: record audio in fixed chunks and transcribe each one.

        Accumulates frames from the sounddevice InputStream callback, slices
        them into chunk_duration_seconds-length blocks, converts each to WAV
        bytes, and sends them to ElevenLabs. Any remaining audio after
        self.recording is set to False is transcribed before the thread exits.
        """
        frames_per_chunk = self.sample_rate * self.chunk_duration_seconds
        accumulated: list[np.ndarray] = []

        def _callback(indata: np.ndarray, frames: int, time, status) -> None:
            accumulated.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                callback=_callback,
            ):
                while self.recording:
                    sd.sleep(200)
                    total_frames = sum(a.shape[0] for a in accumulated)
                    if total_frames >= frames_per_chunk:
                        chunk_data, accumulated = self._split_chunk(
                            accumulated, frames_per_chunk
                        )
                        self._transcribe_chunk(chunk_data)

                # Final chunk after stop() was called
                if accumulated:
                    chunk_data = np.concatenate(accumulated, axis=0)
                    self._transcribe_chunk(chunk_data)

        except sd.PortAudioError as exc:
            print(f"  [Audio error] {exc}")

    def _split_chunk(
        self,
        accumulated: list[np.ndarray],
        frames_per_chunk: int,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Consume exactly frames_per_chunk frames from accumulated and return the
        remainder as a new list.
        """
        full = np.concatenate(accumulated, axis=0)
        chunk = full[:frames_per_chunk]
        remainder = full[frames_per_chunk:]
        return chunk, [remainder] if len(remainder) > 0 else []

    def _transcribe_chunk(self, audio_array: np.ndarray) -> None:
        """Convert a numpy audio array to WAV bytes and send to ElevenLabs."""
        if audio_array.size == 0:
            return

        wav_bytes = self._array_to_wav_bytes(audio_array)
        try:
            transcript = self.client.speech_to_text.convert(
                file=io.BytesIO(wav_bytes),
                model_id="scribe_v2",
                tag_audio_events=True,
                language_code="eng",
                diarize=False,
            )
            text = transcript.text.strip()
            if text:
                self.transcript_segments.append(text)
                print(f"  → {text}")
            else:
                self.transcript_segments.append("")
        except Exception as exc:
            print(f"  [ElevenLabs error] {exc}")
            self.transcript_segments.append("")

    def _array_to_wav_bytes(self, audio_array: np.ndarray) -> bytes:
        """Encode a 16-bit mono numpy array as WAV bytes in memory."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit = 2 bytes per sample
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_array.tobytes())
        return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transcriber = VoiceTranscriber()
    result = transcriber.capture("Test — describe the vehicle fault")
    print("\nFULL TRANSCRIPT:")
    print(result)
