"""Speech-to-text backends. faster-whisper is imported lazily so tests never need it."""

from __future__ import annotations

from typing import Any, Protocol

from sales_copilot.audio import Audio


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: Audio) -> str: ...


class FasterWhisperTranscriber:
    """faster-whisper (CTranslate2) on CPU, int8 by default, greedy decoding."""

    def __init__(
        self,
        model_size: str = "tiny",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        beam_size: int = 1,
        initial_prompt: str | None = None,
    ) -> None:
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper-{model_size}-{compute_type}"
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self._model: Any = WhisperModel(
            model_size, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads
        )

    def transcribe(self, audio: Audio) -> str:
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=self.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=self.initial_prompt,
        )
        return " ".join(s.text.strip() for s in segments).strip()
