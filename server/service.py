from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional, Tuple

from loguru import logger

from pipecat.frames.frames import (
    DataFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TextFrame,
    AudioRawFrame,
STTUpdateSettingsFrame,
STTMuteFrame
)
from pipecat.metrics.metrics import MetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

class LoggingService(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name: str = ""

    @property
    def model_name(self) -> str:
        return self._model_name

    def set_model_name(self, model: str):
        self._model_name = model
        self.set_core_metrics_data(MetricsData(processor=self.name, model=self._model_name))

    async def process_audio_frame(self, frame: AudioRawFrame):
        if not self._muted:
            await self.process_generator(self.run_stt(frame.audio))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes a frame of audio data, either buffering or transcribing it."""
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            logger.debug(f"TextFrame: {frame.text}")
        elif isinstance(frame, AudioRawFrame):
            logger.debug(f"AudioRawFrame")

        await self.push_frame(frame, direction)
    # async def process_frame(self, frame: Frame, direction: FrameDirection):
    #     """Processes a frame of audio data, either buffering or transcribing it."""
    #     await super().process_frame(frame, direction)

    #     if isinstance(frame, AudioRawFrame):
    #         # In this service we accumulate audio internally and at the end we
    #         # push a TextFrame. We also push audio downstream in case someone
    #         # else needs it.
    #         await self.process_audio_frame(frame)
    #         if self._audio_passthrough:
    #             await self.push_frame(frame, direction)
    #     elif isinstance(frame, STTUpdateSettingsFrame):
    #         await self._update_settings(frame.settings)
    #     elif isinstance(frame, STTMuteFrame):
    #         self._muted = frame.mute
    #         logger.debug(f"STT service {'muted' if frame.mute else 'unmuted'}")
    #     else:
    #         await self.push_frame(frame, direction)

    async def _process_text_frame(self, frame: TextFrame):
        text: Optional[str] = None
        if not self._aggregate_sentences:
            text = frame.text
        else:
            self._current_sentence += frame.text
            eos_end_marker = match_endofsentence(self._current_sentence)
            if eos_end_marker:
                text = self._current_sentence[:eos_end_marker]
                self._current_sentence = self._current_sentence[eos_end_marker:]

        if text:
            await self._push_tts_frames(text)

    async def process_generator(self, generator: AsyncGenerator[Frame | None, None]):
        async for f in generator:
            if f:
                if isinstance(f, ErrorFrame):
                    await self.push_error(f)
                else:
                    await self.push_frame(f)