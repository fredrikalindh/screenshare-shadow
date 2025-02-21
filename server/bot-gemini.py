#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

from service import LoggingService
import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    SystemFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class AudioAccumulator(FrameProcessor):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._audio_frames = []
        self._start_secs = 0.2  # this should match VAD start_secs (hardcoding for now)
        self._max_buffer_size_secs = 30 # might want to increase this as it limites the each user response to 30 seconds
        self._user_speaking_vad_state = False
        self._user_speaking_utterance_state = False

    async def reset(self):
        self._audio_frames = []
        self._user_speaking_vad_state = False
        self._user_speaking_utterance_state = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStoppedSpeakingFrame):
            data = b"".join(frame.audio for frame in self._audio_frames)
            logger.debug(
                f"Processing audio buffer seconds: ({len(self._audio_frames)}) ({len(data)}) {len(data) / 2 / 16000}"
            )
            self._user_speaking = False
            # context = GoogleLLMContext()
            # # todo: multimodal support ?

            # context.add_audio_frames_message(text="Audio follows", audio_frames=self._audio_frames)
            # await self.push_frame(OpenAILLMContextFrame(context=context))
       

        await self.push_frame(frame, direction)

async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_audio_passthrough=True,
                # set stop_secs to something roughly similar to the internal setting
                # of the Multimodal Live api, just to align events. This doesn't really
                # matter because we can only use the Multimodal Live API's phrase
                # endpointing, for now.
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            ),
        )

        ambiguity_llm = GeminiMultimodalLiveLLMService(
            api_key=os.getenv("GEMINI_API_KEY"),
            # voice_id="Aoede",  # Puck, Charon, Kore, Fenrir, Aoede
            # system_instruction="Talk like a pirate."
            # transcribe_user_audio=True,
            # transcribe_model_audio=True,
            # inference_on_context_initialization=False,
        )

        llm = GeminiMultimodalLiveLLMService(
            api_key=os.getenv("GEMINI_API_KEY"),
            voice_id="Aoede",  # Puck, Charon, Kore, Fenrir, Aoede
            # system_instruction="Talk like a pirate."
            transcribe_user_audio=True,
            transcribe_model_audio=True,
            # inference_on_context_initialization=False,
        )

        context = OpenAILLMContext(
            [
                {
                    "role": "user",
                    "content": "Say hello.",
                },
            ],
        )
        context_aggregator = llm.create_context_aggregator(context)

        logging_service = LoggingService()


        async def block_user_stopped_speaking(frame):
            return not isinstance(frame, UserStoppedSpeakingFrame)


        pipeline = Pipeline(
            [
                transport.input(),
                # this is probably not going to work, replace with a classifier + buffered gate
                # FunctionFilter(filter=block_user_stopped_speaking),
                context_aggregator.user(),
                logging_service,

                # ambiguity_detector,          # Now has access to both buffer and context
                # gate,                        # Gate opens when ambiguity resolved
                llm,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            # Enable both camera and screenshare. From the client side
            # send just one.
            await transport.capture_participant_video(
                participant["id"], framerate=1, video_source="camera"
            )
            await transport.capture_participant_video(
                participant["id"], framerate=1, video_source="screenVideo"
            )
            await task.queue_frames([context_aggregator.user().get_context_frame()])
            await asyncio.sleep(3)
            logger.debug("Unpausing audio and video")
            llm.set_audio_input_paused(False)
            llm.set_video_input_paused(False)

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())