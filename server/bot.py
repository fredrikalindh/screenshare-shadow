#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Bot Implementation.

This module implements a chatbot using Google's Gemini Multimodal Live model.
It includes:
- Real-time audio/video interaction through Daily
- Animated robot avatar
- Speech-to-speech model

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow using Gemini's streaming capabilities.
"""

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    OutputImageRawFrame,
    SpriteFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

sprites = []
script_dir = os.path.dirname(__file__)

async def main():
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Daily video transport with specific audio parameters
    - Gemini Live multimodal model integration
    - Voice activity detection
    - Animation processing
    - RTVI event handling
    """

    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,       # Enable audio output
                camera_out_enabled=True,      # Enable video output
                vad_enabled=True,             # Enable voice activity detection, 0.5 sec stop time
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            ),
        )

        llm = GeminiMultimodalLiveLLMService(
            api_key=os.getenv("GEMINI_API_KEY"),
            voice_id="Aoede",                    # Puck, Aoede, Charon, Fenrir, Kore, Puck
            transcribe_user_audio=True,          # Enable speech-to-text
            transcribe_model_audio=True,         # Log bot responses
            # params=InputParams(temperature=0.7)  # Set model input params
            # system_instruction=''
        )

        messages = [{
            "role": "user",
            "content": """You are Chatbot, a friendly, helpful robot.
                        Keep responses brief and avoid special characters
                        since output will be converted to audio."""
        }]

        # Set up conversation context and management
        # The context_aggregator will automatically collect conversation context
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        #
        # RTVI events for Pipecat client UI
        #
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        pipeline = Pipeline([
            transport.input(),             # Receives audio/video from the user via WebRTC
            context_aggregator.user(),     # Manages user message history
            llm,                           # Processes speech through Gemini
            rtvi,
            transport.output(),            # Sends audio/video back to the user via WebRTC
            context_aggregator.assistant() # Process bot context
        ])

        # @rtvi.event_handler("on_client_ready")
        # async def on_client_ready(rtvi):
        #     await rtvi.set_bot_ready()
        #     for message in params.actions:
        #         await rtvi.handle_message(message)

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
            # await callbacks.on_first_participant_joined(participant)

        # @transport.event_handler("on_participant_joined")
        # async def on_participant_joined(transport, participant):
        #     await callbacks.on_participant_joined(participant)

        # @transport.event_handler("on_participant_left")
        # async def on_participant_left(transport, participant, reason):
        #     await callbacks.on_participant_left(participant, reason)

        # @transport.event_handler("on_call_state_updated")
        # async def on_call_state_updated(transport, state):
        #     await callbacks.on_call_state_updated(state)

        return pipeline