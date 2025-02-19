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
    Frame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from service import LoggingService

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


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

        # Set up Daily transport with specific audio/video parameters for Gemini
        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=False,
                camera_out_width=1024,
                camera_out_height=576,
                vad_enabled=True,
                vad_audio_passthrough=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            ),
        )

        # Initialize the Gemini Multimodal Live model
        llm = GeminiMultimodalLiveLLMService(
            api_key=os.getenv("GEMINI_API_KEY"),
            voice_id="Aoede",  # Puck, Aoede, Charon, Fenrir, Kore, Puck
            transcribe_user_audio=True,
            transcribe_model_audio=True,
            system_instruction="""You are a helpful assistant that shadows the user's actions and explains them in detail. Your primary goal is to understand the user's task by watching their screen and listening carefully as they describe their process. When the user's explanation or on-screen activity is unclear or when additional details would be helpful, you ask concise, clarifying questions. 

Keep in mind:
- The user will demonstrate a task they need help with by both explaining it verbally and sharing their screen.
- Your role is to observe, note details, and gently interrupt when necessary to gain further clarity about what they are doing and why.
- If the user’s verbal description does not match the on-screen activity, prompt them to clarify or show the relevant details.
- Ask follow-up questions at natural pauses or when you detect ambiguity.

For example, consider the following interactions:

1. **Clarifying the Task’s Purpose:**
   - **User:** "I’d like to prepare a list of everyone attending the same event as me and what they are working on."
   - **You (Assistant):** "That sounds interesting. What are you planning to use this list for?"

2. **Digging into the Process:**
   - **User:** "I'm going to export the attendee list to a Google sheet, and for those with social links, I'll paste the Instagram or X link here. For others, I'll search on LinkedIn."
   - **You (Assistant):** "Could you walk me through how you decide which link to use and what steps you follow if an attendee doesn't have a social link?"

3. **Handling Missing Information:**
   - **User:** "I skipped Abraham because he didn’t have a full name so I wouldn’t be able to find him."
   - **You (Assistant):** "I noticed you skipped Abraham. Can you explain why?"

4. **Ensuring Visual-Action Consistency:**
   - **User:** "Now I’m searching on LinkedIn to find the correct person."
   - **(On-screen, the user’s screen does not yet show the LinkedIn search results.)**
   - **You (Assistant):** "I don’t see the LinkedIn page on your screen yet—could you please bring it up so I can follow along?"

Remember to be brief, clear, and friendly in your questions. Your interjections should always help the user clarify their process without disrupting their flow.

If you don't have any question, just fill words like "Yes", "I see", or "Ok".

Begin the conversation by saying "Hi, I'm here to help you with your task. I'll watch your screen and listen to your explanation. If I don't understand something, I'll ask you to clarify. Could you start by telling me about the task?."
"""
        )

        messages = [
            {
                "role": "user",
                "content": (
                    # "Start by saying the following: 'I'm here to help you with your task. I'll watch your screen and listen to your explanation. If I don't understand something, I'll ask you to clarify. Could you start by telling me about the task?."
                    "Hi"
                ),
            },
        ]

        # Set up conversation context and management
        # The context_aggregator will automatically collect conversation context
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        #
        # RTVI events for Pipecat client UI
        #
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        # Insert our custom stage
        # interrupt_checker = InterruptChecker()
        # logging_service = LoggingService()

        pipeline = Pipeline(
            [
                transport.input(),            # 1) user frames in
                rtvi,                         # 2) UI event handling
                context_aggregator.user(),    # 3) collects user messages
                # logging_service,
                llm,                          # 5) only runs if there's an assistant msg
                transport.output(),           # 7) produce final audio output
                context_aggregator.assistant()# 8) store assistant messages
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                observers=[RTVIObserver(rtvi)],
            ),
        )

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            await rtvi.set_bot_ready()

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([context_aggregator.user().get_context_frame()])
            await transport.capture_participant_video(
                participant["id"], framerate=1, video_source="screenVideo"
            )

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            print(f"Participant left: {participant}")
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
