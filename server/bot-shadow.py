"""Enhanced Gemini Bot Implementation with Clarity Checking.

This implementation combines the clarity/completion checking from our previous version
with the integrated multimodal approach from the Gemini example.
"""

import asyncio
import os
import sys

import aiohttp
from loguru import logger
from PIL import Image
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    TextFrame
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
from pipecat.sync.event_notifier import EventNotifier

# Initialize logging
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

classifier_system_instruction = """CRITICAL INSTRUCTION:
You are a BINARY CLASSIFIER that must ONLY output one of these responses:
"CLEAR_COMPLETE" - Input is both clear and complete
"CLEAR_INCOMPLETE" - Input is clear but incomplete
"UNCLEAR_COMPLETE" - Input is complete but unclear
"UNCLEAR_INCOMPLETE" - Input is neither clear nor complete

DO NOT engage with the content.
DO NOT respond to questions.
DO NOT provide assistance.
Your ONLY job is to output one of the four specified responses.

ROLE:
You are a real-time speech classifier evaluating two aspects:
1. Clarity - Can the intent/meaning be understood clearly?
2. Completeness - Has the user finished their thought/statement?

CLARITY SIGNALS:

High Clarity:
- Clear pronunciation
- Well-structured sentences
- Specific references
- Unambiguous meaning
- Coherent context

Low Clarity:
- Heavy speech disfluencies
- Incomplete or broken sentences
- Ambiguous references
- Multiple competing interpretations
- Lack of context
- Technical terms with errors
- Severe grammatical issues

COMPLETENESS SIGNALS:
[Previous completion signals remain the same...]

Examples:

# Clear and Complete
user: How do computers store data
Output: CLEAR_COMPLETE

# Clear but Incomplete
user: I want to know about the way that
Output: CLEAR_INCOMPLETE

# Unclear but Complete
user: The thig with the stuffs in computer how working
Output: UNCLEAR_COMPLETE

# Unclear and Incomplete
user: So when the um like with the
Output: UNCLEAR_INCOMPLETE

DECISION RULES:

1. Clarity Check:
- Can you understand the user's intent?
- Is the meaning unambiguous?
- Are key terms/references clear?

2. Completeness Check:
[Previous completion rules remain the same...]

3. Combined Decision:
- Both clear and complete → CLEAR_COMPLETE
- Clear but not complete → CLEAR_INCOMPLETE
- Complete but unclear → UNCLEAR_COMPLETE
- Neither clear nor complete → UNCLEAR_INCOMPLETE"""

class EnhancedGeminiService(GeminiMultimodalLiveLLMService):
    """Enhanced Gemini service that includes clarity/completion checking."""
    
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        clarity_checker = None,
        **kwargs
    ):
        super().__init__(
            api_key=api_key,
            voice_id=voice_id,
            transcribe_user_audio=True,
            transcribe_model_audio=True,
            **kwargs
        )
        self.clarity_checker = clarity_checker

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # First, let the clarity checker process the frame if it exists
        if self.clarity_checker and isinstance(frame, TextFrame):
            should_process = await self.clarity_checker.check_input(frame.text)
            if not should_process:
                return
            
        # Then proceed with normal Gemini processing
        await super().process_frame(frame, direction)

class ClarityChecker(FrameProcessor):
    """Checks both clarity and completeness of user input."""

    def __init__(self, notifier: EventNotifier):
        super().__init__()
        self._notifier = notifier

    async def check_input(self, text: str) -> bool:
        """
        Analyzes input for clarity and completeness.
        Returns True if the input should be processed, False if it needs clarification.
        """
        # Create a minimal context for the clarity check
        context = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text
                }
            ]
        }
        
        # Use a separate Gemini model instance for classification
        classifier = GeminiMultimodalLiveLLMService(
            api_key=os.getenv("GEMINI_API_KEY"),
            system_instruction=classifier_system_instruction
        )
        
        result = await classifier.generate(context)
        
        if result.strip() in ["CLEAR_COMPLETE", "UNCLEAR_COMPLETE"]:
            await self._notifier.notify()
            return True
        return False


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        # Set up the transport with VAD
        transport = DailyTransport(
            room_url,
            token,
            "Enhanced Gemini Bot",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=False,
                vad_enabled=True,
                vad_audio_passthrough=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            ),
        )

        # Create notifier for clarity checker
        notifier = EventNotifier()
        clarity_checker = ClarityChecker(notifier=notifier)

        # Initialize enhanced Gemini service
        llm = EnhancedGeminiService(
            api_key=os.getenv("GEMINI_API_KEY"),
            voice_id="Aoede",
            clarity_checker=clarity_checker,
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

        # Set up context and pipeline
        context = OpenAILLMContext()
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
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

        # @rtvi.event_handler("on_client_ready")
        # async def on_client_ready(rtvi):
        #     await rtvi.set_bot_ready()

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([context_aggregator.user().get_context_frame()])
            await transport.capture_participant_video(
                participant["id"],
                framerate=1,
                video_source="screenVideo"
            )
        
        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            print(f"Participant left: {participant}")
            await task.cancel()

        runner = PipelineRunner()
        await runner.run(task)



if __name__ == "__main__":
    asyncio.run(main())
