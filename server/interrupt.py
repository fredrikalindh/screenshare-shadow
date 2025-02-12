from pipecat.pipeline import PipelineStage, PipelineFrame
from typing import List, Any

class InterruptChecker(PipelineStage):
    """
    A custom pipeline stage that decides whether the bot should ask a clarifying question.
    Output is either:
      - A new 'assistant' message with a question if we want to talk
      - Or no frames if we do not want to talk
    """
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug

    async def run(self, frames: List[PipelineFrame]) -> List[PipelineFrame]:
        new_frames = []

        print(f"InterruptChecker: {frames}")
        
        for frame in frames:
            # We only care about frames that contain new user text
            # (e.g. 'transcription' events, or if you're storing them differently
            #  you can check `frame.key` or `frame.payload` accordingly)
            if frame.key == "transcription" and frame.payload:
                user_text = frame.payload.get("text", "")

                # 1) Use any logic or heuristics here:
                #    - pattern matching
                #    - small LLM-based classifier
                #    - or a hand-coded approach
                # For simplicity, let's say we interrupt if we find certain keywords
                # or if the user text is ambiguous.
                # (Replace this with your real logic.)

                if should_interrupt(user_text):
                    # 2) We'll create a new assistant message that effectively
                    #    “interrupts” the user with a question.
                    new_frames.append(
                        PipelineFrame(
                            key="assistant_message",
                            payload={
                                "role": "assistant",
                                "content": "Quick question: Could you clarify how you found that person’s name?"
                            }
                        )
                    )
                else:
                    # If we decide not to talk, we could either return no frames
                    # or we could pass along an “idle” signal.
                    # Just do nothing to remain silent.
                    pass
            else:
                # Pass along other frames unchanged
                new_frames.append(frame)

        return new_frames


def should_interrupt(user_text: str) -> bool:
    """
    Example function that decides whether or not to interrupt
    based on the user's last statement.
    """
    # A trivial placeholder that asks for clarification whenever
    # user mentions 'LinkedIn' or 'search' but doesn't mention 'why'.
    # You can get fancy here with a small LLM or more advanced logic.
    triggers = ["linkedin", "search", "confused", "not sure", "no idea"]
    if any(trigger in user_text.lower() for trigger in triggers):
        return True
    return False
