"""Zero-shot strategy: the experimental baseline."""

from __future__ import annotations

from typing import ClassVar

from src.config import PromptStrategy
from src.prompts.base import (
    INJECTION_GUARD,
    Prompt,
    PromptContext,
    PromptStrategyBase,
    render_review_block,
)


class ZeroShotStrategy(PromptStrategyBase):
    """Plain task instruction, no demonstrations, no persona, no format contract.

    Mechanism: the model receives only the task, the label vocabulary and the
    review. Nothing helps it beyond its own pretraining.

    This is the control condition. Every other strategy adds exactly one
    mechanism on top of this text, so any measured difference is attributable
    to that mechanism.
    """

    name: ClassVar[PromptStrategy] = PromptStrategy.ZERO_SHOT
    description: ClassVar[str] = (
        "Direct instruction with no demonstrations, persona or output schema. "
        "The minimum prompt that still states the task unambiguously."
    )
    hypothesis: ClassVar[str] = (
        "Establishes the performance floor. Expected to be competent on clearly "
        "polarised reviews but weakest on mixed sentiment, and most prone to "
        "answering in a sentence instead of a label."
    )

    def _build(self, review: str, context: PromptContext) -> Prompt:
        user_text = "\n".join(
            [
                "Classify the sentiment of the following movie review.",
                "",
                render_review_block(review),
                "",
                INJECTION_GUARD,
                f"Respond with exactly one word: {context.label_options}.",
            ]
        )
        return Prompt(user_text=user_text)
