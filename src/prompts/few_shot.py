"""Few-shot strategy: in-context learning from fixed demonstrations."""

from __future__ import annotations

from typing import ClassVar

from src.config import PromptStrategy
from src.prompts.base import (
    INJECTION_GUARD,
    Prompt,
    PromptContext,
    PromptStrategyBase,
    render_example_block,
    render_review_block,
)


class FewShotStrategy(PromptStrategyBase):
    """Prepends solved examples so the model infers the task from instances.

    Mechanism: in-context learning. The demonstrations are class-balanced and
    alternate labels (see ``create_few_shot_examples``), so the pattern the
    model picks up is the task itself, not a majority-class or recency bias.

    The demonstrations come from the *train* split and exclude every evaluation
    sample, so no graded review is ever visible inside a prompt.
    """

    name: ClassVar[PromptStrategy] = PromptStrategy.FEW_SHOT
    description: ClassVar[str] = (
        "Fixed, class-balanced solved examples shown before the target review, "
        "demonstrating both the decision and the exact answer format."
    )
    hypothesis: ClassVar[str] = (
        "Demonstrations anchor the label space and the response format, so the "
        "unparseable rate should drop relative to zero-shot. Accuracy gains "
        "depend on whether the examples resemble the reviews being graded."
    )
    requires_examples: ClassVar[bool] = True

    def _build(self, review: str, context: PromptContext) -> Prompt:
        examples = "\n\n".join(
            render_example_block(example) for example in context.few_shot_examples
        )
        user_text = "\n".join(
            [
                "Classify the sentiment of a movie review.",
                "",
                "Worked examples:",
                "",
                examples,
                "",
                "Now classify this review:",
                "",
                render_review_block(review),
                "",
                INJECTION_GUARD,
                f"Respond with exactly one word: {context.label_options}.",
            ]
        )
        return Prompt(user_text=user_text)
