"""Role-based strategy: expert persona plus an explicit annotation guideline."""

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


class RoleBasedStrategy(PromptStrategyBase):
    """Assigns a professional annotator role in the system instruction.

    Mechanism: the persona and its working rules occupy the *system* slot rather
    than the user turn — a structural difference from the baseline, not a
    reworded one. The role is paired with the guideline such an annotator would
    actually follow, so "expert" carries operational meaning instead of being
    flattery.
    """

    name: ClassVar[PromptStrategy] = PromptStrategy.ROLE_BASED
    description: ClassVar[str] = (
        "A professional sentiment-annotator persona, delivered as a system "
        "instruction together with the annotation guideline it works to."
    )
    hypothesis: ClassVar[str] = (
        "Framing the task as professional annotation against a stated guideline "
        "should sharpen decisions on ambiguous reviews, where the guideline "
        "supplies a tie-break the baseline prompt lacks."
    )

    SYSTEM_INSTRUCTION: ClassVar[str] = (
        "You are a professional sentiment annotator on a film-review corpus. "
        "You have labelled tens of thousands of reviews and you apply the "
        "project's annotation guideline consistently.\n"
        "\n"
        "Annotation guideline:\n"
        "1. Label the reviewer's overall verdict on the film, not the mood of "
        "the plot or the fortunes of the characters.\n"
        "2. A review that criticises details but recommends the film is positive; "
        "a review that praises details but concludes against the film is negative.\n"
        "3. Read sarcasm and rhetorical questions as the opinion they imply, not "
        "as their literal wording.\n"
        "4. Every review receives one of the two available labels. You never "
        "abstain and you never invent a third category."
    )

    def _build(self, review: str, context: PromptContext) -> Prompt:
        user_text = "\n".join(
            [
                "Annotate the sentiment of this review.",
                "",
                render_review_block(review),
                "",
                INJECTION_GUARD,
                f"Respond with exactly one word: {context.label_options}.",
            ]
        )
        return Prompt(user_text=user_text, system_instruction=self.SYSTEM_INSTRUCTION)
