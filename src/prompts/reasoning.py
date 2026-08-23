"""Reasoning-aware strategy: an internal decision procedure, label-only output."""

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


class ReasoningAwareStrategy(PromptStrategyBase):
    """Supplies an explicit decision procedure, then demands only the verdict.

    Mechanism: a checklist naming the specific traps in film reviews — mixed
    sentiment, sarcasm, plot summary mistaken for opinion, praise that precedes
    a rejection — so the model applies a consistent procedure rather than an
    unstructured impression.

    Chain-of-thought policy: the model is asked to weigh the evidence
    *internally* and emit only the final label. No reasoning is requested,
    returned or stored, so nothing in ``predictions.csv`` contains hidden
    thought — only the verdict and the raw response.
    """

    name: ClassVar[PromptStrategy] = PromptStrategy.REASONING
    description: ClassVar[str] = (
        "A four-point decision procedure targeting mixed sentiment, sarcasm and "
        "plot-versus-opinion confusion, with the answer restricted to the label."
    )
    hypothesis: ClassVar[str] = (
        "Should help most on genuinely mixed reviews, where the procedure "
        "supplies a tie-break rule. Because no reasoning is emitted, any gain "
        "must come from the instruction itself, not from extra output tokens."
    )

    DECISION_PROCEDURE: ClassVar[tuple[str, ...]] = (
        "Identify the reviewer's overall verdict on the film, separating it from "
        "descriptions of the plot and from the emotions of the characters.",
        "Weigh praise against criticism. Many reviews contain both; what decides "
        "the label is which one the reviewer's conclusion rests on.",
        "Interpret sarcasm, irony and rhetorical questions as the opinion they "
        "imply rather than their literal wording.",
        "Where the review is genuinely balanced, decide on the closing verdict — "
        "the recommendation the reviewer leaves the reader with.",
    )

    def _build(self, review: str, context: PromptContext) -> Prompt:
        steps = "\n".join(
            f"{index}. {step}" for index, step in enumerate(self.DECISION_PROCEDURE, 1)
        )
        user_text = "\n".join(
            [
                "Classify the sentiment of the movie review below.",
                "",
                "Work through these considerations silently, in your head, before "
                "deciding:",
                steps,
                "",
                render_review_block(review),
                "",
                INJECTION_GUARD,
                "",
                "Do not write out your reasoning, your considerations or any "
                "explanation.",
                f"Your entire response must be exactly one word: {context.label_options}.",
            ]
        )
        return Prompt(user_text=user_text)
