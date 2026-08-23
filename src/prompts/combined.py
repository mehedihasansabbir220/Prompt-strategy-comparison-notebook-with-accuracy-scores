"""Combined strategy: composition of the individually-motivated techniques."""

from __future__ import annotations

import json
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
from src.prompts.reasoning import ReasoningAwareStrategy
from src.prompts.role_based import RoleBasedStrategy
from src.prompts.structured import SENTIMENT_FIELD


class CombinedStrategy(PromptStrategyBase):
    """Stacks the expert role, demonstrations, decision procedure and JSON contract.

    Mechanism: composition. It reuses the *same* system instruction as
    :class:`RoleBasedStrategy` and the *same* decision procedure as
    :class:`ReasoningAwareStrategy` by importing them, rather than restating
    them. If a component is edited, this strategy changes with it and the
    comparison stays honest — a copy-pasted variant would silently drift.

    This is the one strategy whose result is not interpretable on its own: it
    measures whether the techniques *compose*, which is only meaningful once
    each component has been measured in isolation.
    """

    name: ClassVar[PromptStrategy] = PromptStrategy.COMBINED
    description: ClassVar[str] = (
        "Expert role (system slot) + balanced demonstrations + explicit decision "
        "procedure + strict JSON output, reusing the components verbatim."
    )
    hypothesis: ClassVar[str] = (
        "If the mechanisms are complementary this should be the strongest "
        "strategy. If they interfere — a longer prompt diluting each instruction "
        "— it may underperform its own best component, which is the informative "
        "outcome either way."
    )
    requires_examples: ClassVar[bool] = True

    #: Reused, not restated: single source of truth for each component.
    SYSTEM_INSTRUCTION: ClassVar[str] = RoleBasedStrategy.SYSTEM_INSTRUCTION
    DECISION_PROCEDURE: ClassVar[tuple[str, ...]] = (
        ReasoningAwareStrategy.DECISION_PROCEDURE
    )

    def _build(self, review: str, context: PromptContext) -> Prompt:
        examples = "\n\n".join(
            render_example_block(example) for example in context.few_shot_examples
        )
        steps = "\n".join(
            f"{index}. {step}" for index, step in enumerate(self.DECISION_PROCEDURE, 1)
        )
        schema = json.dumps({SENTIMENT_FIELD: f"<{'|'.join(context.labels)}>"}, indent=2)
        example_response = json.dumps({SENTIMENT_FIELD: context.labels[-1]})

        user_text = "\n".join(
            [
                "Annotate the sentiment of a movie review, following the "
                "guideline and returning JSON.",
                "",
                "Previously annotated examples:",
                "",
                examples,
                "",
                "Apply these considerations silently before deciding:",
                steps,
                "",
                "Now annotate this review:",
                "",
                render_review_block(review),
                "",
                INJECTION_GUARD,
                "",
                "Required response format:",
                schema,
                "",
                "Rules:",
                f'- "{SENTIMENT_FIELD}" must be exactly {context.label_options}.',
                "- Return the JSON object and nothing else.",
                "- Do not wrap it in markdown code fences.",
                "- Do not write out your reasoning or any explanation.",
                "",
                f"Example of a valid response: {example_response}",
            ]
        )
        return Prompt(user_text=user_text, system_instruction=self.SYSTEM_INSTRUCTION)
