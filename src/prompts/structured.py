"""Structured-output strategy: a machine-readable response contract."""

from __future__ import annotations

import json
from typing import ClassVar

from src.config import PromptStrategy
from src.prompts.base import (
    INJECTION_GUARD,
    Prompt,
    PromptContext,
    PromptStrategyBase,
    render_review_block,
)

#: Key the model must emit. Shared with the parser so prompt and parser cannot
#: drift apart.
SENTIMENT_FIELD: str = "sentiment"


class StructuredOutputStrategy(PromptStrategyBase):
    """Requires a single JSON object as the entire response.

    Mechanism: an output *contract* — a schema, a worked shape, and explicit
    prohibitions on the wrappers models habitually add (markdown fences, a
    preamble, trailing commentary).

    Note on experimental control: the contract is expressed purely in prompt
    text. Provider-side constrained decoding (``response_mime_type``, response
    schemas) would change generation settings and stop this from being a
    prompt-only comparison, so it is deliberately not used.
    """

    name: ClassVar[PromptStrategy] = PromptStrategy.STRUCTURED
    description: ClassVar[str] = (
        "Demands one JSON object with a single 'sentiment' field as the whole "
        "response, with explicit rules against fences, prose and extra keys."
    )
    hypothesis: ClassVar[str] = (
        "Should minimise unparseable responses, which raises effective accuracy "
        "even if the underlying judgement is unchanged. The format constraint "
        "itself is not expected to improve classification quality."
    )

    def _schema_block(self, context: PromptContext) -> str:
        """Render the required response shape as literal JSON."""
        options = "|".join(context.labels)
        return json.dumps({SENTIMENT_FIELD: f"<{options}>"}, indent=2)

    def _build(self, review: str, context: PromptContext) -> Prompt:
        example = json.dumps({SENTIMENT_FIELD: context.labels[-1]})
        user_text = "\n".join(
            [
                "Classify the sentiment of the movie review below and return the "
                "result as JSON.",
                "",
                render_review_block(review),
                "",
                INJECTION_GUARD,
                "",
                "Required response format:",
                self._schema_block(context),
                "",
                "Rules:",
                f'- "{SENTIMENT_FIELD}" must be exactly {context.label_options}.',
                "- Return the JSON object and nothing else.",
                "- Do not wrap it in markdown code fences.",
                "- Do not add explanations, keys, or trailing text.",
                "",
                f"Example of a valid response: {example}",
            ]
        )
        return Prompt(user_text=user_text)
