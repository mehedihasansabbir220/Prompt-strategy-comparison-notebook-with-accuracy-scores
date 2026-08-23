"""Prompt strategy implementations and their registry.

The registry is the single place that maps a :class:`~src.config.PromptStrategy`
enum member to its implementation. The runner iterates the registry rather than
importing strategies individually, so adding a seventh strategy means writing
one module and adding one line here — nothing downstream changes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from src.config import DEFAULT_STRATEGIES, PromptStrategy
from src.prompts.base import (
    Prompt,
    PromptContext,
    PromptError,
    PromptStrategyBase,
    render_example_block,
    render_review_block,
)
from src.prompts.combined import CombinedStrategy
from src.prompts.few_shot import FewShotStrategy
from src.prompts.reasoning import ReasoningAwareStrategy
from src.prompts.role_based import RoleBasedStrategy
from src.prompts.structured import StructuredOutputStrategy
from src.prompts.zero_shot import ZeroShotStrategy

#: Enum member -> implementing class. Every member of ``PromptStrategy`` must
#: appear here; a test enforces it, so the enum and the registry cannot drift.
STRATEGY_REGISTRY: Final[dict[PromptStrategy, type[PromptStrategyBase]]] = {
    PromptStrategy.ZERO_SHOT: ZeroShotStrategy,
    PromptStrategy.FEW_SHOT: FewShotStrategy,
    PromptStrategy.ROLE_BASED: RoleBasedStrategy,
    PromptStrategy.STRUCTURED: StructuredOutputStrategy,
    PromptStrategy.REASONING: ReasoningAwareStrategy,
    PromptStrategy.COMBINED: CombinedStrategy,
}


def get_strategy(name: PromptStrategy | str) -> PromptStrategyBase:
    """Instantiate one strategy by enum member or by its string value.

    Args:
        name: ``PromptStrategy.ZERO_SHOT`` or ``"zero_shot"``.

    Returns:
        A ready-to-use strategy instance.

    Raises:
        PromptError: If no strategy is registered under that name.
    """
    try:
        key = PromptStrategy(name)
    except ValueError as exc:
        known = ", ".join(str(member) for member in STRATEGY_REGISTRY)
        raise PromptError(f"Unknown strategy {name!r}. Available: {known}") from exc
    return STRATEGY_REGISTRY[key]()


def build_strategies(
    names: Iterable[PromptStrategy | str] = DEFAULT_STRATEGIES,
) -> list[PromptStrategyBase]:
    """Instantiate several strategies, preserving the given order.

    Defaults to the configured benchmark order (baseline first, combined last).
    """
    return [get_strategy(name) for name in names]


def describe_strategies(
    names: Iterable[PromptStrategy | str] = DEFAULT_STRATEGIES,
) -> list[dict[str, str]]:
    """Return each strategy's documented purpose and hypothesis.

    Rendered as a table in the notebook so the predictions are visible *before*
    any results are shown.
    """
    return [strategy.describe() for strategy in build_strategies(names)]


__all__ = [
    "STRATEGY_REGISTRY",
    "CombinedStrategy",
    "FewShotStrategy",
    "Prompt",
    "PromptContext",
    "PromptError",
    "PromptStrategyBase",
    "ReasoningAwareStrategy",
    "RoleBasedStrategy",
    "StructuredOutputStrategy",
    "ZeroShotStrategy",
    "build_strategies",
    "describe_strategies",
    "get_strategy",
    "render_example_block",
    "render_review_block",
]
