"""Tests for the prompt strategy layer.

Two kinds of test live here:

* **contract tests** parametrized over every registered strategy, so a new
  strategy is covered the moment it is added to the registry;
* **mechanism tests** asserting that each strategy actually implements the
  distinct technique it claims — not merely different wording.

No LLM is contacted; a prompt is plain text.
"""

from __future__ import annotations

import json

import pytest

from src.config import DEFAULT_STRATEGIES, SENTIMENT_LABELS, PromptStrategy
from src.dataset import FewShotExample
from src.prompts import (
    STRATEGY_REGISTRY,
    CombinedStrategy,
    FewShotStrategy,
    Prompt,
    PromptContext,
    PromptError,
    PromptStrategyBase,
    ReasoningAwareStrategy,
    RoleBasedStrategy,
    StructuredOutputStrategy,
    ZeroShotStrategy,
    build_strategies,
    describe_strategies,
    get_strategy,
)
from src.prompts.structured import SENTIMENT_FIELD

#: The quoted JSON key. The bare word "sentiment" also occurs in ordinary
#: prose, so only the quoted form reliably marks a JSON output contract.
JSON_KEY = f'"{SENTIMENT_FIELD}"'

REVIEW = "A thoughtful film with a slow middle act, but the ending won me over."

EXAMPLES = (
    FewShotExample(review="Dull, predictable and far too long.", label="negative"),
    FewShotExample(review="Beautifully acted and genuinely moving.", label="positive"),
)

ALL_STRATEGIES = list(STRATEGY_REGISTRY.values())
STRATEGY_IDS = [str(name) for name in STRATEGY_REGISTRY]


@pytest.fixture
def context() -> PromptContext:
    return PromptContext(few_shot_examples=EXAMPLES)


# ---------------------------------------------------------------------------
# Contract: every strategy, same guarantees
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES, ids=STRATEGY_IDS)
class TestStrategyContract:
    def test_produces_a_prompt_object(
        self, strategy_class: type[PromptStrategyBase], context: PromptContext
    ) -> None:
        prompt = strategy_class().build_prompt(REVIEW, context)
        assert isinstance(prompt, Prompt)
        assert prompt.user_text.strip()

    def test_prompt_contains_the_review_verbatim(
        self, strategy_class: type[PromptStrategyBase], context: PromptContext
    ) -> None:
        prompt = strategy_class().build_prompt(REVIEW, context)
        assert REVIEW in prompt.user_text

    def test_prompt_states_both_labels(
        self, strategy_class: type[PromptStrategyBase], context: PromptContext
    ) -> None:
        text = strategy_class().build_prompt(REVIEW, context).as_text().lower()
        assert all(label in text for label in SENTIMENT_LABELS)

    def test_prompt_states_the_classification_task(
        self, strategy_class: type[PromptStrategyBase], context: PromptContext
    ) -> None:
        text = strategy_class().build_prompt(REVIEW, context).as_text().lower()
        assert "sentiment" in text or "classif" in text

    def test_prompt_constrains_the_answer_format(
        self, strategy_class: type[PromptStrategyBase], context: PromptContext
    ) -> None:
        # Either a one-word answer or a JSON contract; never an open response.
        text = strategy_class().build_prompt(REVIEW, context).as_text().lower()
        assert "exactly one word" in text or "json" in text

    def test_review_is_delimited(
        self, strategy_class: type[PromptStrategyBase], context: PromptContext
    ) -> None:
        prompt = strategy_class().build_prompt(REVIEW, context)
        assert f"<review>\n{REVIEW}\n</review>" in prompt.user_text

    def test_declares_its_identity(
        self, strategy_class: type[PromptStrategyBase]
    ) -> None:
        strategy = strategy_class()
        assert isinstance(strategy.name, PromptStrategy)
        assert len(strategy.description) > 20
        assert len(strategy.hypothesis) > 20

    def test_describe_returns_metadata(
        self, strategy_class: type[PromptStrategyBase]
    ) -> None:
        described = strategy_class().describe()
        assert set(described) == {
            "name",
            "description",
            "hypothesis",
            "requires_examples",
        }

    def test_is_deterministic(
        self, strategy_class: type[PromptStrategyBase], context: PromptContext
    ) -> None:
        first = strategy_class().build_prompt(REVIEW, context)
        second = strategy_class().build_prompt(REVIEW, context)
        assert first == second

    def test_rejects_empty_review(
        self, strategy_class: type[PromptStrategyBase], context: PromptContext
    ) -> None:
        with pytest.raises(PromptError, match="must not be empty"):
            strategy_class().build_prompt("   ", context)

    def test_rejects_non_string_review(
        self, strategy_class: type[PromptStrategyBase], context: PromptContext
    ) -> None:
        with pytest.raises(PromptError, match="must be a string"):
            strategy_class().build_prompt(None, context)  # type: ignore[arg-type]

    def test_review_is_not_truncated(
        self, strategy_class: type[PromptStrategyBase], context: PromptContext
    ) -> None:
        long_review = "This film was remarkable. " * 400
        prompt = strategy_class().build_prompt(long_review, context)
        assert long_review.strip() in prompt.user_text

    def test_prompt_is_immutable(
        self, strategy_class: type[PromptStrategyBase], context: PromptContext
    ) -> None:
        prompt = strategy_class().build_prompt(REVIEW, context)
        with pytest.raises(Exception):
            prompt.user_text = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Strategies must be genuinely different, not reworded
# ---------------------------------------------------------------------------


class TestStrategiesAreDistinct:
    def test_all_six_prompts_differ(self, context: PromptContext) -> None:
        rendered = {
            str(name): cls().build_prompt(REVIEW, context).as_text()
            for name, cls in STRATEGY_REGISTRY.items()
        }
        assert len(set(rendered.values())) == len(STRATEGY_REGISTRY)

    def test_each_prompt_carries_a_distinct_mechanism(
        self, context: PromptContext
    ) -> None:
        rendered = {
            name: cls().build_prompt(REVIEW, context)
            for name, cls in STRATEGY_REGISTRY.items()
        }
        # Demonstrations: only the few-shot family shows worked examples.
        with_examples = {
            name for name, p in rendered.items() if "<example>" in p.user_text
        }
        assert with_examples == {PromptStrategy.FEW_SHOT, PromptStrategy.COMBINED}

        # Persona: only the role family uses the system slot.
        with_persona = {
            name for name, p in rendered.items() if p.system_instruction is not None
        }
        assert with_persona == {PromptStrategy.ROLE_BASED, PromptStrategy.COMBINED}

        # Output contract: only the structured family demands JSON.
        with_json = {
            name for name, p in rendered.items() if JSON_KEY in p.user_text
        }
        assert with_json == {PromptStrategy.STRUCTURED, PromptStrategy.COMBINED}

        # Decision procedure: only the reasoning family enumerates one.
        with_procedure = {
            name for name, p in rendered.items() if "silently" in p.user_text
        }
        assert with_procedure == {PromptStrategy.REASONING, PromptStrategy.COMBINED}

    def test_zero_shot_is_the_minimal_baseline(self, context: PromptContext) -> None:
        prompt = ZeroShotStrategy().build_prompt(REVIEW, context)
        assert prompt.system_instruction is None
        assert "<example>" not in prompt.user_text
        assert JSON_KEY not in prompt.user_text
        # The baseline must be the shortest prompt of the six.
        lengths = {
            str(name): cls().build_prompt(REVIEW, context).char_count
            for name, cls in STRATEGY_REGISTRY.items()
        }
        assert lengths["zero_shot"] == min(lengths.values())


# ---------------------------------------------------------------------------
# Per-strategy mechanism tests
# ---------------------------------------------------------------------------


class TestZeroShot:
    def test_works_without_any_context(self) -> None:
        prompt = ZeroShotStrategy().build_prompt(REVIEW)
        assert "exactly one word" in prompt.user_text


class TestFewShot:
    def test_renders_every_demonstration_with_its_label(
        self, context: PromptContext
    ) -> None:
        prompt = FewShotStrategy().build_prompt(REVIEW, context)
        for example in EXAMPLES:
            assert example.review in prompt.user_text
            assert f"sentiment: {example.label}" in prompt.user_text

    def test_demonstrations_precede_the_target_review(
        self, context: PromptContext
    ) -> None:
        text = FewShotStrategy().build_prompt(REVIEW, context).user_text
        assert text.index(EXAMPLES[0].review) < text.index(REVIEW)

    def test_requires_examples(self) -> None:
        assert FewShotStrategy.requires_examples is True
        with pytest.raises(PromptError, match="requires few-shot examples"):
            FewShotStrategy().build_prompt(REVIEW)

    def test_prompt_grows_with_more_demonstrations(self) -> None:
        small = FewShotStrategy().build_prompt(
            REVIEW, PromptContext(few_shot_examples=EXAMPLES)
        )
        large = FewShotStrategy().build_prompt(
            REVIEW, PromptContext(few_shot_examples=EXAMPLES * 3)
        )
        assert large.char_count > small.char_count


class TestRoleBased:
    def test_persona_lives_in_the_system_instruction(
        self, context: PromptContext
    ) -> None:
        prompt = RoleBasedStrategy().build_prompt(REVIEW, context)
        assert prompt.system_instruction is not None
        assert "annotator" in prompt.system_instruction.lower()

    def test_system_instruction_carries_an_operational_guideline(self) -> None:
        instruction = RoleBasedStrategy.SYSTEM_INSTRUCTION
        assert "guideline" in instruction.lower()
        # A guideline, not an adjective: numbered, actionable rules.
        assert instruction.count("\n") >= 4
        for rule in ("1.", "2.", "3.", "4."):
            assert rule in instruction

    def test_forbids_abstaining(self) -> None:
        assert "never abstain" in RoleBasedStrategy.SYSTEM_INSTRUCTION.lower()


class TestStructuredOutput:
    def test_schema_is_valid_json(self, context: PromptContext) -> None:
        prompt = StructuredOutputStrategy().build_prompt(REVIEW, context)
        start = prompt.user_text.index("{", prompt.user_text.index("Required"))
        end = prompt.user_text.index("}", start) + 1
        schema = json.loads(prompt.user_text[start:end])
        assert set(schema) == {SENTIMENT_FIELD}

    def test_example_response_parses_to_a_valid_label(
        self, context: PromptContext
    ) -> None:
        text = StructuredOutputStrategy().build_prompt(REVIEW, context).user_text
        example = text.rsplit("Example of a valid response: ", 1)[1].strip()
        assert json.loads(example)[SENTIMENT_FIELD] in SENTIMENT_LABELS

    def test_forbids_markdown_fences_and_extra_text(
        self, context: PromptContext
    ) -> None:
        text = StructuredOutputStrategy().build_prompt(REVIEW, context).user_text
        assert "markdown code fences" in text
        assert "nothing else" in text

    def test_uses_no_persona_or_demonstrations(self, context: PromptContext) -> None:
        prompt = StructuredOutputStrategy().build_prompt(REVIEW, context)
        assert prompt.system_instruction is None
        assert "<example>" not in prompt.user_text


class TestReasoningAware:
    def test_enumerates_the_full_decision_procedure(
        self, context: PromptContext
    ) -> None:
        text = ReasoningAwareStrategy().build_prompt(REVIEW, context).user_text
        for index in range(1, len(ReasoningAwareStrategy.DECISION_PROCEDURE) + 1):
            assert f"{index}." in text

    def test_procedure_targets_the_known_failure_modes(self) -> None:
        joined = " ".join(ReasoningAwareStrategy.DECISION_PROCEDURE).lower()
        for trap in ("plot", "sarcasm", "praise", "verdict"):
            assert trap in joined

    def test_never_requests_visible_reasoning(self, context: PromptContext) -> None:
        text = ReasoningAwareStrategy().build_prompt(REVIEW, context).user_text.lower()
        assert "silently" in text
        assert "do not write out your reasoning" in text
        assert "step by step" not in text

    def test_answer_is_restricted_to_a_single_word(
        self, context: PromptContext
    ) -> None:
        text = ReasoningAwareStrategy().build_prompt(REVIEW, context).user_text
        assert "exactly one word" in text


class TestCombined:
    def test_stacks_all_four_mechanisms(self, context: PromptContext) -> None:
        prompt = CombinedStrategy().build_prompt(REVIEW, context)
        assert prompt.system_instruction is not None       # role
        assert "<example>" in prompt.user_text             # few-shot
        assert "silently" in prompt.user_text              # decision procedure
        assert JSON_KEY in prompt.user_text                # structured output

    def test_reuses_components_verbatim_rather_than_copying(self) -> None:
        assert CombinedStrategy.SYSTEM_INSTRUCTION == RoleBasedStrategy.SYSTEM_INSTRUCTION
        assert (
            CombinedStrategy.DECISION_PROCEDURE
            == ReasoningAwareStrategy.DECISION_PROCEDURE
        )

    def test_requires_examples(self) -> None:
        assert CombinedStrategy.requires_examples is True
        with pytest.raises(PromptError, match="requires few-shot examples"):
            CombinedStrategy().build_prompt(REVIEW)

    def test_is_the_longest_prompt(self, context: PromptContext) -> None:
        lengths = {
            str(name): cls().build_prompt(REVIEW, context).char_count
            for name, cls in STRATEGY_REGISTRY.items()
        }
        assert lengths["combined"] == max(lengths.values())


# ---------------------------------------------------------------------------
# Prompt and context value objects
# ---------------------------------------------------------------------------


class TestPromptValueObject:
    def test_as_text_includes_both_slots(self) -> None:
        prompt = Prompt(user_text="classify this", system_instruction="you are expert")
        assert "[system]" in prompt.as_text()
        assert "you are expert" in prompt.as_text()
        assert "classify this" in prompt.as_text()

    def test_as_text_omits_the_system_header_when_unused(self) -> None:
        assert Prompt(user_text="classify this").as_text() == "classify this"

    def test_char_count_matches_rendered_length(self) -> None:
        prompt = Prompt(user_text="abc", system_instruction="de")
        assert prompt.char_count == len(prompt.as_text())


class TestPromptContext:
    def test_defaults_to_the_configured_label_vocabulary(self) -> None:
        assert PromptContext().labels == SENTIMENT_LABELS

    def test_label_options_renders_both_classes(self) -> None:
        assert PromptContext().label_options == '"negative" or "positive"'

    def test_rejects_a_single_label(self) -> None:
        with pytest.raises(PromptError, match="At least two labels"):
            PromptContext(labels=("positive",))

    def test_is_immutable(self) -> None:
        with pytest.raises(Exception):
            PromptContext().labels = ("a", "b")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_covers_every_configured_strategy(self) -> None:
        assert set(STRATEGY_REGISTRY) == set(PromptStrategy)

    def test_registry_keys_match_class_identity(self) -> None:
        for key, cls in STRATEGY_REGISTRY.items():
            assert cls.name is key

    def test_get_strategy_accepts_enum_and_string(self) -> None:
        assert isinstance(get_strategy(PromptStrategy.ZERO_SHOT), ZeroShotStrategy)
        assert isinstance(get_strategy("zero_shot"), ZeroShotStrategy)

    def test_get_strategy_rejects_unknown_names(self) -> None:
        with pytest.raises(PromptError, match="Unknown strategy"):
            get_strategy("chain_of_thought")

    def test_build_strategies_preserves_configured_order(self) -> None:
        built = build_strategies()
        assert [s.name for s in built] == list(DEFAULT_STRATEGIES)

    def test_build_strategies_accepts_a_subset(self) -> None:
        built = build_strategies(["zero_shot", "combined"])
        assert [str(s.name) for s in built] == ["zero_shot", "combined"]

    def test_describe_strategies_documents_all_six(self) -> None:
        described = describe_strategies()
        assert len(described) == len(PromptStrategy)
        assert all(entry["hypothesis"] for entry in described)

    def test_strategy_names_are_unique(self) -> None:
        names = [cls.name for cls in STRATEGY_REGISTRY.values()]
        assert len(set(names)) == len(names)


class TestBaseClassEnforcement:
    def test_subclass_without_identity_is_rejected(self) -> None:
        with pytest.raises(PromptError, match="non-empty"):

            class Nameless(PromptStrategyBase):
                def _build(self, review: str, context: PromptContext) -> Prompt:
                    return Prompt(user_text=review)

    def test_prompt_losing_the_review_is_rejected(self) -> None:
        class Forgetful(PromptStrategyBase):
            name = PromptStrategy.ZERO_SHOT
            description = "drops the review entirely, on purpose"
            hypothesis = "should be caught by base-class validation"

            def _build(self, review: str, context: PromptContext) -> Prompt:
                return Prompt(user_text="classify sentiment: positive or negative")

        with pytest.raises(PromptError, match="did not include the review"):
            Forgetful().build_prompt(REVIEW)

    def test_prompt_missing_a_label_is_rejected(self) -> None:
        class HalfVocabulary(PromptStrategyBase):
            name = PromptStrategy.ZERO_SHOT
            description = "mentions only one of the two classes"
            hypothesis = "should be caught by base-class validation"

            def _build(self, review: str, context: PromptContext) -> Prompt:
                return Prompt(user_text=f"sentiment, positive? {review}")

        with pytest.raises(PromptError, match="never mentions label"):
            HalfVocabulary().build_prompt(REVIEW)
