"""Turn a raw model response into exactly one label.

The parser is shared by every strategy. That is deliberate: if each strategy
parsed its own output, a lenient parser would hand its strategy free accuracy
and the comparison would measure parser quality instead of prompt quality.

Design rule: **never guess**. A response is resolved only when the evidence is
unambiguous. Anything else returns ``unknown``, which is reported as its own
reliability metric rather than being scored as a wrong answer.

Nothing here contacts a provider; a response is plain text.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from typing import Any, Final, Literal

from src.config import LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_UNKNOWN, SENTIMENT_LABELS

logger = logging.getLogger(__name__)

#: What the parser can return. ``unknown`` is not a sentiment class.
ParsedLabel = Literal["negative", "positive", "unknown"]

#: Unambiguous abbreviations, accepted only when they are the *entire* response.
#: Never matched inside free text, where "pos" could be an artefact.
LABEL_ALIASES: Final[dict[str, str]] = {
    "pos": LABEL_POSITIVE,
    "neg": LABEL_NEGATIVE,
}

#: JSON keys accepted as carrying the verdict. ``sentiment`` is the contracted
#: key; the rest are common near-misses that are still unambiguous.
STRUCTURED_KEYS: Final[tuple[str, ...]] = (
    "sentiment",
    "label",
    "answer",
    "prediction",
    "classification",
)

#: Characters models wrap answers in: quotes (straight and curly), markdown
#: emphasis, list bullets, terminal punctuation.
DECORATION_CHARS: Final[str] = " \t\n\r\"'`*_-–—.,:;!?()[]{}“”‘’<>|/\\#"

#: Explicit answer markers, e.g. ``sentiment: positive``, ``Answer = negative``.
ANSWER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:sentiment|answer|label|classification|verdict|result|prediction)\b"
    r"\s*(?:is|are|:|=|->)+\s*"
    r"[\"'`*\s]*(" + "|".join(SENTIMENT_LABELS) + r")\b",
    re.IGNORECASE,
)

#: Any standalone mention of a label. Word boundaries stop "positively" or
#: "negativity" from counting as an answer.
LABEL_MENTION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(" + "|".join(SENTIMENT_LABELS) + r")\b", re.IGNORECASE
)

#: An enumeration of both options, e.g. ``"positive" or "negative"``. This is
#: the prompt's own instruction echoed back and carries no decision, so it is
#: removed before the text is analysed.
OPTION_ECHO_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[\"'`]?\b(?:" + "|".join(SENTIMENT_LABELS) + r")\b[\"'`]?"
    r"\s*(?:,|/|\bor\b|\band\b)\s*"
    r"[\"'`]?\b(?:" + "|".join(SENTIMENT_LABELS) + r")\b[\"'`]?",
    re.IGNORECASE,
)

#: Negation directly before a label flips or voids its meaning. Rather than
#: inferring the opposite class — which would be guessing — the mention is
#: discarded.
#: A negation counts if it is followed by at most three words before the label,
#: so "not positive" and "I would not call this positive" are both caught while
#: a negation about the *other* class, further back, is not.
NEGATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:not|isn'?t|wasn'?t|aren'?t|don'?t|doesn'?t|cannot|can'?t|no|never"
    r"|neither|nor|non|hardly|rather\s+than)\b(?:\W+\w+){0,3}\W*$",
    re.IGNORECASE,
)

#: How far back to look for a negation, in characters.
NEGATION_LOOKBEHIND: Final[int] = 40


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def normalize_label(value: Any) -> str | None:
    """Return the canonical label for ``value``, or ``None`` if it is not one.

    Strips decoration and case, then accepts an exact label or an unambiguous
    abbreviation. Used for both JSON values and whole-response matches.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip().strip(DECORATION_CHARS).strip().lower()
    if cleaned in SENTIMENT_LABELS:
        return cleaned
    return LABEL_ALIASES.get(cleaned)


def _iter_json_objects(text: str) -> Iterator[str]:
    """Yield every balanced ``{...}`` block in ``text``, outermost first.

    A brace counter is used rather than a regex because JSON nests, and the
    scanner ignores braces inside string literals so an embedded ``"}"`` cannot
    end a block early. This is what lets fenced or commented-on JSON be found
    inside otherwise chatty output.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : index + 1]


def _labels_in_payload(payload: Any) -> set[str]:
    """Collect every label found under an accepted key, at any nesting depth.

    Returns a set so a conflicting object — two keys disagreeing — can be
    detected by the caller instead of silently resolving to whichever came first.
    """
    found: set[str] = set()

    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(key, str) and key.strip().lower() in STRUCTURED_KEYS:
                label = normalize_label(value)
                if label:
                    found.add(label)
            found |= _labels_in_payload(value)
    elif isinstance(payload, list):
        for item in payload:
            found |= _labels_in_payload(item)

    return found


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


def parse_structured_response(response: Any) -> str | None:
    """Extract a label from a JSON response.

    Kept separate from the free-text parser because the structured and combined
    strategies promise a JSON contract, and how well they honour it is itself a
    measurable result.

    Handles a bare object, an object inside markdown fences, an object with
    commentary around it, nested objects, and arrays.

    Args:
        response: Raw model text, or an already-decoded object.

    Returns:
        ``"positive"`` / ``"negative"`` if exactly one label is present under an
        accepted key, ``None`` if no JSON verdict was found at all, and
        :data:`~src.config.LABEL_UNKNOWN` if the JSON contradicts itself.
    """
    if isinstance(response, (dict, list)):
        found = _labels_in_payload(response)
        return _resolve(found, source="decoded object")

    if not isinstance(response, str) or not response.strip():
        return None

    for block in _iter_json_objects(response):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        found = _labels_in_payload(payload)
        if found:
            return _resolve(found, source="json")

    # A whole-response array, e.g. '[{"sentiment": "positive"}]'.
    stripped = response.strip()
    if stripped.startswith("["):
        try:
            found = _labels_in_payload(json.loads(stripped))
        except json.JSONDecodeError:
            return None
        if found:
            return _resolve(found, source="json array")

    return None


def _resolve(found: set[str], *, source: str) -> str | None:
    """Turn a set of candidate labels into a verdict, refusing to break ties."""
    if len(found) == 1:
        return found.pop()
    if len(found) > 1:
        logger.debug("Ambiguous %s response: %s", source, sorted(found))
        return LABEL_UNKNOWN
    return None


# ---------------------------------------------------------------------------
# Free text
# ---------------------------------------------------------------------------


#: A negation cannot reach across these: they end the clause it belongs to.
#: Without this, "The reviewer was not impressed. Sentiment: negative" would
#: have its verdict voided by a negation from the previous sentence.
CLAUSE_BOUNDARY_PATTERN: Final[re.Pattern[str]] = re.compile(r"[.!?;:\n\r]")


def _mention_is_negated(text: str, match_start: int) -> bool:
    """Whether the label at ``match_start`` is negated within its own clause."""
    window = text[max(0, match_start - NEGATION_LOOKBEHIND) : match_start]
    boundaries = list(CLAUSE_BOUNDARY_PATTERN.finditer(window))
    if boundaries:
        window = window[boundaries[-1].end() :]
    return bool(NEGATION_PATTERN.search(window))


def _labels_in_text(text: str) -> set[str]:
    """Collect standalone, non-negated label mentions from free text."""
    return {
        match.group(1).lower()
        for match in LABEL_MENTION_PATTERN.finditer(text)
        if not _mention_is_negated(text, match.start())
    }


def parse_sentiment_response(response: Any) -> ParsedLabel:
    """Parse a raw model response into exactly one label.

    Resolution order, stopping at the first stage that yields an unambiguous
    answer:

    1. **Guard** — anything that is not usable text is ``unknown``.
    2. **Structured** — a JSON verdict, if the response contains one.
    3. **Exact match** — the whole response is a label, once decoration such as
       quotes, markdown emphasis and punctuation is stripped.
    4. **Explicit marker** — a phrase like ``sentiment: positive`` or
       ``The answer is negative``.
    5. **Sole mention** — exactly one label is named in the whole response.

    Anything else is ``unknown``: both labels named, a negated label, a label
    only as part of a longer word, or no label at all.

    Args:
        response: Raw text from the model. Non-string input is tolerated and
            resolves to ``unknown`` rather than raising, so one malformed
            response cannot abort a benchmark run.

    Returns:
        ``"positive"``, ``"negative"`` or ``"unknown"``.
    """
    # 1. Guard.
    if isinstance(response, (dict, list)):
        structured = parse_structured_response(response)
        return structured or LABEL_UNKNOWN  # type: ignore[return-value]
    if not isinstance(response, str):
        return LABEL_UNKNOWN
    text = response.strip()
    if not text:
        return LABEL_UNKNOWN

    # 2. Structured output.
    structured = parse_structured_response(text)
    if structured is not None:
        return structured  # type: ignore[return-value]

    # 3. The entire response is a label.
    exact = normalize_label(text)
    if exact:
        return exact  # type: ignore[return-value]

    # 4/5 operate on text with the prompt's own option list removed, so an
    # echoed "positive or negative" is not mistaken for a decision.
    without_echo = OPTION_ECHO_PATTERN.sub(" ", text)

    # 4. Explicit answer marker.
    marked = {
        match.group(1).lower()
        for match in ANSWER_PATTERN.finditer(without_echo)
        if not _mention_is_negated(without_echo, match.start(1))
    }
    if len(marked) == 1:
        return marked.pop()  # type: ignore[return-value]

    # 5. A single standalone mention anywhere in the response.
    mentioned = _labels_in_text(without_echo)
    if len(mentioned) == 1:
        return mentioned.pop()  # type: ignore[return-value]

    logger.debug("Unparseable response (%d chars): %r", len(text), text[:120])
    return LABEL_UNKNOWN


def is_resolved(label: str) -> bool:
    """Whether a parsed label is a real class rather than ``unknown``."""
    return label in SENTIMENT_LABELS
