"""Credibility Evaluation Agent.

Evaluates the credibility of a ``ParsedNews`` item across multiple
dimensions and returns a composite ``CredibilityScore``.
"""

from __future__ import annotations

from typing import List

from fake_news_detection.agents.base_agent import BaseAgent
from fake_news_detection.models.news import (
    CredibilityDimension,
    CredibilityScore,
    ParsedNews,
)
from fake_news_detection.utils.text_utils import source_credibility_score


# ---------------------------------------------------------------------------
# Heuristic thresholds
# ---------------------------------------------------------------------------
_HIGH_CAPS_RATIO = 0.30       # > 30 % uppercase letters → flag
_HIGH_SENSATIONAL = 4         # > 4 sensational words → flag
_MIN_WORD_COUNT = 100         # very short articles are less credible
_MIN_QUOTED_SOURCES = 1       # at least one quoted source boosts credibility
_HIGH_EXCLAMATIONS = 3        # > 3 exclamation marks → flag


class CredibilityEvaluationAgent(BaseAgent):
    """Agent that scores a parsed news item for credibility.

    Six weighted dimensions contribute to the overall score:

    1. **Source credibility** – based on domain reputation.
    2. **Writing style** – caps ratio, exclamation marks, sensationalism.
    3. **Content quality** – length, readability, sourcing.
    4. **Author presence** – whether an identifiable author is credited.
    5. **Claim substantiation** – numeric evidence and quoted sources.
    6. **Temporal freshness** – whether publication date is provided.
    """

    _DIMENSION_WEIGHTS = {
        "source_credibility": 0.30,
        "writing_style": 0.20,
        "content_quality": 0.20,
        "author_presence": 0.10,
        "claim_substantiation": 0.15,
        "temporal_freshness": 0.05,
    }

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(name="CredibilityEvaluationAgent", verbose=verbose)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, input_data: ParsedNews) -> CredibilityScore:  # type: ignore[override]
        """Evaluate the credibility of *input_data*.

        Args:
            input_data: Parsed news item to evaluate.

        Returns:
            A ``CredibilityScore`` with per-dimension breakdown and flags.
        """
        self._info(f"Evaluating '{input_data.item.item_id}'")

        dimensions = self._score_all_dimensions(input_data)
        overall = self._weighted_average(dimensions)
        flags = self._collect_flags(input_data)

        result = CredibilityScore(
            overall_score=round(overall, 4),
            dimensions=dimensions,
            flags=flags,
        )
        self._debug(
            f"Score: {result.overall_score:.2f} ({result.level.value}), "
            f"{len(flags)} flags"
        )
        return result

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_all_dimensions(self, parsed: ParsedNews) -> List[CredibilityDimension]:
        scorers = [
            self._score_source,
            self._score_writing_style,
            self._score_content_quality,
            self._score_author_presence,
            self._score_claim_substantiation,
            self._score_temporal_freshness,
        ]
        return [s(parsed) for s in scorers]

    def _score_source(self, parsed: ParsedNews) -> CredibilityDimension:
        domain = parsed.item.source_domain
        score = source_credibility_score(domain) if domain else 0.4
        explanation = (
            f"Domain '{domain}' has known "
            + ("high" if score >= 0.75 else "low" if score < 0.3 else "unknown")
            + " credibility."
        )
        return CredibilityDimension(
            name="source_credibility",
            score=score,
            weight=self._DIMENSION_WEIGHTS["source_credibility"],
            explanation=explanation,
        )

    def _score_writing_style(self, parsed: ParsedNews) -> CredibilityDimension:
        lf = parsed.linguistic_features
        penalties = 0.0
        notes = []

        if lf.caps_ratio > _HIGH_CAPS_RATIO:
            penalties += 0.35
            notes.append(f"high caps ratio ({lf.caps_ratio:.0%})")
        if lf.sensational_word_count > _HIGH_SENSATIONAL:
            penalties += 0.25
            notes.append(f"{lf.sensational_word_count} sensational words")
        if lf.exclamation_count > _HIGH_EXCLAMATIONS:
            penalties += 0.20
            notes.append(f"{lf.exclamation_count} exclamation marks")

        score = max(0.0, 1.0 - penalties)
        explanation = "; ".join(notes) if notes else "Writing style appears neutral."
        return CredibilityDimension(
            name="writing_style",
            score=round(score, 4),
            weight=self._DIMENSION_WEIGHTS["writing_style"],
            explanation=explanation,
        )

    def _score_content_quality(self, parsed: ParsedNews) -> CredibilityDimension:
        lf = parsed.linguistic_features
        score = 0.5  # baseline
        notes = []

        if lf.word_count >= _MIN_WORD_COUNT:
            score += 0.20
        else:
            notes.append(f"short article ({lf.word_count} words)")

        if lf.quoted_source_count >= _MIN_QUOTED_SOURCES:
            score += 0.15
        else:
            notes.append("no quoted sources found")

        # Readability: ideal range 0.30–0.70
        if 0.30 <= lf.readability_score <= 0.70:
            score += 0.10
        else:
            notes.append(f"unusual readability ({lf.readability_score:.2f})")

        score = min(1.0, score)
        explanation = "; ".join(notes) if notes else "Content quality is adequate."
        return CredibilityDimension(
            name="content_quality",
            score=round(score, 4),
            weight=self._DIMENSION_WEIGHTS["content_quality"],
            explanation=explanation,
        )

    @staticmethod
    def _score_author_presence(parsed: ParsedNews) -> CredibilityDimension:
        has_author = bool(parsed.item.author and parsed.item.author.strip())
        score = 1.0 if has_author else 0.2
        return CredibilityDimension(
            name="author_presence",
            score=score,
            weight=CredibilityEvaluationAgent._DIMENSION_WEIGHTS["author_presence"],
            explanation=(
                f"Author identified: {parsed.item.author!r}"
                if has_author
                else "No author information provided."
            ),
        )

    @staticmethod
    def _score_claim_substantiation(parsed: ParsedNews) -> CredibilityDimension:
        lf = parsed.linguistic_features
        score = 0.0
        notes = []

        if lf.numeric_claim_count > 0:
            score += min(0.50, lf.numeric_claim_count * 0.10)
        else:
            notes.append("no numeric evidence")

        if lf.quoted_source_count > 0:
            score += min(0.50, lf.quoted_source_count * 0.15)
        else:
            notes.append("no quoted sources")

        explanation = (
            "; ".join(notes)
            if notes
            else f"{lf.numeric_claim_count} numeric claims, {lf.quoted_source_count} quoted sources."
        )
        return CredibilityDimension(
            name="claim_substantiation",
            score=round(min(1.0, score), 4),
            weight=CredibilityEvaluationAgent._DIMENSION_WEIGHTS["claim_substantiation"],
            explanation=explanation,
        )

    @staticmethod
    def _score_temporal_freshness(parsed: ParsedNews) -> CredibilityDimension:
        has_date = parsed.item.published_at is not None
        score = 1.0 if has_date else 0.3
        return CredibilityDimension(
            name="temporal_freshness",
            score=score,
            weight=CredibilityEvaluationAgent._DIMENSION_WEIGHTS["temporal_freshness"],
            explanation=(
                f"Published at: {parsed.item.published_at}"
                if has_date
                else "No publication date provided."
            ),
        )

    # ------------------------------------------------------------------
    # Aggregation and flagging
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_average(dimensions: List[CredibilityDimension]) -> float:
        total_weight = sum(d.weight for d in dimensions)
        if total_weight == 0:
            return 0.5
        return sum(d.score * d.weight for d in dimensions) / total_weight

    @staticmethod
    def _collect_flags(parsed: ParsedNews) -> List[str]:
        flags: List[str] = []
        lf = parsed.linguistic_features

        if lf.caps_ratio > _HIGH_CAPS_RATIO:
            flags.append("EXCESSIVE_CAPS")
        if lf.sensational_word_count > _HIGH_SENSATIONAL:
            flags.append("SENSATIONALISM")
        if lf.exclamation_count > _HIGH_EXCLAMATIONS:
            flags.append("EXCESSIVE_EXCLAMATIONS")
        if lf.word_count < _MIN_WORD_COUNT:
            flags.append("VERY_SHORT_CONTENT")
        if not parsed.item.author:
            flags.append("NO_AUTHOR")
        if not parsed.item.published_at:
            flags.append("NO_PUBLICATION_DATE")
        if not parsed.item.source_domain:
            flags.append("NO_SOURCE_DOMAIN")
        return flags
