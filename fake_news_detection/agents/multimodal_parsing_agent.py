"""Multimodal Parsing Agent.

Responsible for extracting structured features from a raw ``NewsItem``,
including text linguistics, detected entities, factual claims, and basic
image metadata.
"""

from __future__ import annotations

from fake_news_detection.agents.base_agent import BaseAgent
from fake_news_detection.models.news import LinguisticFeatures, NewsItem, ParsedNews
from fake_news_detection.utils.text_utils import (
    caps_ratio,
    count_numeric_claims,
    count_quoted_sources,
    count_sentences,
    count_sensational_words,
    count_words,
    extract_claims,
    extract_entities,
    readability_score,
)


class MultimodalParsingAgent(BaseAgent):
    """Agent that parses a raw ``NewsItem`` into a rich ``ParsedNews`` object.

    The agent performs:

    * **Linguistic analysis** – word / sentence counts, caps ratio,
      sensationalism indicators, readability.
    * **Entity extraction** – naive capitalisation-based named-entity
      recognition.
    * **Claim extraction** – identifies declarative sentences likely to
      contain factual claims.
    * **Media accounting** – counts attached images / videos from the item
      metadata.
    """

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(name="MultimodalParsingAgent", verbose=verbose)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, input_data: NewsItem) -> ParsedNews:  # type: ignore[override]
        """Parse *input_data* and return a ``ParsedNews`` result.

        Args:
            input_data: The raw news item to analyse.

        Returns:
            A ``ParsedNews`` object containing all extracted features.
        """
        self._info(f"Parsing news item '{input_data.item_id}'")
        full_text = f"{input_data.title}\n\n{input_data.content}"

        linguistic = self._analyse_linguistics(full_text)
        entities = extract_entities(full_text)
        claims = extract_claims(input_data.content)
        image_count, has_video = self._count_media(input_data)

        result = ParsedNews(
            item=input_data,
            linguistic_features=linguistic,
            detected_entities=entities,
            detected_claims=claims,
            image_count=image_count,
            has_video=has_video,
        )
        self._debug(
            f"Parsed: {linguistic.word_count} words, "
            f"{len(entities)} entities, {len(claims)} claims"
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _analyse_linguistics(text: str) -> LinguisticFeatures:
        words = count_words(text)
        sentences = count_sentences(text)
        avg_sent_len = (words / sentences) if sentences > 0 else 0.0
        return LinguisticFeatures(
            word_count=words,
            sentence_count=sentences,
            avg_sentence_length=round(avg_sent_len, 2),
            exclamation_count=text.count("!"),
            question_count=text.count("?"),
            caps_ratio=round(caps_ratio(text), 4),
            sensational_word_count=count_sensational_words(text),
            quoted_source_count=count_quoted_sources(text),
            numeric_claim_count=count_numeric_claims(text),
            readability_score=round(readability_score(text), 4),
        )

    @staticmethod
    def _count_media(item: NewsItem) -> tuple:
        image_count = len(item.image_urls)
        has_video = bool(item.metadata.get("video_url") or item.metadata.get("has_video"))
        return image_count, has_video
