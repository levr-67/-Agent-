"""Tests for individual agents."""

from __future__ import annotations

from datetime import datetime

import pytest

from fake_news_detection.agents import (
    CredibilityEvaluationAgent,
    MultimodalParsingAgent,
    ReportGenerationAgent,
    TraceAnalysisAgent,
)
from fake_news_detection.models.news import (
    CredibilityLevel,
    DetectionReport,
    NewsItem,
    PropagationGraph,
    PropagationNode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def credible_item():
    return NewsItem(
        title="Central Bank Raises Interest Rates by 0.5%",
        content=(
            'The central bank announced a 0.5% increase in interest rates on Thursday, '
            'citing persistent inflation figures that have remained above the 3% target '
            'for the past six months. "We are committed to price stability," said the '
            'governor in a press conference. Analysts from 12 major investment banks had '
            'widely predicted the move. The decision was unanimous among the 9 board '
            'members. Markets responded with a 1.2% rise in the currency exchange rate.'
        ),
        source_url="https://reuters.com/finance/rates",
        author="Jane Smith",
        published_at=datetime(2024, 3, 15, 10, 0, 0),
    )


@pytest.fixture
def fake_item():
    return NewsItem(
        title="SHOCKING!!! Government HIDING Alien Technology – MUST SHARE NOW!!!",
        content=(
            "INCREDIBLE unbelievable SHOCKING secret exposed by whistleblower!!! "
            "The government has been HIDING alien technology for DECADES!!! "
            "SHARE this before it gets CENSORED!!! This is EXPLOSIVE information "
            "that they don't want you to know!!! OUTRAGEOUS cover-up!!!"
        ),
        source_url="https://beforeitsnews.com/aliens/123",
    )


@pytest.fixture
def simple_graph():
    nodes = [
        PropagationNode(
            node_id="root",
            platform="twitter",
            user_id="user1",
            timestamp=datetime(2024, 3, 15, 10, 0, 0),
            share_count=500,
            is_verified_account=True,
        ),
        PropagationNode(
            node_id="child1",
            platform="facebook",
            user_id="user2",
            timestamp=datetime(2024, 3, 15, 12, 0, 0),
            parent_id="root",
            share_count=100,
        ),
        PropagationNode(
            node_id="child2",
            platform="twitter",
            user_id="user3",
            timestamp=datetime(2024, 3, 15, 14, 0, 0),
            parent_id="root",
            share_count=80,
        ),
        PropagationNode(
            node_id="grandchild1",
            platform="reddit",
            user_id="user4",
            timestamp=datetime(2024, 3, 15, 16, 0, 0),
            parent_id="child1",
            share_count=30,
        ),
    ]
    edges = [("root", "child1"), ("root", "child2"), ("child1", "grandchild1")]
    return PropagationGraph(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# MultimodalParsingAgent
# ---------------------------------------------------------------------------

class TestMultimodalParsingAgent:
    def test_parses_word_count(self, credible_item):
        agent = MultimodalParsingAgent()
        result = agent.process(credible_item)
        assert result.linguistic_features.word_count > 0

    def test_parses_sentence_count(self, credible_item):
        agent = MultimodalParsingAgent()
        result = agent.process(credible_item)
        assert result.linguistic_features.sentence_count > 0

    def test_extracts_entities(self, credible_item):
        agent = MultimodalParsingAgent()
        result = agent.process(credible_item)
        assert isinstance(result.detected_entities, list)

    def test_extracts_claims(self, credible_item):
        agent = MultimodalParsingAgent()
        result = agent.process(credible_item)
        assert isinstance(result.detected_claims, list)

    def test_image_count_from_urls(self):
        item = NewsItem(
            title="T", content="C",
            image_urls=["http://img.com/1.jpg", "http://img.com/2.jpg"],
        )
        agent = MultimodalParsingAgent()
        result = agent.process(item)
        assert result.image_count == 2

    def test_has_video_from_metadata(self):
        item = NewsItem(
            title="T", content="C",
            metadata={"video_url": "http://v.com/clip.mp4"},
        )
        agent = MultimodalParsingAgent()
        result = agent.process(item)
        assert result.has_video is True

    def test_sensational_word_detection(self, fake_item):
        agent = MultimodalParsingAgent()
        result = agent.process(fake_item)
        assert result.linguistic_features.sensational_word_count > 4

    def test_exclamation_count(self, fake_item):
        agent = MultimodalParsingAgent()
        result = agent.process(fake_item)
        assert result.linguistic_features.exclamation_count >= 3


# ---------------------------------------------------------------------------
# CredibilityEvaluationAgent
# ---------------------------------------------------------------------------

class TestCredibilityEvaluationAgent:
    def test_credible_article_scores_high(self, credible_item):
        parser = MultimodalParsingAgent()
        evaluator = CredibilityEvaluationAgent()
        parsed = parser.process(credible_item)
        score = evaluator.process(parsed)
        assert score.overall_score >= 0.5
        assert score.level in (CredibilityLevel.HIGH, CredibilityLevel.MEDIUM)

    def test_fake_article_scores_low(self, fake_item):
        parser = MultimodalParsingAgent()
        evaluator = CredibilityEvaluationAgent()
        parsed = parser.process(fake_item)
        score = evaluator.process(parsed)
        assert score.overall_score < 0.5

    def test_score_in_valid_range(self, credible_item):
        parser = MultimodalParsingAgent()
        evaluator = CredibilityEvaluationAgent()
        parsed = parser.process(credible_item)
        score = evaluator.process(parsed)
        assert 0.0 <= score.overall_score <= 1.0

    def test_dimensions_present(self, credible_item):
        parser = MultimodalParsingAgent()
        evaluator = CredibilityEvaluationAgent()
        parsed = parser.process(credible_item)
        score = evaluator.process(parsed)
        dim_names = [d.name for d in score.dimensions]
        assert "source_credibility" in dim_names
        assert "writing_style" in dim_names

    def test_flags_for_fake_article(self, fake_item):
        parser = MultimodalParsingAgent()
        evaluator = CredibilityEvaluationAgent()
        parsed = parser.process(fake_item)
        score = evaluator.process(parsed)
        assert len(score.flags) > 0
        # Should flag sensationalism and missing author
        assert any(f in score.flags for f in ("SENSATIONALISM", "EXCESSIVE_EXCLAMATIONS", "NO_AUTHOR"))

    def test_no_author_flag(self):
        item = NewsItem(title="T", content="Some content here " * 10)
        parser = MultimodalParsingAgent()
        evaluator = CredibilityEvaluationAgent()
        parsed = parser.process(item)
        score = evaluator.process(parsed)
        assert "NO_AUTHOR" in score.flags


# ---------------------------------------------------------------------------
# TraceAnalysisAgent
# ---------------------------------------------------------------------------

class TestTraceAnalysisAgent:
    def test_depth_computed(self, simple_graph):
        agent = TraceAnalysisAgent()
        result = agent.process(simple_graph)
        assert result.metrics.depth == 2  # root -> child -> grandchild

    def test_breadth_computed(self, simple_graph):
        agent = TraceAnalysisAgent()
        result = agent.process(simple_graph)
        assert result.metrics.breadth >= 2  # root has 2 direct children

    def test_platform_diversity(self, simple_graph):
        agent = TraceAnalysisAgent()
        result = agent.process(simple_graph)
        # twitter, facebook, reddit = 3
        assert result.metrics.platform_diversity == 3

    def test_origin_platform(self, simple_graph):
        agent = TraceAnalysisAgent()
        result = agent.process(simple_graph)
        assert result.origin_platform == "twitter"

    def test_total_reach(self, simple_graph):
        agent = TraceAnalysisAgent()
        result = agent.process(simple_graph)
        assert result.metrics.total_reach == 710  # 500+100+80+30

    def test_timeline_sorted(self, simple_graph):
        agent = TraceAnalysisAgent()
        result = agent.process(simple_graph)
        ts_list = [e["timestamp"] for e in result.timeline]
        assert ts_list == sorted(ts_list)

    def test_empty_graph(self):
        agent = TraceAnalysisAgent()
        empty = PropagationGraph()
        result = agent.process(empty)
        assert result.metrics.depth == 0
        assert result.metrics.breadth == 0

    def test_render_graph(self, simple_graph):
        agent = TraceAnalysisAgent()
        ascii_graph = agent.render_graph(simple_graph)
        assert "root" in ascii_graph


# ---------------------------------------------------------------------------
# ReportGenerationAgent
# ---------------------------------------------------------------------------

class TestReportGenerationAgent:
    def _make_report(self, credible_item):
        parser = MultimodalParsingAgent()
        evaluator = CredibilityEvaluationAgent()
        parsed = parser.process(credible_item)
        score = evaluator.process(parsed)
        return DetectionReport(
            item_id=credible_item.item_id,
            news_item=credible_item,
            parsed_news=parsed,
            credibility_score=score,
            summary="Test report",
        )

    def test_process_returns_html_string(self, credible_item):
        agent = ReportGenerationAgent()
        report = self._make_report(credible_item)
        html = agent.process(report)
        assert isinstance(html, str)
        assert "<html" in html.lower()

    def test_html_contains_score(self, credible_item):
        agent = ReportGenerationAgent()
        report = self._make_report(credible_item)
        html = agent.process(report)
        assert "%" in html

    def test_html_contains_title(self, credible_item):
        agent = ReportGenerationAgent()
        report = self._make_report(credible_item)
        html = agent.process(report)
        assert "Central Bank" in html

    def test_save_creates_file(self, tmp_path, credible_item):
        agent = ReportGenerationAgent(output_dir=str(tmp_path))
        report = self._make_report(credible_item)
        path = agent.save(report)
        assert os.path.exists(path)
        assert path.endswith(".html")

    def test_batch_creates_file(self, tmp_path, credible_item, fake_item):
        from fake_news_detection.models.news import NewsItem
        agent = ReportGenerationAgent(output_dir=str(tmp_path))
        reports = [self._make_report(credible_item), self._make_report(fake_item)]
        path = agent.process_batch(reports)
        assert os.path.exists(path)


import os
