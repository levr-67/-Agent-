"""Tests for the Orchestrator."""

from __future__ import annotations

import os
from datetime import datetime

import pytest

from fake_news_detection.models.news import (
    DetectionReport,
    NewsItem,
    PropagationGraph,
    PropagationNode,
)
from fake_news_detection.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_item():
    return NewsItem(
        title="New Study Links Coffee Consumption to Improved Memory",
        content=(
            "Researchers at the University of Science published a study on Tuesday "
            "showing that drinking 2 cups of coffee per day is associated with a "
            "15% improvement in short-term memory performance among adults aged 40-60. "
            'Lead researcher Dr. Maria Chen said: "The results are significant and '
            "replicated across three independent cohorts totalling 4,200 participants.\" "
            "The study was published in the Journal of Neurological Science."
        ),
        source_url="https://reuters.com/health/coffee-memory",
        author="Tom Williams",
        published_at=datetime(2024, 6, 1, 9, 0, 0),
    )


@pytest.fixture
def sample_graph():
    nodes = [
        PropagationNode(
            node_id="n1", platform="twitter", user_id="u1",
            timestamp=datetime(2024, 6, 1, 9, 0, 0), share_count=200,
            is_verified_account=True,
        ),
        PropagationNode(
            node_id="n2", platform="facebook", user_id="u2",
            timestamp=datetime(2024, 6, 1, 10, 30, 0), parent_id="n1",
            share_count=80,
        ),
        PropagationNode(
            node_id="n3", platform="reddit", user_id="u3",
            timestamp=datetime(2024, 6, 1, 12, 0, 0), parent_id="n1",
            share_count=50,
        ),
    ]
    return PropagationGraph(
        nodes=nodes,
        edges=[("n1", "n2"), ("n1", "n3")],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def test_run_returns_detection_report(self, tmp_path, sample_item):
        orch = Orchestrator(output_dir=str(tmp_path), save_reports=False)
        report = orch.run(sample_item)
        assert isinstance(report, DetectionReport)

    def test_run_populates_credibility(self, tmp_path, sample_item):
        orch = Orchestrator(output_dir=str(tmp_path), save_reports=False)
        report = orch.run(sample_item)
        assert report.credibility_score is not None
        assert 0.0 <= report.credibility_score.overall_score <= 1.0

    def test_run_populates_parsed_news(self, tmp_path, sample_item):
        orch = Orchestrator(output_dir=str(tmp_path), save_reports=False)
        report = orch.run(sample_item)
        assert report.parsed_news is not None

    def test_run_without_graph_no_trace(self, tmp_path, sample_item):
        orch = Orchestrator(output_dir=str(tmp_path), save_reports=False)
        report = orch.run(sample_item)
        assert report.trace_analysis is None

    def test_run_with_graph_adds_trace(self, tmp_path, sample_item, sample_graph):
        orch = Orchestrator(output_dir=str(tmp_path), save_reports=False)
        report = orch.run(sample_item, propagation_graph=sample_graph)
        assert report.trace_analysis is not None
        assert report.trace_analysis.metrics.depth > 0

    def test_run_saves_html_report(self, tmp_path, sample_item):
        orch = Orchestrator(output_dir=str(tmp_path), save_reports=True)
        report = orch.run(sample_item)
        files = list(tmp_path.glob("*.html"))
        assert len(files) == 1

    def test_batch_run_returns_all_reports(self, tmp_path):
        items = [
            NewsItem(
                title=f"Article {i}",
                content="Content " * 30,
                source_url=f"https://example.com/{i}",
            )
            for i in range(4)
        ]
        orch = Orchestrator(output_dir=str(tmp_path), save_reports=False)
        reports = orch.batch_run(items)
        assert len(reports) == 4

    def test_batch_run_saves_batch_report(self, tmp_path):
        items = [
            NewsItem(title=f"Article {i}", content="Content " * 20)
            for i in range(3)
        ]
        orch = Orchestrator(output_dir=str(tmp_path), save_reports=True)
        orch.batch_run(items)
        # Should have 3 individual + 1 batch report = 4 HTML files
        files = list(tmp_path.glob("*.html"))
        assert len(files) == 4

    def test_batch_with_graphs(self, tmp_path, sample_item, sample_graph):
        orch = Orchestrator(output_dir=str(tmp_path), save_reports=False)
        reports = orch.batch_run(
            [sample_item],
            propagation_graphs={sample_item.item_id: sample_graph},
        )
        assert reports[0].trace_analysis is not None

    def test_summarise_high(self, tmp_path, sample_item):
        orch = Orchestrator(output_dir=str(tmp_path), save_reports=False)
        report = orch.run(sample_item)
        # reuters.com is a high-credibility domain
        assert "credible" in report.summary.lower() or "mixed" in report.summary.lower()
