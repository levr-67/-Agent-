"""Tests for data models."""

from datetime import datetime

import pytest

from fake_news_detection.models.news import (
    CredibilityLevel,
    CredibilityScore,
    DetectionReport,
    NewsItem,
    PropagationGraph,
    PropagationNode,
    _extract_domain,
)


class TestNewsItem:
    def test_item_id_auto_generated(self):
        item = NewsItem(title="Test", content="Test content")
        assert item.item_id
        assert len(item.item_id) > 0

    def test_two_items_have_different_ids(self):
        a = NewsItem(title="A", content="A")
        b = NewsItem(title="B", content="B")
        assert a.item_id != b.item_id

    def test_domain_extracted_from_url(self):
        item = NewsItem(
            title="T", content="C",
            source_url="https://www.reuters.com/article/123"
        )
        assert item.source_domain == "reuters.com"

    def test_domain_not_overwritten_if_provided(self):
        item = NewsItem(
            title="T", content="C",
            source_url="https://reuters.com/article",
            source_domain="custom.com",
        )
        assert item.source_domain == "custom.com"


class TestExtractDomain:
    @pytest.mark.parametrize("url,expected", [
        ("https://www.reuters.com/article", "reuters.com"),
        ("http://bbc.co.uk/news/1", "bbc.co.uk"),
        ("https://nytimes.com/", "nytimes.com"),
        ("https://site.com:8080/path", "site.com"),
        ("not-a-url", "not-a-url"),
        ("", ""),
    ])
    def test_extraction(self, url, expected):
        assert _extract_domain(url) == expected


class TestCredibilityLevel:
    @pytest.mark.parametrize("score,expected", [
        (1.0, CredibilityLevel.HIGH),
        (0.75, CredibilityLevel.HIGH),
        (0.74, CredibilityLevel.MEDIUM),
        (0.50, CredibilityLevel.MEDIUM),
        (0.49, CredibilityLevel.LOW),
        (0.25, CredibilityLevel.LOW),
        (0.24, CredibilityLevel.FAKE),
        (0.0, CredibilityLevel.FAKE),
    ])
    def test_from_score(self, score, expected):
        assert CredibilityLevel.from_score(score) == expected


class TestCredibilityScore:
    def test_level_derived_from_score(self):
        cs = CredibilityScore(overall_score=0.8)
        assert cs.level == CredibilityLevel.HIGH

    def test_fake_level(self):
        cs = CredibilityScore(overall_score=0.1)
        assert cs.level == CredibilityLevel.FAKE


class TestPropagationGraph:
    def _make_graph(self):
        nodes = [
            PropagationNode(node_id="a", platform="twitter"),
            PropagationNode(node_id="b", platform="facebook"),
            PropagationNode(node_id="c", platform="twitter"),
        ]
        edges = [("a", "b"), ("a", "c")]
        return PropagationGraph(nodes=nodes, edges=edges)

    def test_node_count(self):
        g = self._make_graph()
        assert g.node_count == 3

    def test_edge_count(self):
        g = self._make_graph()
        assert g.edge_count == 2

    def test_roots(self):
        g = self._make_graph()
        roots = g.get_roots()
        assert len(roots) == 1
        assert roots[0].node_id == "a"
