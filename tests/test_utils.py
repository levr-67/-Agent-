"""Tests for utility functions."""

from __future__ import annotations

import pytest

from fake_news_detection.utils.text_utils import (
    caps_ratio,
    count_numeric_claims,
    count_quoted_sources,
    count_sentences,
    count_sensational_words,
    count_words,
    extract_entities,
    extract_claims,
    readability_score,
    source_credibility_score,
)
from fake_news_detection.utils.graph_utils import (
    build_adjacency,
    compute_breadth,
    compute_depth,
    compute_platform_diversity,
    compute_verified_ratio,
    render_ascii_graph,
)
from fake_news_detection.models.news import PropagationGraph, PropagationNode


class TestTextUtils:
    def test_count_words(self):
        assert count_words("Hello world") == 2
        assert count_words("") == 0
        assert count_words("one") == 1

    def test_count_sentences(self):
        assert count_sentences("Hello. World.") >= 2
        assert count_sentences("No sentence ending") == 1

    def test_caps_ratio_full_caps(self):
        assert caps_ratio("HELLO") == 1.0

    def test_caps_ratio_lower(self):
        assert caps_ratio("hello") == 0.0

    def test_caps_ratio_mixed(self):
        ratio = caps_ratio("HeLLo")
        assert 0.0 < ratio < 1.0

    def test_count_sensational_words(self):
        assert count_sensational_words("shocking unbelievable scandal hoax viral") >= 4

    def test_count_sensational_words_none(self):
        assert count_sensational_words("The bank raised rates today.") == 0

    def test_count_quoted_sources(self):
        text = 'He said "this is a very important statement" yesterday.'
        assert count_quoted_sources(text) >= 1

    def test_count_numeric_claims(self):
        assert count_numeric_claims("The rate rose 0.5% to 4.25%.") >= 2
        assert count_numeric_claims("No numbers here.") == 0

    def test_readability_score_range(self):
        score = readability_score("The cat sat on the mat. It was a nice cat.")
        assert 0.0 <= score <= 1.0

    def test_extract_entities_finds_proper_nouns(self):
        entities = extract_entities("John Smith visited New York last week.")
        assert any("John" in e or "New York" in e for e in entities)

    def test_source_credibility_high(self):
        assert source_credibility_score("reuters.com") == 1.0

    def test_source_credibility_low(self):
        assert source_credibility_score("infowars.com") == 0.0

    def test_source_credibility_unknown(self):
        score = source_credibility_score("some-unknown-blog.net")
        assert score == 0.5


class TestGraphUtils:
    def _make_graph(self):
        nodes = [
            PropagationNode(node_id="a", platform="twitter", is_verified_account=True),
            PropagationNode(node_id="b", platform="facebook"),
            PropagationNode(node_id="c", platform="reddit"),
        ]
        edges = [("a", "b"), ("b", "c")]
        return PropagationGraph(nodes=nodes, edges=edges)

    def test_build_adjacency(self):
        g = self._make_graph()
        adj = build_adjacency(g)
        assert adj["a"] == ["b"]
        assert adj["b"] == ["c"]
        assert "c" not in adj

    def test_compute_depth(self):
        g = self._make_graph()
        assert compute_depth(g) == 2

    def test_compute_breadth(self):
        # Linear chain: max width = 1
        g = self._make_graph()
        assert compute_breadth(g) == 1

    def test_compute_depth_empty(self):
        assert compute_depth(PropagationGraph()) == 0

    def test_platform_diversity(self):
        g = self._make_graph()
        assert compute_platform_diversity(g) == 3

    def test_verified_ratio(self):
        g = self._make_graph()
        ratio = compute_verified_ratio(g)
        # 1 out of 3 nodes verified
        assert abs(ratio - 1 / 3) < 0.01

    def test_render_ascii_graph(self):
        g = self._make_graph()
        output = render_ascii_graph(g)
        assert "a" in output
        assert "b" in output
        assert "c" in output

    def test_render_empty_graph(self):
        output = render_ascii_graph(PropagationGraph())
        assert "empty" in output
