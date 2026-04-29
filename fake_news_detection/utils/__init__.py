"""Utility package for fake news detection helpers."""

from .text_utils import (
    caps_ratio,
    count_numeric_claims,
    count_quoted_sources,
    count_sentences,
    count_sensational_words,
    count_words,
    extract_claims,
    extract_entities,
    readability_score,
    source_credibility_score,
)
from .graph_utils import (
    build_adjacency,
    build_timeline,
    compute_breadth,
    compute_cascade_length,
    compute_depth,
    compute_platform_diversity,
    compute_verified_ratio,
    render_ascii_graph,
)

__all__ = [
    "caps_ratio",
    "count_numeric_claims",
    "count_quoted_sources",
    "count_sentences",
    "count_sensational_words",
    "count_words",
    "extract_claims",
    "extract_entities",
    "readability_score",
    "source_credibility_score",
    "build_adjacency",
    "build_timeline",
    "compute_breadth",
    "compute_cascade_length",
    "compute_depth",
    "compute_platform_diversity",
    "compute_verified_ratio",
    "render_ascii_graph",
]
