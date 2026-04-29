"""Data models for the fake news detection system."""

from .news import (
    NewsItem,
    ParsedNews,
    CredibilityScore,
    PropagationNode,
    PropagationGraph,
    TraceAnalysis,
    DetectionReport,
)

__all__ = [
    "NewsItem",
    "ParsedNews",
    "CredibilityScore",
    "PropagationNode",
    "PropagationGraph",
    "TraceAnalysis",
    "DetectionReport",
]
