"""Top-level package for the fake news detection system."""

from fake_news_detection.orchestrator import Orchestrator
from fake_news_detection.models.news import (
    NewsItem,
    ParsedNews,
    CredibilityScore,
    PropagationGraph,
    PropagationNode,
    TraceAnalysis,
    DetectionReport,
)

__all__ = [
    "Orchestrator",
    "NewsItem",
    "ParsedNews",
    "CredibilityScore",
    "PropagationGraph",
    "PropagationNode",
    "TraceAnalysis",
    "DetectionReport",
]
