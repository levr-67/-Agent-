"""Core data models for fake news detection."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class CredibilityLevel(Enum):
    """Qualitative credibility classification."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    FAKE = "fake"

    @classmethod
    def from_score(cls, score: float) -> "CredibilityLevel":
        """Map a 0–1 score to a credibility level."""
        if score >= 0.75:
            return cls.HIGH
        if score >= 0.50:
            return cls.MEDIUM
        if score >= 0.25:
            return cls.LOW
        return cls.FAKE


@dataclass
class NewsItem:
    """Raw news item submitted for analysis."""

    title: str
    content: str
    source_url: str = ""
    source_domain: str = ""
    author: str = ""
    published_at: Optional[datetime] = None
    image_urls: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not self.source_domain and self.source_url:
            self.source_domain = _extract_domain(self.source_url)


@dataclass
class LinguisticFeatures:
    """Linguistic features extracted from news text."""

    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    exclamation_count: int = 0
    question_count: int = 0
    caps_ratio: float = 0.0
    sensational_word_count: int = 0
    quoted_source_count: int = 0
    numeric_claim_count: int = 0
    readability_score: float = 0.0


@dataclass
class ParsedNews:
    """Result of the multimodal parsing agent."""

    item: NewsItem
    linguistic_features: LinguisticFeatures = field(default_factory=LinguisticFeatures)
    detected_entities: List[str] = field(default_factory=list)
    detected_claims: List[str] = field(default_factory=list)
    image_count: int = 0
    has_video: bool = False
    parsed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CredibilityDimension:
    """Score for a single credibility dimension."""

    name: str
    score: float  # 0.0–1.0
    weight: float  # relative weight
    explanation: str = ""


@dataclass
class CredibilityScore:
    """Overall credibility evaluation result."""

    overall_score: float  # 0.0–1.0
    level: CredibilityLevel = field(init=False)
    dimensions: List[CredibilityDimension] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        self.level = CredibilityLevel.from_score(self.overall_score)


@dataclass
class PropagationNode:
    """A single node in the news propagation graph."""

    node_id: str
    platform: str = ""
    user_id: str = ""
    timestamp: Optional[datetime] = None
    parent_id: Optional[str] = None
    share_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    is_verified_account: bool = False


@dataclass
class PropagationGraph:
    """Graph representing how a news item spread."""

    nodes: List[PropagationNode] = field(default_factory=list)
    # edges as (source_id, target_id) pairs
    edges: List[tuple] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def get_roots(self) -> List[PropagationNode]:
        """Return nodes with no parent (origin nodes)."""
        child_ids = {e[1] for e in self.edges}
        return [n for n in self.nodes if n.node_id not in child_ids]


@dataclass
class TraceMetrics:
    """Quantitative metrics about the propagation trace."""

    depth: int = 0
    breadth: int = 0
    spread_velocity: float = 0.0  # shares per hour in first 24 h
    total_reach: int = 0
    platform_diversity: int = 0
    verified_share_ratio: float = 0.0
    cascade_length: float = 0.0


@dataclass
class TraceAnalysis:
    """Result of the trace analysis agent."""

    graph: PropagationGraph = field(default_factory=PropagationGraph)
    metrics: TraceMetrics = field(default_factory=TraceMetrics)
    origin_platform: str = ""
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DetectionReport:
    """Final detection report combining all agent outputs."""

    item_id: str
    news_item: NewsItem
    parsed_news: Optional[ParsedNews] = None
    credibility_score: Optional[CredibilityScore] = None
    trace_analysis: Optional[TraceAnalysis] = None
    summary: str = ""
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """Extract the domain from a URL string without external dependencies."""
    try:
        # strip scheme
        if "://" in url:
            url = url.split("://", 1)[1]
        # strip path / query
        url = url.split("/")[0].split("?")[0].split("#")[0]
        # strip port
        url = url.split(":")[0]
        # strip leading www.
        if url.startswith("www."):
            url = url[4:]
        return url.lower()
    except Exception:
        return ""
