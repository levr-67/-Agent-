"""Trace Analysis Agent.

Analyses the propagation graph of a news item and computes spread metrics.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from fake_news_detection.agents.base_agent import BaseAgent
from fake_news_detection.models.news import (
    PropagationGraph,
    PropagationNode,
    TraceAnalysis,
    TraceMetrics,
)
from fake_news_detection.utils.graph_utils import (
    build_timeline,
    compute_breadth,
    compute_cascade_length,
    compute_depth,
    compute_platform_diversity,
    compute_verified_ratio,
    render_ascii_graph,
)


class TraceAnalysisAgent(BaseAgent):
    """Agent that analyses how a piece of news has propagated.

    Accepts a ``PropagationGraph`` and returns a ``TraceAnalysis`` containing:

    * Topological metrics (depth, breadth, cascade length).
    * Temporal metrics (spread velocity in first 24 h).
    * Platform diversity.
    * Verified vs. unverified share ratio.
    * A chronological timeline of sharing events.
    * An ASCII visualisation of the propagation tree.
    """

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(name="TraceAnalysisAgent", verbose=verbose)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, input_data: PropagationGraph) -> TraceAnalysis:  # type: ignore[override]
        """Analyse the propagation graph and return a ``TraceAnalysis``.

        Args:
            input_data: The propagation graph to analyse.

        Returns:
            A ``TraceAnalysis`` with metrics and timeline.
        """
        self._info(
            f"Analysing graph: {input_data.node_count} nodes, "
            f"{input_data.edge_count} edges"
        )

        metrics = self._compute_metrics(input_data)
        timeline = build_timeline(input_data)
        origin_platform = self._detect_origin_platform(input_data)

        result = TraceAnalysis(
            graph=input_data,
            metrics=metrics,
            origin_platform=origin_platform,
            timeline=timeline,
        )
        self._debug(
            f"Depth: {metrics.depth}, Breadth: {metrics.breadth}, "
            f"Velocity: {metrics.spread_velocity:.1f} shares/h"
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_metrics(self, graph: PropagationGraph) -> TraceMetrics:
        depth = compute_depth(graph)
        breadth = compute_breadth(graph)
        platform_div = compute_platform_diversity(graph)
        verified_ratio = compute_verified_ratio(graph)
        cascade_len = compute_cascade_length(graph)
        total_reach = sum(n.share_count for n in graph.nodes)
        velocity = self._compute_velocity(graph)

        return TraceMetrics(
            depth=depth,
            breadth=breadth,
            spread_velocity=round(velocity, 2),
            total_reach=total_reach,
            platform_diversity=platform_div,
            verified_share_ratio=round(verified_ratio, 4),
            cascade_length=round(cascade_len, 2),
        )

    @staticmethod
    def _compute_velocity(graph: PropagationGraph) -> float:
        """Compute shares-per-hour during the first 24 hours."""
        timed_nodes = [n for n in graph.nodes if n.timestamp is not None]
        if len(timed_nodes) < 2:
            return 0.0
        timed_nodes.sort(key=lambda n: n.timestamp)  # type: ignore[arg-type]
        t0: datetime = timed_nodes[0].timestamp  # type: ignore[assignment]
        cutoff = t0 + timedelta(hours=24)
        first_day = [n for n in timed_nodes if n.timestamp <= cutoff]  # type: ignore[operator]
        if len(first_day) < 2:
            return 0.0
        span_hours = (
            (first_day[-1].timestamp - first_day[0].timestamp).total_seconds() / 3600  # type: ignore[operator]
        )
        if span_hours < 0.01:
            return 0.0
        return (len(first_day) - 1) / span_hours

    @staticmethod
    def _detect_origin_platform(graph: PropagationGraph) -> str:
        roots = graph.get_roots()
        if not roots:
            return "unknown"
        platforms = [r.platform for r in roots if r.platform]
        return platforms[0] if platforms else "unknown"

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def render_graph(self, graph: PropagationGraph) -> str:
        """Return an ASCII art tree of the propagation graph."""
        return render_ascii_graph(graph)
