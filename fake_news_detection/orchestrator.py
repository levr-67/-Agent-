"""Orchestrator – coordinates all agents for end-to-end detection.

Usage example::

    from fake_news_detection.orchestrator import Orchestrator
    from fake_news_detection.models import NewsItem

    orch = Orchestrator(output_dir="reports")
    report = orch.run(NewsItem(title="...", content="..."))
    orch.batch_run([item1, item2, ...])
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from fake_news_detection.agents import (
    CredibilityEvaluationAgent,
    MultimodalParsingAgent,
    ReportGenerationAgent,
    TraceAnalysisAgent,
)
from fake_news_detection.models.news import (
    DetectionReport,
    NewsItem,
    PropagationGraph,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the four agents in a sequential pipeline.

    Pipeline stages
    ---------------
    1. **MultimodalParsingAgent**  – parse raw ``NewsItem`` → ``ParsedNews``
    2. **CredibilityEvaluationAgent** – score credibility → ``CredibilityScore``
    3. **TraceAnalysisAgent** – analyse propagation → ``TraceAnalysis``
    4. **ReportGenerationAgent** – render HTML report → file on disk

    Args:
        output_dir: Directory where HTML reports are saved.
        verbose: Enable debug logging in each agent.
        save_reports: Whether to persist HTML reports to disk.
    """

    def __init__(
        self,
        output_dir: str = "reports",
        verbose: bool = False,
        save_reports: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.save_reports = save_reports

        self._parser = MultimodalParsingAgent(verbose=verbose)
        self._evaluator = CredibilityEvaluationAgent(verbose=verbose)
        self._tracer = TraceAnalysisAgent(verbose=verbose)
        self._reporter = ReportGenerationAgent(output_dir=output_dir, verbose=verbose)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        news_item: NewsItem,
        propagation_graph: Optional[PropagationGraph] = None,
    ) -> DetectionReport:
        """Run the full detection pipeline on a single *news_item*.

        Args:
            news_item: The news article to analyse.
            propagation_graph: Optional pre-built propagation graph.  If
                ``None``, trace analysis is skipped.

        Returns:
            A completed ``DetectionReport``.
        """
        logger.info("Pipeline start: '%s'", news_item.item_id)

        # Stage 1 – parse
        parsed = self._parser.process(news_item)

        # Stage 2 – credibility
        credibility = self._evaluator.process(parsed)

        # Stage 3 – trace (optional)
        trace = None
        if propagation_graph is not None:
            trace = self._tracer.process(propagation_graph)

        # Assemble report
        report = DetectionReport(
            item_id=news_item.item_id,
            news_item=news_item,
            parsed_news=parsed,
            credibility_score=credibility,
            trace_analysis=trace,
            summary=self._summarise(credibility.overall_score, credibility.flags),
        )

        # Stage 4 – save report
        if self.save_reports:
            self._reporter.save(report)

        logger.info(
            "Pipeline done: score=%.2f level=%s",
            credibility.overall_score,
            credibility.level.value,
        )
        return report

    def batch_run(
        self,
        news_items: List[NewsItem],
        propagation_graphs: Optional[Dict[str, PropagationGraph]] = None,
        batch_output_path: Optional[str] = None,
    ) -> List[DetectionReport]:
        """Run the detection pipeline on a *list* of news items.

        Args:
            news_items: Articles to process.
            propagation_graphs: Optional mapping of ``item_id`` →
                ``PropagationGraph``.
            batch_output_path: Path for the combined batch HTML report.

        Returns:
            List of ``DetectionReport`` objects in the same order as input.
        """
        graphs = propagation_graphs or {}
        reports: List[DetectionReport] = []

        for item in news_items:
            graph = graphs.get(item.item_id)
            report = self.run(item, propagation_graph=graph)
            reports.append(report)

        # Generate combined batch report
        if self.save_reports:
            self._reporter.process_batch(reports, output_path=batch_output_path)

        return reports

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise(score: float, flags: List[str]) -> str:
        if score >= 0.75:
            verdict = "This article appears credible."
        elif score >= 0.50:
            verdict = "This article has mixed credibility signals."
        elif score >= 0.25:
            verdict = "This article shows several low-credibility indicators."
        else:
            verdict = "This article is likely fake or highly unreliable."

        if flags:
            flag_str = ", ".join(flags[:5])
            return f"{verdict} Issues detected: {flag_str}."
        return verdict
