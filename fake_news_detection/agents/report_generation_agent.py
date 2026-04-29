"""Report Generation Agent.

Produces a rich, interactive HTML report from a ``DetectionReport``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import List, Optional

from fake_news_detection.agents.base_agent import BaseAgent
from fake_news_detection.models.news import (
    CredibilityLevel,
    DetectionReport,
)

# ---------------------------------------------------------------------------
# Colour palette for credibility levels
# ---------------------------------------------------------------------------
_LEVEL_COLORS = {
    CredibilityLevel.HIGH: ("#27ae60", "#d5f5e3"),
    CredibilityLevel.MEDIUM: ("#f39c12", "#fef9e7"),
    CredibilityLevel.LOW: ("#e67e22", "#fdebd0"),
    CredibilityLevel.FAKE: ("#c0392b", "#fadbd8"),
}

_LEVEL_LABELS = {
    CredibilityLevel.HIGH: "HIGH CREDIBILITY",
    CredibilityLevel.MEDIUM: "MEDIUM CREDIBILITY",
    CredibilityLevel.LOW: "LOW CREDIBILITY",
    CredibilityLevel.FAKE: "LIKELY FAKE",
}


class ReportGenerationAgent(BaseAgent):
    """Agent that renders ``DetectionReport`` objects as HTML.

    Single report:  ``process(report)``
    Batch report:   ``process_batch(reports, output_path)``

    The generated HTML is self-contained and uses only standard web
    technologies (HTML5 + CSS + vanilla JS) so it can be opened in any
    browser without a server.
    """

    def __init__(self, output_dir: str = "reports", verbose: bool = False) -> None:
        super().__init__(name="ReportGenerationAgent", verbose=verbose)
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, input_data: DetectionReport) -> str:  # type: ignore[override]
        """Render *input_data* as an HTML string.

        Args:
            input_data: The detection report to render.

        Returns:
            A complete HTML document as a string.
        """
        self._info(f"Generating report for '{input_data.item_id}'")
        html = self._render_single_report(input_data)
        return html

    def save(self, report: DetectionReport, path: Optional[str] = None) -> str:
        """Save a report to *path* (or auto-generate one under ``output_dir``).

        Returns the absolute file path of the saved report.
        """
        if path is None:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, f"report_{report.item_id}.html")
        html = self.process(report)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        self._info(f"Saved report → {path}")
        return path

    def process_batch(
        self, reports: List[DetectionReport], output_path: Optional[str] = None
    ) -> str:
        """Render a batch of reports into a single HTML dashboard.

        Args:
            reports: List of detection reports.
            output_path: Optional path to write the file; if omitted the
                         file is written to ``output_dir/batch_report.html``.

        Returns:
            The absolute file path of the saved batch report.
        """
        self._info(f"Generating batch report for {len(reports)} items")
        html = self._render_batch_report(reports)
        if output_path is None:
            os.makedirs(self.output_dir, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"batch_report_{ts}.html")
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        self._info(f"Batch report saved → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Single-report rendering
    # ------------------------------------------------------------------

    def _render_single_report(self, report: DetectionReport) -> str:
        cs = report.credibility_score
        level = cs.level if cs else CredibilityLevel.MEDIUM
        fg, bg = _LEVEL_COLORS[level]
        label = _LEVEL_LABELS[level]

        title = report.news_item.title or "(No title)"
        score_pct = f"{cs.overall_score * 100:.1f}%" if cs else "N/A"

        # Dimension table rows
        dim_rows = ""
        if cs:
            for dim in cs.dimensions:
                bar_width = int(dim.score * 100)
                dim_rows += f"""
                <tr>
                  <td>{dim.name.replace('_', ' ').title()}</td>
                  <td>
                    <div class="bar-bg">
                      <div class="bar-fill" style="width:{bar_width}%;background:{fg}"></div>
                    </div>
                  </td>
                  <td>{dim.score:.2f}</td>
                  <td class="explanation">{dim.explanation}</td>
                </tr>"""

        # Flags
        flags_html = ""
        if cs and cs.flags:
            tags = "".join(f'<span class="flag">{f}</span>' for f in cs.flags)
            flags_html = f'<div class="flags"><strong>Flags:</strong> {tags}</div>'

        # Propagation metrics
        trace_html = ""
        ta = report.trace_analysis
        if ta:
            m = ta.metrics
            trace_html = f"""
            <div class="section">
              <h2>📡 Propagation Analysis</h2>
              <table class="metrics-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Origin Platform</td><td>{ta.origin_platform}</td></tr>
                <tr><td>Graph Depth</td><td>{m.depth}</td></tr>
                <tr><td>Max Breadth</td><td>{m.breadth}</td></tr>
                <tr><td>Spread Velocity (first 24 h)</td><td>{m.spread_velocity} shares/h</td></tr>
                <tr><td>Total Reach</td><td>{m.total_reach:,} shares</td></tr>
                <tr><td>Platform Diversity</td><td>{m.platform_diversity} platform(s)</td></tr>
                <tr><td>Verified Share Ratio</td><td>{m.verified_share_ratio:.1%}</td></tr>
                <tr><td>Avg Cascade Length</td><td>{m.cascade_length:.2f}</td></tr>
              </table>
              {self._render_timeline(ta.timeline)}
            </div>"""

        # Entities and claims
        parsed = report.parsed_news
        entities_html = ""
        claims_html = ""
        if parsed:
            if parsed.detected_entities:
                entity_tags = "".join(
                    f'<span class="entity">{e}</span>' for e in parsed.detected_entities[:15]
                )
                entities_html = f'<div class="section"><h2>🏷️ Detected Entities</h2><div class="entity-list">{entity_tags}</div></div>'
            if parsed.detected_claims:
                claim_items = "".join(
                    f"<li>{c}</li>" for c in parsed.detected_claims
                )
                claims_html = f'<div class="section"><h2>📋 Extracted Claims</h2><ul class="claims">{claim_items}</ul></div>'

        return _HTML_TEMPLATE.format(
            title=_escape(title),
            score_pct=score_pct,
            level_label=label,
            fg_color=fg,
            bg_color=bg,
            item_id=report.item_id,
            source=_escape(report.news_item.source_domain or "Unknown"),
            author=_escape(report.news_item.author or "Unknown"),
            published=str(report.news_item.published_at or "Unknown"),
            generated=report.generated_at.strftime("%Y-%m-%d %H:%M UTC"),
            summary=_escape(report.summary or ""),
            dim_rows=dim_rows,
            flags_html=flags_html,
            trace_html=trace_html,
            entities_html=entities_html,
            claims_html=claims_html,
        )

    @staticmethod
    def _render_timeline(timeline: list) -> str:
        if not timeline:
            return ""
        rows = ""
        for event in timeline[:20]:
            rows += (
                f"<tr><td>{event.get('timestamp','')}</td>"
                f"<td>{event.get('platform','')}</td>"
                f"<td>{event.get('user_id','')}</td>"
                f"<td>{event.get('share_count', 0):,}</td></tr>"
            )
        return f"""
        <h3>Timeline (first 20 events)</h3>
        <table class="metrics-table">
          <tr><th>Timestamp</th><th>Platform</th><th>User</th><th>Shares</th></tr>
          {rows}
        </table>"""

    # ------------------------------------------------------------------
    # Batch-report rendering
    # ------------------------------------------------------------------

    def _render_batch_report(self, reports: List[DetectionReport]) -> str:
        cards = ""
        chart_labels: List[str] = []
        chart_scores: List[float] = []
        chart_colors: List[str] = []

        for r in reports:
            cs = r.credibility_score
            level = cs.level if cs else CredibilityLevel.MEDIUM
            fg, bg = _LEVEL_COLORS[level]
            label = _LEVEL_LABELS[level]
            score_pct = f"{cs.overall_score * 100:.1f}%" if cs else "N/A"
            flags = ", ".join(cs.flags[:3]) if cs and cs.flags else "—"

            short_title = (r.news_item.title or r.item_id)[:80]
            chart_labels.append(short_title[:30] + ("…" if len(short_title) > 30 else ""))
            chart_scores.append(round(cs.overall_score * 100, 1) if cs else 50.0)
            chart_colors.append(fg)

            cards += f"""
            <div class="card" style="border-left:5px solid {fg};background:{bg}">
              <div class="card-header">
                <span class="badge" style="background:{fg}">{label}</span>
                <span class="score">{score_pct}</span>
              </div>
              <h3 class="card-title">{_escape(short_title)}</h3>
              <p class="card-meta">
                Source: {_escape(r.news_item.source_domain or '—')} &nbsp;|&nbsp;
                Author: {_escape(r.news_item.author or '—')} &nbsp;|&nbsp;
                Flags: {flags}
              </p>
              <p class="card-summary">{_escape(r.summary or '')}</p>
            </div>"""

        labels_json = json.dumps(chart_labels)
        scores_json = json.dumps(chart_scores)
        colors_json = json.dumps(chart_colors)

        stats = self._batch_stats(reports)

        return _BATCH_TEMPLATE.format(
            total=len(reports),
            fake_count=stats["fake"],
            low_count=stats["low"],
            medium_count=stats["medium"],
            high_count=stats["high"],
            avg_score=f"{stats['avg_score'] * 100:.1f}%",
            cards=cards,
            generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            labels_json=labels_json,
            scores_json=scores_json,
            colors_json=colors_json,
        )

    @staticmethod
    def _batch_stats(reports: List[DetectionReport]) -> dict:
        counts = {l: 0 for l in CredibilityLevel}
        scores = []
        for r in reports:
            cs = r.credibility_score
            if cs:
                counts[cs.level] += 1
                scores.append(cs.overall_score)
        avg = sum(scores) / len(scores) if scores else 0.5
        return {
            "fake": counts[CredibilityLevel.FAKE],
            "low": counts[CredibilityLevel.LOW],
            "medium": counts[CredibilityLevel.MEDIUM],
            "high": counts[CredibilityLevel.HIGH],
            "avg_score": avg,
        }


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

def _escape(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Fake News Detection Report – {title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f8; color: #2c3e50; }}
  .header {{ background: {fg_color}; color: #fff; padding: 24px 40px; }}
  .header h1 {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 6px; }}
  .verdict {{ display: inline-block; background: rgba(255,255,255,.25);
              padding: 6px 16px; border-radius: 20px; font-size: 1rem;
              font-weight: 700; letter-spacing: .05em; }}
  .score-big {{ font-size: 3rem; font-weight: 800; margin: 10px 0; }}
  .container {{ max-width: 960px; margin: 30px auto; padding: 0 20px; }}
  .section {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
              padding: 24px; margin-bottom: 20px; }}
  .section h2 {{ font-size: 1.1rem; margin-bottom: 14px; color: #34495e; }}
  .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: .9rem; }}
  .meta-item {{ background: #f8f9fa; padding: 10px; border-radius: 6px; }}
  .meta-item strong {{ display: block; color: #7f8c8d; font-size: .78rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
  th, td {{ padding: 8px 12px; border-bottom: 1px solid #ecf0f1; text-align: left; }}
  th {{ background: #f8f9fa; font-weight: 600; }}
  .bar-bg {{ background: #ecf0f1; border-radius: 4px; height: 10px; width: 140px; }}
  .bar-fill {{ height: 10px; border-radius: 4px; }}
  .explanation {{ color: #7f8c8d; font-size: .82rem; max-width: 280px; }}
  .flag {{ display: inline-block; background: #e74c3c; color: #fff;
           padding: 2px 8px; border-radius: 12px; font-size: .78rem;
           margin: 2px; }}
  .flags {{ margin-top: 12px; }}
  .metrics-table td:first-child {{ font-weight: 600; width: 220px; }}
  .entity {{ display: inline-block; background: #eaf4fb; color: #2980b9;
             border: 1px solid #aed6f1; padding: 2px 8px; border-radius: 12px;
             font-size: .82rem; margin: 2px; }}
  .entity-list {{ line-height: 1.8; }}
  .claims li {{ margin: 6px 0; font-size: .9rem; line-height: 1.5; }}
  footer {{ text-align: center; font-size: .8rem; color: #95a5a6; margin: 30px 0; }}
</style>
</head>
<body>
<div class="header">
  <h1>{title}</h1>
  <div class="score-big">{score_pct}</div>
  <span class="verdict">{level_label}</span>
</div>
<div class="container">
  <!-- Metadata -->
  <div class="section">
    <h2>📰 Article Metadata</h2>
    <div class="meta-grid">
      <div class="meta-item"><strong>Item ID</strong>{item_id}</div>
      <div class="meta-item"><strong>Source</strong>{source}</div>
      <div class="meta-item"><strong>Author</strong>{author}</div>
      <div class="meta-item"><strong>Published</strong>{published}</div>
      <div class="meta-item"><strong>Report generated</strong>{generated}</div>
    </div>
    {flags_html}
    <p style="margin-top:14px;font-style:italic;color:#7f8c8d">{summary}</p>
  </div>

  <!-- Credibility dimensions -->
  <div class="section">
    <h2>🔍 Credibility Dimensions</h2>
    <table>
      <tr><th>Dimension</th><th>Score</th><th>Value</th><th>Notes</th></tr>
      {dim_rows}
    </table>
  </div>

  {trace_html}
  {entities_html}
  {claims_html}
</div>
<footer>Generated by Fake News Detection System &bull; {generated}</footer>
</body>
</html>
"""

_BATCH_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Fake News Detection – Batch Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f8; color: #2c3e50; }}
  .header {{ background: #2c3e50; color: #fff; padding: 20px 40px; }}
  .header h1 {{ font-size: 1.5rem; }}
  .header p {{ color: #bdc3c7; font-size: .9rem; margin-top: 4px; }}
  .stats {{ display: flex; gap: 16px; padding: 20px 40px; flex-wrap: wrap; }}
  .stat-box {{ background: #fff; border-radius: 8px; padding: 16px 24px;
               box-shadow: 0 1px 4px rgba(0,0,0,.08); text-align: center; flex: 1; min-width: 120px; }}
  .stat-box .num {{ font-size: 2rem; font-weight: 700; }}
  .stat-box .lbl {{ font-size: .8rem; color: #7f8c8d; margin-top: 4px; }}
  .chart-section {{ background: #fff; margin: 0 40px 20px; border-radius: 8px;
                    padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .chart-section h2 {{ font-size: 1rem; margin-bottom: 14px; }}
  .cards {{ padding: 0 40px 40px; display: grid; gap: 16px;
            grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); }}
  .card {{ background: #fff; border-radius: 8px; padding: 18px;
           box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .card-header {{ display: flex; justify-content: space-between; align-items: center;
                  margin-bottom: 8px; }}
  .badge {{ padding: 3px 10px; border-radius: 12px; color: #fff;
            font-size: .75rem; font-weight: 700; letter-spacing: .04em; }}
  .score {{ font-size: 1.5rem; font-weight: 800; }}
  .card-title {{ font-size: .95rem; font-weight: 600; margin-bottom: 6px; }}
  .card-meta {{ font-size: .8rem; color: #7f8c8d; margin-bottom: 6px; }}
  .card-summary {{ font-size: .85rem; font-style: italic; color: #555; }}
  footer {{ text-align: center; font-size: .8rem; color: #95a5a6; padding: 20px; }}
</style>
</head>
<body>
<div class="header">
  <h1>🕵️ Fake News Detection – Batch Report</h1>
  <p>Generated {generated} &bull; {total} articles analysed</p>
</div>

<div class="stats">
  <div class="stat-box"><div class="num">{total}</div><div class="lbl">Total Articles</div></div>
  <div class="stat-box" style="color:#27ae60"><div class="num">{high_count}</div><div class="lbl">High Credibility</div></div>
  <div class="stat-box" style="color:#f39c12"><div class="num">{medium_count}</div><div class="lbl">Medium Credibility</div></div>
  <div class="stat-box" style="color:#e67e22"><div class="num">{low_count}</div><div class="lbl">Low Credibility</div></div>
  <div class="stat-box" style="color:#c0392b"><div class="num">{fake_count}</div><div class="lbl">Likely Fake</div></div>
  <div class="stat-box"><div class="num">{avg_score}</div><div class="lbl">Avg Score</div></div>
</div>

<div class="chart-section">
  <h2>📊 Credibility Scores by Article</h2>
  <canvas id="scoreChart" height="120"></canvas>
</div>

<div class="cards">
{cards}
</div>

<footer>Fake News Detection System &bull; {generated}</footer>

<script>
const ctx = document.getElementById('scoreChart').getContext('2d');
new Chart(ctx, {{
  type: 'bar',
  data: {{
    labels: {labels_json},
    datasets: [{{
      label: 'Credibility Score (%)',
      data: {scores_json},
      backgroundColor: {colors_json},
      borderRadius: 4,
    }}]
  }},
  options: {{
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      y: {{ min: 0, max: 100, ticks: {{ callback: v => v + '%' }} }}
    }}
  }}
}});
</script>
</body>
</html>
"""
