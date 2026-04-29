# 🕵️ 基于多Agent协作的虚假新闻识别与溯源系统

> A Multi-Agent System for Fake News Detection, Credibility Scoring, and Propagation Analysis

---

## Overview

This project implements a modular **multi-agent pipeline** for detecting and analysing fake news. Each agent is responsible for a distinct analytical stage; all agents are coordinated by a central **Orchestrator** that also supports **batch processing**.

```
NewsItem ──► MultimodalParsingAgent ──► ParsedNews
                                            │
                                            ▼
                              CredibilityEvaluationAgent ──► CredibilityScore
                                            │
             PropagationGraph ──► TraceAnalysisAgent ──► TraceAnalysis
                                            │
                              ReportGenerationAgent ──► HTML Report
```

---

## Agents

| Agent | Input | Output | Responsibility |
|---|---|---|---|
| **MultimodalParsingAgent** | `NewsItem` | `ParsedNews` | Linguistic analysis, entity & claim extraction, media accounting |
| **CredibilityEvaluationAgent** | `ParsedNews` | `CredibilityScore` | Six-dimension weighted credibility scoring with flag detection |
| **TraceAnalysisAgent** | `PropagationGraph` | `TraceAnalysis` | Propagation depth/breadth/velocity, timeline, platform diversity |
| **ReportGenerationAgent** | `DetectionReport` | HTML file | Interactive single & batch HTML reports with chart visualisation |

---

## Credibility Dimensions

The `CredibilityEvaluationAgent` scores each article on six weighted dimensions:

| Dimension | Weight | Description |
|---|---|---|
| Source credibility | 30 % | Domain reputation (known reliable / unreliable domains) |
| Writing style | 20 % | Caps ratio, sensational vocabulary, exclamation marks |
| Content quality | 20 % | Article length, readability, presence of quoted sources |
| Author presence | 10 % | Whether a named author is credited |
| Claim substantiation | 15 % | Numeric evidence and direct quotations |
| Temporal freshness | 5 % | Whether a publication date is provided |

**Credibility levels:** HIGH (≥ 0.75) · MEDIUM (0.50–0.74) · LOW (0.25–0.49) · FAKE (< 0.25)

---

## Project Structure

```
fake_news_detection/
├── agents/
│   ├── base_agent.py                   # Abstract base class
│   ├── multimodal_parsing_agent.py     # Stage 1 – parsing
│   ├── credibility_evaluation_agent.py # Stage 2 – credibility scoring
│   ├── trace_analysis_agent.py         # Stage 3 – propagation analysis
│   └── report_generation_agent.py      # Stage 4 – HTML report generation
├── models/
│   └── news.py                         # All data models / dataclasses
├── utils/
│   ├── text_utils.py                   # NLP helpers, domain lists
│   └── graph_utils.py                  # Graph metrics, ASCII visualisation
├── orchestrator.py                     # Pipeline coordinator
└── main.py                             # CLI entry point

examples/
├── credible_article.json               # Sample credible news item + propagation
├── fake_article.json                   # Sample fake news item + propagation
└── batch_articles.json                 # Batch of 4 articles for demo

tests/
├── test_models.py
├── test_agents.py
├── test_orchestrator.py
└── test_utils.py
```

---

## Installation

```bash
pip install -e .
```

No external ML dependencies required – the system uses rule-based heuristics and
standard library modules only.

---

## CLI Usage

### Analyse a single article

```bash
fake-news-detect analyse examples/credible_article.json
fake-news-detect analyse examples/fake_article.json --output-dir reports/
```

### Batch processing

```bash
fake-news-detect batch examples/batch_articles.json --output-dir reports/
```

### Example output

```
============================================================
  Fake News Detection Result
============================================================
  Article : Scientists Discover New Species of Deep-Sea Fish in Pacific Ocean
  Score   : 96.00% (HIGH)
  Flags   : none
  Reach   : 5,372 | Depth: 2 | Vel: 1.0 shares/h
  Summary : This article appears credible.

  Report saved to: reports/
============================================================
```

```
============================================================
  Fake News Detection Result
============================================================
  Article : SHOCKING!!! GOVERNMENT HIDING CURE FOR ALL DISEASES
  Score   : 19.50% (FAKE)
  Flags   : EXCESSIVE_CAPS, SENSATIONALISM, EXCESSIVE_EXCLAMATIONS,
            VERY_SHORT_CONTENT, NO_AUTHOR, NO_PUBLICATION_DATE
  Reach   : 21,950 | Depth: 2 | Vel: 1.71 shares/h
  Summary : This article is likely fake or highly unreliable.
============================================================
```

---

## Python API

```python
from fake_news_detection import Orchestrator, NewsItem, PropagationGraph, PropagationNode
from datetime import datetime

# Build a news item
item = NewsItem(
    title="Central Bank Raises Rates",
    content="The central bank raised rates by 0.5% ...",
    source_url="https://reuters.com/finance/rates",
    author="Jane Smith",
    published_at=datetime(2024, 3, 15),
)

# Optional propagation graph
graph = PropagationGraph(
    nodes=[
        PropagationNode(node_id="orig", platform="twitter",
                        timestamp=datetime(2024, 3, 15, 9, 0), share_count=1000,
                        is_verified_account=True),
        PropagationNode(node_id="rt1", platform="facebook",
                        timestamp=datetime(2024, 3, 15, 10, 0), parent_id="orig",
                        share_count=200),
    ],
    edges=[("orig", "rt1")],
)

# Run the full pipeline
orch = Orchestrator(output_dir="reports")
report = orch.run(item, propagation_graph=graph)

print(f"Score: {report.credibility_score.overall_score:.2%}")
print(f"Level: {report.credibility_score.level.value}")

# Batch mode
reports = orch.batch_run([item1, item2, item3])
```

---

## Input JSON Format

```json
{
  "title": "Article title",
  "content": "Full article text...",
  "source_url": "https://example.com/article",
  "author": "Author Name",
  "published_at": "2024-03-15T10:00:00",
  "image_urls": ["https://example.com/img.jpg"],
  "metadata": {},
  "propagation": {
    "nodes": [
      {
        "node_id": "orig",
        "platform": "twitter",
        "user_id": "user_handle",
        "timestamp": "2024-03-15T09:00:00",
        "share_count": 1000,
        "is_verified_account": true
      }
    ],
    "edges": [["orig", "child1"]]
  }
}
```

For batch processing, provide a JSON array of such objects.

---

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

82 tests covering models, all four agents, utilities, and the orchestrator.

---

## Architecture Notes

- **Stateless agents** – each agent processes one item at a time with no shared mutable state, enabling safe parallel or batch use.
- **Pluggable scoring** – the heuristic scorers in `CredibilityEvaluationAgent` and the domain lists in `text_utils.py` can be replaced by ML models without changing the agent interface.
- **Self-contained HTML reports** – generated reports use only HTML5/CSS/vanilla JS (Chart.js loaded from CDN for the batch dashboard).
- **Zero heavy dependencies** – the entire system runs on Python ≥ 3.9 with no external packages beyond `pytest` for testing.
