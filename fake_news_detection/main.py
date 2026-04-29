"""CLI entry point for the fake news detection system."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from fake_news_detection.models.news import NewsItem, PropagationGraph, PropagationNode
from fake_news_detection.orchestrator import Orchestrator


def _build_news_item(data: Dict[str, Any]) -> NewsItem:
    """Build a ``NewsItem`` from a plain dictionary."""
    published_at: Optional[datetime] = None
    if data.get("published_at"):
        try:
            published_at = datetime.fromisoformat(data["published_at"])
        except ValueError:
            pass
    item = NewsItem(
        title=data.get("title", ""),
        content=data.get("content", ""),
        source_url=data.get("source_url", ""),
        source_domain=data.get("source_domain", ""),
        author=data.get("author", ""),
        published_at=published_at,
        image_urls=data.get("image_urls", []),
        metadata=data.get("metadata", {}),
    )
    if data.get("item_id"):
        item.item_id = data["item_id"]
    return item


def _build_propagation_graph(data: Dict[str, Any]) -> PropagationGraph:
    """Build a ``PropagationGraph`` from a plain dictionary."""
    nodes = []
    for nd in data.get("nodes", []):
        ts = None
        if nd.get("timestamp"):
            try:
                ts = datetime.fromisoformat(nd["timestamp"])
            except ValueError:
                pass
        nodes.append(
            PropagationNode(
                node_id=nd.get("node_id", ""),
                platform=nd.get("platform", ""),
                user_id=nd.get("user_id", ""),
                timestamp=ts,
                parent_id=nd.get("parent_id"),
                share_count=nd.get("share_count", 0),
                like_count=nd.get("like_count", 0),
                comment_count=nd.get("comment_count", 0),
                is_verified_account=nd.get("is_verified_account", False),
            )
        )
    edges = [tuple(e) for e in data.get("edges", [])]
    return PropagationGraph(nodes=nodes, edges=edges)


def cmd_analyse(args: argparse.Namespace) -> None:
    """Analyse a single JSON news file."""
    with open(args.input, encoding="utf-8") as fh:
        data = json.load(fh)

    item = _build_news_item(data)
    graph: Optional[PropagationGraph] = None
    if data.get("propagation"):
        graph = _build_propagation_graph(data["propagation"])

    orch = Orchestrator(output_dir=args.output_dir, verbose=args.verbose)
    report = orch.run(item, propagation_graph=graph)

    cs = report.credibility_score
    print(f"\n{'='*60}")
    print(f"  Fake News Detection Result")
    print(f"{'='*60}")
    print(f"  Article : {report.news_item.title[:70]}")
    print(f"  Score   : {cs.overall_score:.2%} ({cs.level.value.upper()})")
    print(f"  Flags   : {', '.join(cs.flags) or 'none'}")
    if report.trace_analysis:
        m = report.trace_analysis.metrics
        print(f"  Reach   : {m.total_reach:,} | Depth: {m.depth} | Vel: {m.spread_velocity} shares/h")
    print(f"  Summary : {report.summary}")
    if args.output_dir:
        print(f"\n  Report saved to: {args.output_dir}/")
    print(f"{'='*60}\n")


def cmd_batch(args: argparse.Namespace) -> None:
    """Process a JSON file containing a list of news items."""
    with open(args.input, encoding="utf-8") as fh:
        items_data = json.load(fh)

    if not isinstance(items_data, list):
        print("ERROR: batch input must be a JSON array of news items.", file=sys.stderr)
        sys.exit(1)

    news_items = [_build_news_item(d) for d in items_data]
    graphs: Dict[str, PropagationGraph] = {}
    for d, item in zip(items_data, news_items):
        if d.get("propagation"):
            graphs[item.item_id] = _build_propagation_graph(d["propagation"])

    orch = Orchestrator(output_dir=args.output_dir, verbose=args.verbose)
    reports = orch.batch_run(news_items, propagation_graphs=graphs)

    print(f"\nBatch processed {len(reports)} articles:")
    for r in reports:
        cs = r.credibility_score
        tag = f"[{cs.level.value.upper():6}]" if cs else "[  N/A  ]"
        score = f"{cs.overall_score:.0%}" if cs else " N/A"
        print(f"  {tag} {score}  {r.news_item.title[:60]}")
    print(f"\nReports saved to: {args.output_dir}/\n")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fake-news-detect",
        description="Multi-agent fake news detection system",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug output")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- analyse ----
    p_analyse = sub.add_parser("analyse", help="Analyse a single news item JSON file")
    p_analyse.add_argument("input", help="Path to JSON file with news item")
    p_analyse.add_argument(
        "--output-dir", "-o", default="reports", help="Directory for HTML reports"
    )
    p_analyse.set_defaults(func=cmd_analyse)

    # ---- batch ----
    p_batch = sub.add_parser("batch", help="Process a batch of news items from a JSON array file")
    p_batch.add_argument("input", help="Path to JSON file with list of news items")
    p_batch.add_argument(
        "--output-dir", "-o", default="reports", help="Directory for HTML reports"
    )
    p_batch.set_defaults(func=cmd_batch)

    ns = parser.parse_args(argv)
    ns.func(ns)


if __name__ == "__main__":
    main()
