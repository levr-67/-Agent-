"""Graph utilities for propagation analysis."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from fake_news_detection.models.news import PropagationGraph, PropagationNode


def build_adjacency(graph: PropagationGraph) -> Dict[str, List[str]]:
    """Return a forward-adjacency dict {parent_id: [child_id, ...]}."""
    adj: Dict[str, List[str]] = defaultdict(list)
    for src, dst in graph.edges:
        adj[src].append(dst)
    return dict(adj)


def compute_depth(graph: PropagationGraph) -> int:
    """Return the maximum depth (longest path from any root)."""
    if not graph.nodes:
        return 0
    adj = build_adjacency(graph)
    roots = [n.node_id for n in graph.get_roots()]
    if not roots:
        roots = [graph.nodes[0].node_id]

    max_depth = 0
    for root in roots:
        queue: deque = deque([(root, 0)])
        visited: Set[str] = set()
        while queue:
            nid, depth = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            max_depth = max(max_depth, depth)
            for child in adj.get(nid, []):
                queue.append((child, depth + 1))
    return max_depth


def compute_breadth(graph: PropagationGraph) -> int:
    """Return the maximum number of nodes at any single BFS level."""
    if not graph.nodes:
        return 0
    adj = build_adjacency(graph)
    roots = [n.node_id for n in graph.get_roots()]
    if not roots:
        roots = [graph.nodes[0].node_id]

    level_counts: Dict[int, int] = defaultdict(int)
    for root in roots:
        queue: deque = deque([(root, 0)])
        visited: Set[str] = set()
        while queue:
            nid, level = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            level_counts[level] += 1
            for child in adj.get(nid, []):
                queue.append((child, level + 1))
    return max(level_counts.values(), default=0)


def compute_platform_diversity(graph: PropagationGraph) -> int:
    """Return the number of distinct platforms in the graph."""
    return len({n.platform for n in graph.nodes if n.platform})


def compute_verified_ratio(graph: PropagationGraph) -> float:
    """Return fraction of nodes that belong to verified accounts."""
    if not graph.nodes:
        return 0.0
    verified = sum(1 for n in graph.nodes if n.is_verified_account)
    return verified / len(graph.nodes)


def compute_cascade_length(graph: PropagationGraph) -> float:
    """Return the average path length from each root to its leaf nodes."""
    if not graph.nodes:
        return 0.0
    adj = build_adjacency(graph)
    roots = [n.node_id for n in graph.get_roots()]
    if not roots:
        return 0.0

    total_length = 0.0
    total_paths = 0
    for root in roots:
        stack: List[Tuple[str, int]] = [(root, 0)]
        visited: Set[str] = set()
        while stack:
            nid, depth = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            children = adj.get(nid, [])
            if not children:
                total_length += depth
                total_paths += 1
            for child in children:
                stack.append((child, depth + 1))
    return total_length / total_paths if total_paths else 0.0


def build_timeline(graph: PropagationGraph) -> List[dict]:
    """Return nodes sorted by timestamp as a timeline list."""
    nodes_with_ts = [n for n in graph.nodes if n.timestamp is not None]
    nodes_with_ts.sort(key=lambda n: n.timestamp)  # type: ignore[arg-type]
    return [
        {
            "node_id": n.node_id,
            "platform": n.platform,
            "user_id": n.user_id,
            "timestamp": n.timestamp.isoformat() if n.timestamp else None,
            "share_count": n.share_count,
        }
        for n in nodes_with_ts
    ]


def render_ascii_graph(graph: PropagationGraph) -> str:
    """Render a simple ASCII tree representation of the propagation graph."""
    if not graph.nodes:
        return "(empty graph)"

    adj = build_adjacency(graph)
    roots = graph.get_roots()
    if not roots:
        roots = graph.nodes[:1]

    lines: List[str] = []

    def _walk(node_id: str, prefix: str, is_last: bool) -> None:
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{node_id}")
        children = adj.get(node_id, [])
        child_prefix = prefix + ("    " if is_last else "│   ")
        for idx, child in enumerate(children):
            _walk(child, child_prefix, idx == len(children) - 1)

    for root in roots:
        lines.append(root.node_id)
        children = adj.get(root.node_id, [])
        for idx, child in enumerate(children):
            _walk(child, "", idx == len(children) - 1)
    return "\n".join(lines)
