"""Graph helpers for segment linking and grouping."""
from __future__ import annotations
from typing import Iterable, Mapping, Any, Dict, List, Set, Tuple
from collections import defaultdict
import numpy as np

from .geometry import compute_angle  # one-way dependency on geometry

Link = Mapping[str, Any]

def classify_links(
    links: Iterable[Link],
    system_segments: Set[str],
    segment_map: Mapping[str, Mapping[str, Any]],
    bent_thresh_deg: float = 30.0,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Split links into 'end_to_end' (plus bent_end < thresh) vs 'branch' links.
    Returns (end_to_end_links, branch_links) as lists of (seg1, seg2).
    """
    e2e: list[tuple[str, str]] = []
    branch: list[tuple[str, str]] = []
    for L in links:
        s1, s2 = L["seg1"], L["seg2"]
        if s1 not in system_segments or s2 not in system_segments:
            continue
        how = L.get("how", "")
        if how == "end_to_end":
            e2e.append((s1, s2))
        elif how == "bent_end":
            ang = compute_angle(segment_map[s1]["points"], segment_map[s2]["points"])
            if ang is not None and ang < bent_thresh_deg:
                e2e.append((s1, s2))
            else:
                branch.append((s1, s2))
        else:
            branch.append((s1, s2))
    return e2e, branch

def build_end_to_end_map(end_to_end_links: Iterable[tuple[str, str]]) -> Dict[str, List[str]]:
    """Adjacency map for end-to-end connectivity."""
    m: Dict[str, List[str]] = defaultdict(list)
    for a, b in end_to_end_links:
        m[a].append(b)
        m[b].append(a)
    return m

def merge_end_links(start_seg: str, end_map: Mapping[str, list[str]], visited: Set[str]) -> Set[str]:
    """Return the full connected set (group) reachable via end_to_end links from start_seg."""
    stack = [start_seg]
    group: Set[str] = set()
    while stack:
        s = stack.pop()
        if s in visited:
            continue
        visited.add(s)
        group.add(s)
        for nbr in end_map.get(s, []):
            if nbr not in visited:
                stack.append(nbr)
    return group

def connected_components_end_to_end(
    system_segments: Iterable[str],
    end_map: Mapping[str, list[str]],
) -> list[set[str]]:
    """All end_to_end groups across system_segments."""
    comps: list[set[str]] = []
    visited: Set[str] = set()
    for seg in system_segments:
        if seg not in visited:
            comps.append(merge_end_links(seg, end_map, visited))
    return comps

def build_branch_adjacency(branch_links: Iterable[tuple[str, str]]) -> Dict[str, List[str]]:
    """Adjacency over 'branch' links (non-merged connectivity)."""
    m: Dict[str, List[str]] = defaultdict(list)
    for a, b in branch_links:
        m[a].append(b)
        m[b].append(a)
    return m
