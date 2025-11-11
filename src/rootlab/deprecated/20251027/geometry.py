"""Geometry helpers: lengths, widths, distances for root segments."""
from __future__ import annotations
import math
from typing import Iterable, Tuple, Mapping, Any
import numpy as np

Point = Tuple[float, float]

def segment_length(points: Iterable[Point]) -> float:
    """Compute polyline length from ordered points [(x1,y1), (x2,y2), ...]."""
    pts = list(points)
    if len(pts) < 2:
        return 0.0
    return sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(pts, pts[1:]))

def segment_width(meta: Mapping[str, Any]) -> float:
    """Width from metadata with keys 'width_left' and 'width_right'."""
    wl = float(meta.get("width_left", 0.0))
    wr = float(meta.get("width_right", 0.0))
    return abs(wr - wl)

def compute_angle(points1, points2):
    """Compute angle between two connected segments (degrees)."""
    if len(points1) < 2 or len(points2) < 2:
        return None
    p1_last, p1_prev = np.array(points1[-1]), np.array(points1[-2])
    p2_first, p2_next = np.array(points2[0]), np.array(points2[1])
    v1 = p1_last - p1_prev
    v2 = p2_next - p2_first
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return None
    dot = np.dot(v1, v2)
    cos_angle = max(-1, min(1, dot / (norm1 * norm2)))
    return math.degrees(math.acos(cos_angle))

# Define public functions
__all__ = ["segment_length", "segment_width", "compute_angle"]
