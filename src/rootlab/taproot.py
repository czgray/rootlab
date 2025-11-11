# Copyright 2025 Christian Gray
#
# File: taproot.py
#
# Author: Christian Gray (czgray@princeton.edu)
# 
# RootLab is a free, open-source package developed by Christian Gray
# as part of his Ph.D. research at Princeton University.
# It is designed to aid in the analysis of plant root architecture.
# If you use this package in your work, please cite it.
# See the "NOTICE" file for citation details.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterable, Union, Sequence
from dataclasses import dataclass

import math
import numpy as np
import networkx as nx
import json

from .geometry import segment_length as seg_len, segment_width as seg_w, compute_angle
from .graph import (
    classify_links,                   # splits links into end_to_end vs branch links (bent_end threshold)
    build_end_to_end_map,             # map for end_to_end merging
    connected_components_end_to_end,  # branch groups as connected comps under end_to_end(+bent) links
    build_branch_adjacency,           # (import kept for parity; not strictly required here)
)
from .viz import _VIZ, _collect_segment_drawables, _auto_limits, plot_taproot_candidates

@dataclass
class TaprootParams:
    # Link classification
    bent_thresh_deg: float = 30.0

    # Axis inference from thick segments
    axis_width_quantile: float = 0.85   # top 15% widths define axis

    # Bridging between branch groups
    bridge_max_gap_px: float = 35.0
    bridge_axis_angle_max_deg: float = 40.0
    max_pct_fabricated: float = 0.40

    # Search limits / candidates
    start_k_groups: int = 5             # try top-K crown-most thick groups
    kth_paths_per_start: int = 64       # soft cap per start to enumerate

    # Scoring (width/length reward, depth reward, thinning/bridge penalties)
    alpha_width: float = 1.0
    w_W: float = 1.0
    w_D: float = 0.25
    w_T: float = 0.4
    bridge_cost_per_px: float = 1.0

@dataclass
class TaprootResult:
    sample_id: str
    system_index: int
    best_idx: int                      # index within candidates (0-based), or -1 if none
    candidates: List[Dict[str, Any]]   # sorted (top few) candidates for this system

# -------------------------
# Small helpers
# -------------------------
def _segment_centroid(points: Iterable[Iterable[float]]) -> np.ndarray:
    pts = np.asarray(list(points), dtype=float)
    return pts.mean(axis=0) if len(pts) else np.array([np.nan, np.nan])

def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    cos = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return math.degrees(math.acos(cos))

def _weighted_pca_axis(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """First principal direction with weights w >= 0."""
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0, None)
    wsum = w.sum()
    if wsum <= 0 or len(X) < 2:
        # unweighted fallback
        Xc = X - X.mean(axis=0, keepdims=True)
        cov = (Xc.T @ Xc) / max(len(X) - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, np.argmax(vals)]
        return v / (np.linalg.norm(v) + 1e-12)
    mu = (w[:, None] * X).sum(axis=0) / wsum
    Xc = X - mu
    cov = (Xc * w[:, None]).T @ Xc / wsum
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]
    return v / (np.linalg.norm(v) + 1e-12)

def _project_along(v_axis: np.ndarray, pts: np.ndarray) -> np.ndarray:
    v = v_axis / (np.linalg.norm(v_axis) + 1e-12)
    return pts @ v

# -------------------------
# Core
# -------------------------
def find_taproot_from_json(
    json_file: str | Path,
    sample_id: Union[str, Sequence[str]],   # accept str OR list/tuple of str
    system_id: Optional[str] = None,
    params: TaprootParams = TaprootParams(),
) -> Dict[Tuple[str, int], TaprootResult]:
    """
    Build branch groups the same way as your DFS flow (end_to_end + bent_end),
    infer a main axis from the thickest segments, allow small axis-aligned
    bridges between groups, enumerate forward-moving paths, and score them.

    Returns: dict[(sample_id, system_index)] -> TaprootResult
      Each candidate includes:
        - "polyline" (list[[x,y], ...]) along the path (rep-seg chains + straight bridges)
        - "mask_seg_ids" (list[str]) : union of segment IDs on the path (for labeling)
        - observed_length_px, fabricated_length_px, pct_fabricated
        - depth_along_axis_px (projection span along the inferred axis)
        - score, score_parts
    """
    import json
    path = Path(json_file)
    data = json.loads(path.read_text(encoding="utf-8"))

    # Normalize to a list so later code can use list semantics safely
    _sample_id_list = [sample_id] if isinstance(sample_id, str) else list(sample_id)
    
    results: Dict[Tuple[str, int], TaprootResult] = {}
    
    for img in data.get("images", []):
        sid = img["info"].get("experiment")
        if sid not in _sample_id_list:
            continue

        # Flatten segments and links for this sample
        seg_map: Dict[str, Dict[str, Any]] = {
            str(seg["name"]): seg for root in img["roots"] for seg in root["segments"]
        }
        links: List[Dict[str, Any]] = [lk for root in img["roots"] for lk in root["links"]]

        # Systems = connected components of (segments, links)
        G = nx.Graph()
        G.add_nodes_from(seg_map.keys())
        G.add_edges_from((str(lk["seg1"]), str(lk["seg2"])) for lk in links)
        components = sorted(nx.connected_components(G), key=len, reverse=True)

        # honor system_id if only one sample is requested
        if system_id is not None and len(_sample_id_list) == 1:
            try:
                idx = int(str(system_id).split("_")[-1]) - 1
            except Exception:
                idx = 0
            sys_indices = [max(0, min(idx, len(components) - 1))]
        else:
            sys_indices = list(range(len(components)))

        for sys_idx in sys_indices:
            system_segments = set(components[sys_idx])
            if not system_segments:
                results[(sid, sys_idx)] = TaprootResult(sid, sys_idx, -1, [])
                continue

            # Link classification (your logic)
            e2e_links, branch_links = classify_links(
                links, system_segments, seg_map, bent_thresh_deg=params.bent_thresh_deg
            )
            e2e_map = build_end_to_end_map(e2e_links)

            # Branch groups under end_to_end(+bent) connectivity
            groups = connected_components_end_to_end(system_segments, e2e_map)  # list[set[str]]
            if not groups:
                results[(sid, sys_idx)] = TaprootResult(sid, sys_idx, -1, [])
                continue

            # Per-segment stats
            seg_ids = list(system_segments)
            centroids = np.array([_segment_centroid(seg_map[s]["points"]) for s in seg_ids], float)
            widths    = np.array([max(seg_w(seg_map[s]["meta"]), 0.0) for s in seg_ids], float)
            lengths   = np.array([seg_len(seg_map[s]["points"]) for s in seg_ids], float)
            weights   = widths * np.maximum(lengths, 1e-6)

            # Axis from thick segments (quantile of width)
            q = np.quantile(widths[widths > 0], params.axis_width_quantile) if np.any(widths > 0) else 0.0
            mask = widths >= q if q > 0 else np.ones_like(widths, bool)
            axis = _weighted_pca_axis(centroids[mask], weights[mask])

            # Orient axis so widths *decrease* as projection increases (tapering)
            projs = _project_along(axis, centroids)
            if np.cov(projs, widths, bias=True)[0, 1] > 0:  # widths increase with proj -> flip
                axis = -axis

            # Group-level summaries
            group_stats = []
            for gid, g in enumerate(groups):
                g_list = list(g)
                g_widths  = np.array([max(seg_w(seg_map[s]["meta"]), 0.0) for s in g_list], float)
                g_lengths = np.array([seg_len(seg_map[s]["points"]) for s in g_list], float)
                g_weights = g_widths * np.maximum(g_lengths, 1e-6)
                g_centers = np.array([_segment_centroid(seg_map[s]["points"]) for s in g_list], float)

                proj = _project_along(axis, g_centers)
                proj_min = float(np.min(proj)) if len(proj) else 0.0
                proj_max = float(np.max(proj)) if len(proj) else 0.0

                tot_len = float(np.sum(g_lengths))
                w_avg   = float(np.sum(g_widths * g_lengths) / max(tot_len, 1e-9)) if tot_len > 0 else 0.0

                rep_idx = int(np.argmax(g_weights)) if len(g_weights) else 0
                rep_seg = g_list[rep_idx]
                rep_pts = np.asarray(seg_map[rep_seg]["points"], float)

                # orient rep_pts to increase along axis
                if rep_pts.shape[0] >= 2:
                    pr = _project_along(axis, rep_pts)
                    if pr[-1] < pr[0]:
                        rep_pts = rep_pts[::-1]

                group_stats.append({
                    "gid": gid,
                    "seg_ids": g_list,
                    "rep_seg": rep_seg,
                    "rep_pts": rep_pts,
                    "width": w_avg,
                    "length": tot_len,
                    "proj_min": proj_min,
                    "proj_max": proj_max,
                })

            # Group graph: natural adjacency from branch_links
            seg_to_gid = {s: gid for gid, g in enumerate(groups) for s in g}
            adj: Dict[int, set[int]] = {gid: set() for gid in range(len(groups))}
            edge_info: Dict[Tuple[int, int], Dict[str, float | bool]] = {}

            for (a, b) in branch_links:
                ga, gb = seg_to_gid[a], seg_to_gid[b]
                if ga == gb:
                    continue
                u, v = sorted((ga, gb))
                adj[u].add(v); adj[v].add(u)
                edge_info[(u, v)] = {"is_bridge": False, "gap_len": 0.0, "angle": 0.0}

            # Add small axis-aligned bridges between groups
            def endpoints(poly: np.ndarray):
                return (poly[0], poly[-1]) if poly.shape[0] else (None, None)

            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if j in adj[i]:
                        continue
                    p1s, p1e = endpoints(group_stats[i]["rep_pts"])
                    p2s, p2e = endpoints(group_stats[j]["rep_pts"])
                    if p1s is None or p2s is None:
                        continue
                    best_gap, best_vec = None, None
                    for a in (p1s, p1e):
                        for b in (p2s, p2e):
                            vec = b - a
                            gap = float(np.linalg.norm(vec))
                            if best_gap is None or gap < best_gap:
                                best_gap, best_vec = gap, vec
                    if best_gap is None:
                        continue
                    ang = _angle_deg(best_vec, axis)
                    if best_gap <= params.bridge_max_gap_px and ang <= params.bridge_axis_angle_max_deg:
                        adj[i].add(j); adj[j].add(i)
                        u, v = (i, j) if i < j else (j, i)
                        edge_info[(u, v)] = {"is_bridge": True, "gap_len": best_gap, "angle": ang}

            # Start groups: crown-most (small proj_min) but still wide
            widths_all = np.array([g["width"] for g in group_stats])
            wq30 = float(np.quantile(widths_all, 0.70)) if len(widths_all) else 0.0
            sorted_groups = sorted(group_stats, key=lambda g: (g["proj_min"], -g["width"]))
            starts = [g for g in sorted_groups if g["width"] >= wq30][:max(1, params.start_k_groups)]

            def path_score(path: List[int]) -> Tuple[float, Dict[str, float]]:
                # Reward wide-long groups and forward depth; penalize thickening + bridges
                W = sum((group_stats[g]["width"] ** params.alpha_width) * group_stats[g]["length"] for g in path)
                D = max(group_stats[path[-1]]["proj_max"] - group_stats[path[0]]["proj_min"], 0.0)
                T = 0.0
                for i in range(len(path) - 1):
                    w0 = group_stats[path[i]]["width"]
                    w1 = group_stats[path[i + 1]]["width"]
                    if w1 > w0:
                        T += (w1 - w0)
                B = 0.0
                for i in range(len(path) - 1):
                    u, v = sorted((path[i], path[i + 1]))
                    meta = edge_info.get((u, v))
                    if meta and meta.get("is_bridge"):
                        B += float(meta["gap_len"]) * params.bridge_cost_per_px
                score = params.w_W * W + params.w_D * D - params.w_T * T - B
                return score, {"W": W, "D": D, "T": T, "B": B}

            def enumerate_paths_from(start_gid: int) -> List[List[int]]:
                # Forward-only (non-decreasing proj_min) simple paths
                paths: List[List[int]] = []
                stack = [(start_gid, [start_gid])]
                cap = params.kth_paths_per_start
                while stack and len(paths) < cap:
                    node, path_nodes = stack.pop()
                    forward_neighbors = [
                        nb for nb in adj.get(node, [])
                        if nb not in path_nodes and
                           group_stats[nb]["proj_min"] >= group_stats[node]["proj_min"]
                    ]
                    if forward_neighbors:
                        for nb in forward_neighbors:
                            stack.append((nb, path_nodes + [nb]))
                    # record current prefix as a candidate too
                    paths.append(path_nodes)
                # light dedup
                uniq, seen = [], set()
                for p in paths:
                    key = (p[-1], tuple(p))
                    if key not in seen:
                        seen.add(key); uniq.append(p)
                return uniq

            candidates = []
            for s in starts:
                for path_groups in enumerate_paths_from(s["gid"]):
                    sc, parts = path_score(path_groups)

                    # compute fabricated length (bridges) & observed length (within groups)
                    fab = 0.0
                    for i in range(len(path_groups) - 1):
                        u, v = sorted((path_groups[i], path_groups[i + 1]))
                        meta = edge_info.get((u, v))
                        if meta and meta.get("is_bridge"):
                            fab += float(meta["gap_len"])

                    # union mask of segments on the path
                    seg_union = {s for g in path_groups for s in groups[g]}

                    # polyline = concat of each group's rep polyline with straight-line links
                    pts_list = []
                    for idx, gix in enumerate(path_groups):
                        gp = np.asarray(group_stats[gix]["rep_pts"], float)
                        if gp.shape[0]:
                            if pts_list:
                                a = pts_list[-1][-1]
                                b = gp[0]
                                if np.linalg.norm(b - a) > 1e-6:
                                    pts_list.append(np.vstack([a, b]))  # straight bridge
                            pts_list.append(gp)
                    poly = np.vstack(pts_list) if pts_list else np.zeros((0, 2), float)

                    # observed length = sum len(rep polylines)
                    obs_len = 0.0
                    for gix in path_groups:
                        gp = np.asarray(group_stats[gix]["rep_pts"], float)
                        if gp.shape[0] >= 2:
                            obs_len += float(np.sum(np.linalg.norm(np.diff(gp, axis=0), axis=1)))

                    depth = float(group_stats[path_groups[-1]]["proj_max"] - group_stats[path_groups[0]]["proj_min"])
                    pct_fab = fab / max(obs_len + fab, 1e-9)

                    candidates.append({
                        "path_groups": path_groups,
                        "score": sc,
                        "score_parts": parts,
                        "mask_seg_ids": list(seg_union),
                        "polyline": poly.tolist(),
                        "observed_length_px": obs_len,
                        "fabricated_length_px": fab,
                        "pct_fabricated": pct_fab,
                        "depth_along_axis_px": depth,
                        "start_gid": path_groups[0],
                        "end_gid": path_groups[-1],
                    })

            # rank and take top few (filter by fabrication if possible)
            candidates.sort(key=lambda c: (-c["score"], c["pct_fabricated"], -c["depth_along_axis_px"]))
            filtered = [c for c in candidates if c["pct_fabricated"] <= params.max_pct_fabricated]
            final = filtered if filtered else candidates
            best_idx = 0 if final else -1
            results[(sid, sys_idx)] = TaprootResult(sample_id=sid, system_index=sys_idx,
                                                    best_idx=best_idx, candidates=final[:5])
    return results

# ==== super taproot combiner ================================================
from typing import Sequence, Union

@dataclass
class SuperTaprootParams:
    # Pairwise “compatibility” thresholds between taproot candidates
    max_angle_diff_deg: float = 12.0      # parallelism tolerance
    max_lateral_sep_px: float = 120.0      # how far apart sideways they can be
    max_gap_along_axis_px: float = 400.0  # how far apart in sequence they can be
    # Bridging when we actually stitch polylines
    bridge_max_gap_px: float = 400.0      # straight bridge length allowed when stitching
    # Keep only “solid” components
    max_pct_fabricated: float = 0.60      # drop super taproots mostly made of bridges

@dataclass
class SuperTaprootResult:
    sample_id: str
    polyline: List[List[float]]
    mask_seg_ids: List[str]
    used: List[Tuple[int, int]]          # [(system_index, candidate_rank)] included
    observed_length_px: float
    fabricated_length_px: float
    pct_fabricated: float
    depth_along_axis_px: float
    axis_vec: Tuple[float, float]

def _pc1(points: np.ndarray) -> np.ndarray:
    """Unit first principal axis for Nx2 points."""
    X = points - points.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / max(len(X)-1, 1)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]
    return v / (np.linalg.norm(v) + 1e-12)

def _angle_between_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    c = float(np.clip(abs(np.dot(u, v)), -1.0, 1.0))  # abs → directionless parallelism
    return math.degrees(math.acos(c))

def _proj(u: np.ndarray, P: np.ndarray) -> np.ndarray:
    u = u / (np.linalg.norm(u) + 1e-12)
    return P @ u

def _perp_dist_between_centroids(u: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> float:
    """Distance between centroids perpendicular to u."""
    d = c2 - c1
    u = u / (np.linalg.norm(u) + 1e-12)
    return float(np.linalg.norm(d - (np.dot(d, u) * u)))

def _concat_with_straight_bridge(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return concatenated polyline and fabricated (bridge) length."""
    if A.size == 0:
        return B.copy(), 0.0
    a = A[-1]; b = B[0]
    gap = float(np.linalg.norm(b - a))
    if gap > 1e-9:
        bridge = np.vstack([a, b])
        return np.vstack([A, bridge, B]), gap
    return np.vstack([A, B]), 0.0

def build_super_taproot(
    *,
    json_file: str | Path,
    sample_id: str,
    taproot_results: Dict[Tuple[str, int], TaprootResult],
    params: SuperTaprootParams = SuperTaprootParams(),
    per_system_topk: int = 1,      # take top-k per system into the pool
) -> List[SuperTaprootResult]:
    """
    Combine per-system taproot candidates into one or more 'super taproots'
    for the WHOLE sample by merging candidates that are:
      - nearly parallel (angle ≤ max_angle_diff_deg),
      - in sequence along a shared axis (gap along axis ≤ max_gap_along_axis_px),
      - not far apart sideways (lateral sep ≤ max_lateral_sep_px).
    Returns 0..N super taproots sorted by (observed+fabricated) length desc.
    """
    # ----- 0) Gather the candidate pool across all systems for this sample
    pool = []  # list of dicts with derived geometry + source info
    for (sid, sys_idx), res in taproot_results.items():
        if sid != sample_id or not res.candidates:
            continue
        for j, cand in enumerate(res.candidates[:max(0, per_system_topk)]):
            P = np.asarray(cand.get("polyline", []), float)
            if P.shape[0] < 2:
                continue
            # axis from polyline points
            u = _pc1(P)
            # orient so projection increases along the path
            t = _proj(u, P)
            if t[-1] < t[0]:
                u = -u
                P = P[::-1]
                t = t[::-1]
            c = P.mean(axis=0)
            pool.append({
                "sys_idx": sys_idx,
                "rank": j + 1,
                "poly": P,
                "axis": u,
                "t_min": float(t.min()),
                "t_max": float(t.max()),
                "centroid": c,
                "mask": set(cand.get("mask_seg_ids", [])),
                "obs": float(cand.get("observed_length_px", 0.0)),
            })
    if not pool:
        return []

    # ----- 1) Build a compatibility graph between candidates
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(len(pool)))

    for i in range(len(pool)):
        for j in range(i+1, len(pool)):
            ui, uj = pool[i]["axis"], pool[j]["axis"]
            ang = _angle_between_deg(ui, uj)
            if ang > params.max_angle_diff_deg:
                continue
            # shared axis = normalized(ui + uj) with direction enforced
            if np.dot(ui, uj) < 0:
                uj = -uj
            u = (ui + uj) / (np.linalg.norm(ui + uj) + 1e-12)
            # sequence gap along axis
            ti_min, ti_max = pool[i]["t_min"], pool[i]["t_max"]
            tj_min, tj_max = pool[j]["t_min"], pool[j]["t_max"]
            # recompute projections on shared axis to be safe
            ti = _proj(u, pool[i]["poly"])
            tj = _proj(u, pool[j]["poly"])
            gap = max(0.0, max(tj.min() - ti.max(), ti.min() - tj.max()))
            if gap > params.max_gap_along_axis_px:
                continue
            # lateral offset (centroids perpendicular to shared axis)
            lat = _perp_dist_between_centroids(u, pool[i]["centroid"], pool[j]["centroid"])
            if lat > params.max_lateral_sep_px:
                continue
            # They are compatible → add edge
            G.add_edge(i, j)

    # ----- 2) Connected components → each becomes a super taproot
    super_results: List[SuperTaprootResult] = []
    for comp in nx.connected_components(G):
        comp = sorted(list(comp), key=lambda k: pool[k]["t_min"])  # rough order
        # group axis from all points in the component
        all_pts = np.vstack([pool[k]["poly"] for k in comp])
        u = _pc1(all_pts)
        # order candidates strictly by projection along group axis
        comp_sorted = sorted(
            comp,
            key=lambda k: _proj(u, pool[k]["poly"]).min()
        )

        # stitch polylines in order with straight bridges (if not too long)
        poly = np.zeros((0,2), float)
        obs_sum = 0.0
        fab_sum = 0.0
        for idx, k in enumerate(comp_sorted):
            P = pool[k]["poly"]
            # ensure oriented to increase along u
            tt = _proj(u, P)
            if tt[-1] < tt[0]:
                P = P[::-1]
            if poly.size == 0:
                poly = P.copy()
            else:
                gap_vec = P[0] - poly[-1]
                gap = float(np.linalg.norm(gap_vec))
                if gap <= params.bridge_max_gap_px:
                    poly, g = _concat_with_straight_bridge(poly, P)
                    fab_sum += g
                else:
                    # too big a jump → start a new chain? for now, skip stitching this candidate
                    # (alternatively: start a new super root; to keep simple we skip)
                    continue
            # observed += intrinsic poly length
            obs_sum += float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))

        if poly.shape[0] < 2:
            continue

        # union mask and bookkeeping
        mask = set().union(*[pool[k]["mask"] for k in comp_sorted])
        # depth along axis (projection span)
        t = _proj(u, poly)
        depth = float(t.max() - t.min())
        pct_fab = fab_sum / max(obs_sum + fab_sum, 1e-9)

        if pct_fab <= params.max_pct_fabricated:
            used = [(pool[k]["sys_idx"], pool[k]["rank"]) for k in comp_sorted]
            super_results.append(SuperTaprootResult(
                sample_id=sample_id,
                polyline=poly.tolist(),
                mask_seg_ids=sorted(mask),
                used=used,
                observed_length_px=obs_sum,
                fabricated_length_px=fab_sum,
                pct_fabricated=pct_fab,
                depth_along_axis_px=depth,
                axis_vec=(float(u[0]), float(u[1]))
            ))

    # sort by total length (obs+fab), then depth
    super_results.sort(key=lambda r: (-(r.observed_length_px + r.fabricated_length_px), -r.depth_along_axis_px))
    return super_results
# ==== end super taproot combiner ============================================

# ==== chain-based super taproot combiner ====================================
from typing import Sequence, Union
from dataclasses import dataclass

@dataclass
class SuperTaprootChainParams:
    # Geometric compatibility
    max_angle_diff_deg: float = 12.0      # how parallel candidates must be
    max_lateral_sep_px: float = 60.0      # max sideways distance between centroids
    max_overlap_px: float = 8.0           # allow small overlap along axis (<= this)
    seq_slack_px: float = 12.0            # small tolerance on ordering (tmax_i <= tmin_j + slack)
    # Scoring
    node_obs_weight: float = 1.0          # reward per observed pixel on each node
    node_fab_penalty: float = 0.5         # penalty per fabricated pixel on each node
    edge_gap_penalty: float = 0.02        # penalty per px of gap between chains
    edge_lat_penalty: float = 0.02        # penalty per px lateral offset between chains
    edge_angle_penalty: float = 0.1       # penalty per degree of angle difference
    # Stitching
    bridge_max_gap_px: float = 180.0      # max allowed straight bridge when stitching
    # Output filtering
    max_pct_fabricated: float = 0.60      # drop chain if mostly bridges

@dataclass
class SuperTaprootResult:
    sample_id: str
    polyline: list[list[float]]
    mask_seg_ids: list[str]
    used: list[tuple[int, int]]          # [(system_index, candidate_rank)] included
    observed_length_px: float
    fabricated_length_px: float
    pct_fabricated: float
    depth_along_axis_px: float
    axis_vec: tuple[float, float]

def _pc1(points: np.ndarray) -> np.ndarray:
    X = points - points.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / max(len(X)-1, 1)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]
    return v / (np.linalg.norm(v) + 1e-12)

def _angle_between_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    c = float(np.clip(abs(np.dot(u, v)), -1.0, 1.0))  # directionless parallelism
    return math.degrees(math.acos(c))

def _proj(u: np.ndarray, P: np.ndarray) -> np.ndarray:
    u = u / (np.linalg.norm(u) + 1e-12)
    return P @ u

def _perp_dist(u: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    u = u / (np.linalg.norm(u) + 1e-12)
    d = b - a
    return float(np.linalg.norm(d - np.dot(d, u) * u))

def _concat_with_straight_bridge(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    if A.size == 0:
        return B.copy(), 0.0
    gap = float(np.linalg.norm(B[0] - A[-1]))
    if gap > 1e-9:
        bridge = np.vstack([A[-1], B[0]])
        return np.vstack([A, bridge, B]), gap
    return np.vstack([A, B]), 0.0

def build_super_taproot_chain(
    *,
    json_file: str | Path,
    sample_id: str,
    taproot_results: dict[tuple[str, int], TaprootResult],
    params: SuperTaprootChainParams = SuperTaprootChainParams(),
    per_system_topk: int = 1,      # how many per-system candidates to consider
    return_top_k_chains: int = 1,  # return best K chains (usually 1)
) -> list[SuperTaprootResult]:
    """
    Build one or more 'super taproot' chains for the WHOLE sample by selecting
    a maximum-score sequence of per-system taproots that are parallel, ordered
    along a shared axis, and close in space. This *disfavors side-by-side overlaps*.
    """
    # ---- 0) Gather pool
    pool = []  # elements with geometry + source info
    for (sid, sys_idx), res in taproot_results.items():
        if sid != sample_id or not res.candidates:
            continue
        for j, cand in enumerate(res.candidates[:max(0, per_system_topk)]):
            P = np.asarray(cand.get("polyline", []), float)
            if P.shape[0] < 2:
                continue
            obs = float(cand.get("observed_length_px", 0.0))
            fab = float(cand.get("fabricated_length_px", 0.0))
            pool.append({
                "sys_idx": sys_idx,
                "rank": j + 1,
                "poly": P,
                "obs": obs,
                "fab": fab,
                "mask": set(cand.get("mask_seg_ids", [])),
            })
    if not pool:
        return []

    # ---- 1) Global axis from all points
    all_pts = np.vstack([p["poly"] for p in pool])
    u = _pc1(all_pts)

    # Normalize each candidate to this axis: orientation, interval, centroid, heading
    for p in pool:
        t = _proj(u, p["poly"])
        if t[-1] < t[0]:
            p["poly"] = p["poly"][::-1]
            t = t[::-1]
        p["t_min"] = float(t.min())
        p["t_max"] = float(t.max())
        p["t_mid"] = float((t.min() + t.max()) * 0.5)
        p["centroid"] = p["poly"].mean(axis=0)
        # local axis for angle check
        p["axis"] = _pc1(p["poly"])

    # ---- 2) Build directed edges (i -> j) for valid sequence
    n = len(pool)
    edges_from = [[] for _ in range(n)]
    # sort nodes by t_mid so edges go forward (DAG)
    order = sorted(range(n), key=lambda k: pool[k]["t_mid"])
    index = {k: i for i, k in enumerate(order)}

    for ii in range(n):
        i = order[ii]
        for jj in range(ii+1, n):
            j = order[jj]
            # Angle parallelism
            ang = _angle_between_deg(pool[i]["axis"], pool[j]["axis"])
            if ang > params.max_angle_diff_deg:
                continue
            # Sequence: i should end before j starts (allow tiny overlap)
            overlap = min(pool[i]["t_max"], pool[j]["t_max"]) - max(pool[i]["t_min"], pool[j]["t_min"])
            if overlap > params.max_overlap_px:
                # too much overlap => side-by-side; don't connect
                continue
            # Also allow small negative gap (slack) to be robust
            if pool[i]["t_max"] > pool[j]["t_min"] + params.seq_slack_px:
                continue
            # Lateral offset
            lat = _perp_dist(u, pool[i]["centroid"], pool[j]["centroid"])
            if lat > params.max_lateral_sep_px:
                continue
            # Edge penalty (we subtract it)
            gap = max(0.0, pool[j]["t_min"] - pool[i]["t_max"])
            edge_pen = (params.edge_gap_penalty * gap
                        + params.edge_lat_penalty * lat
                        + params.edge_angle_penalty * ang)
            edges_from[i].append((j, edge_pen, gap, lat, ang))

    # ---- 3) Longest path DP with node rewards − edge penalties
    best_score = [-1e18] * n
    back = [-1] * n
    # node reward
    def node_reward(k: int) -> float:
        return params.node_obs_weight * pool[k]["obs"] - params.node_fab_penalty * pool[k]["fab"]

    for k in order:
        # start a new chain at k
        best_score[k] = max(best_score[k], node_reward(k))
        for (j, edge_pen, gap, lat, ang) in edges_from[k]:
            cand = best_score[k] + node_reward(j) - edge_pen
            if cand > best_score[j]:
                best_score[j] = cand
                back[j] = k

    # ---- 4) Recover top-K chains (greedy peel)
    results: list[SuperTaprootResult] = []
    used_nodes: set[int] = set()
    for _ in range(return_top_k_chains):
        # pick best terminal not already used
        term = max((k for k in range(n) if k not in used_nodes), key=lambda k: best_score[k], default=None)
        if term is None or best_score[term] <= -1e17:
            break
        # backtrack
        chain = []
        cur = term
        while cur != -1 and cur not in used_nodes:
            chain.append(cur)
            cur = back[cur]
        chain = chain[::-1]
        if not chain:
            break
        # mark used
        used_nodes.update(chain)

        # Stitch polyline & compute metrics
        poly = np.zeros((0,2), float)
        obs_sum = 0.0
        fab_sum = 0.0
        masks = []
        for idx, k in enumerate(chain):
            P = pool[k]["poly"]
            # ensure forward along u
            tt = _proj(u, P)
            if tt[-1] < tt[0]:
                P = P[::-1]
            if poly.size == 0:
                poly = P.copy()
            else:
                gap = float(np.linalg.norm(P[0] - poly[-1]))
                if gap <= params.bridge_max_gap_px:
                    poly, g = _concat_with_straight_bridge(poly, P)
                    fab_sum += g
                else:
                    # large gap: stop the chain here
                    break
            obs_sum += float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
            masks.append(pool[k]["mask"])

        if poly.shape[0] < 2:
            continue

        pct_fab = fab_sum / max(obs_sum + fab_sum, 1e-9)
        if pct_fab > params.max_pct_fabricated:
            continue

        t = _proj(u, poly)
        depth = float(t.max() - t.min())
        used = [(pool[k]["sys_idx"], pool[k]["rank"]) for k in chain]

        results.append(SuperTaprootResult(
            sample_id=sample_id,
            polyline=poly.tolist(),
            mask_seg_ids=sorted(set().union(*masks)) if masks else [],
            used=used,
            observed_length_px=obs_sum,
            fabricated_length_px=fab_sum,
            pct_fabricated=pct_fab,
            depth_along_axis_px=depth,
            axis_vec=(float(u[0]), float(u[1])),
        ))

    # Sort by total length then depth
    results.sort(key=lambda r: (-(r.observed_length_px + r.fabricated_length_px), -r.depth_along_axis_px))
    return results
# ==== end chain-based super taproot combiner =================================

# ---- Global (per-image) vertical taproot -----------------------------------
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional, Union
import math
import numpy as np
import networkx as nx
import json


# ---- Global vertical taproot (simple, one per image) -----------------------
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional, Union
import json, math
import numpy as np
import networkx as nx

# Reuse your helpers already in taproot.py:
# - TaprootParams, TaprootResult
# - seg_len, seg_w
# - classify_links, build_end_to_end_map, connected_components_end_to_end

##-- Defunct - remove and remove all instances of use.
@dataclass
class ImageVerticalParams(TaprootParams):
    w_vertical: float = 1.0
    vertical_decay_px: float = 220.0
    heading_smooth_tol_deg: float = 10.0
    w_heading_smooth: float = 0.5
    bridge_max_lateral_px: float = 20.0

    max_overlap_px: float = 8.0
    seq_slack_px: float = 10.0
    max_gap_along_axis_px: float = 180.0
    max_lateral_sep_px: float = 50.0
    max_heading_angle_diff_deg: float = 15.0

@dataclass
class GlobalTaprootParams(TaprootParams):
    """
    Simple per-image vertical taproot parameters.
    Keeps TaprootParams base fields and adds a few focused knobs.
    """
    # Build groups (end_to_end + small-angle bent)
    bent_thresh_deg: float = 25.0

    # Seed emphasis (favor thickness)
    start_width_quantile: float = 0.80   # top 20% width groups as seeds
    start_k_groups: int = 8              # try up to K starts

    # Bridging across groups (inline alignment)
    bridge_max_gap_px: float = 45.0      # max euclid gap to bridge
    bridge_axis_angle_max_deg: float = 30.0  # |angle(bridge, vertical)| <= this
    bridge_max_lateral_px: float = 20.0      # lateral (perp-to-vertical) cap for bridge
    colinear_tol_deg: float = 20.0       # |angle(bridge, heading_i/j)| <= this
    heading_diff_max_deg: float = 25.0   # |heading_i - heading_j| <= this

    # Sequence along vertical axis (down = decreasing y; use t = -y increasing)
    max_overlap_px: float = 8.0          # allow tiny overlap
    seq_slack_px: float = 12.0           # tolerance for ordering

    # Scoring weights
    alpha_width: float = 1.0             # node reward uses (width^alpha)*length
    w_W: float = 1.0                     # node reward weight
    w_D: float = 0.25                    # vertical depth reward
    w_vertical: float = 0.8              # per-node penalty for |angle to vertical|
    w_smooth: float = 0.4                # per-edge penalty for heading change > tol
    heading_smooth_tol_deg: float = 12.0 # allowed change between consecutive groups
    bridge_cost_per_px: float = 0.02     # per-pixel penalty for bridge euclid gap
    bridge_lateral_cost_per_px: float = 0.02  # extra per-pixel lateral penalty

    # Fabrication filter
    max_pct_fabricated: float = 0.40


def find_vertical_taproot_per_image_v2(
    json_file: str | Path,
    sample_id: Union[str, List[str]],          # str or list[str]
    params: GlobalTaprootParams = GlobalTaprootParams(),
) -> Dict[Tuple[str, int], TaprootResult]:
    """
    Find ONE vertical taproot per image by chaining thick, near-vertical groups
    across the whole sample. Groups are built via end_to_end (+ small bent angle).
    We then connect groups either by natural adjacency (branch links) or by small
    inline bridges that are near-vertical, nearly colinear, and have small lateral drift.

    Returns: dict[(sample_id, -1)] -> TaprootResult with exactly one candidate.
    """
    # normalize sample ids
    if isinstance(sample_id, str):
        sids = [sample_id]
    elif isinstance(sample_id, list) and all(isinstance(s, str) for s in sample_id):
        sids = sample_id
    else:
        raise TypeError("sample_id must be str or list[str].")

    v = np.array([0.0, -1.0], float)  # vertical down = decreasing y
    def _proj_v(P: np.ndarray) -> np.ndarray:
        return P @ v

    def _angle_deg(u: np.ndarray, w: np.ndarray) -> float:
        nu, nw = np.linalg.norm(u), np.linalg.norm(w)
        if nu == 0 or nw == 0:
            return 0.0
        c = float(np.clip(np.dot(u, w) / (nu * nw), -1.0, 1.0))
        return math.degrees(math.acos(c))

    def _segment_centroid(points) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        return pts.mean(axis=0) if len(pts) else np.array([np.nan, np.nan])

    results: Dict[Tuple[str, int], TaprootResult] = {}

    data = json.loads(Path(json_file).read_text(encoding="utf-8"))

    for img in data.get("images", []):
        sid = img["info"].get("experiment")
        if sid not in sids:
            continue

        # Flatten everything (whole image)
        seg_map: Dict[str, Dict[str, Any]] = {
            str(seg["name"]): seg for root in img["roots"] for seg in root["segments"]
        }
        links: List[Dict[str, Any]] = [lk for root in img["roots"] for lk in root.get("links", [])]
        all_seg_ids = list(seg_map.keys())

        if not all_seg_ids:
            results[(sid, -1)] = TaprootResult(sid, -1, -1, [])
            continue

        # Classify links for the whole image (with small bent angle)
        e2e_links, branch_links = classify_links(
            links, set(all_seg_ids), seg_map, bent_thresh_deg=params.bent_thresh_deg
        )
        e2e_map = build_end_to_end_map(e2e_links)

        # Groups = connected comps under e2e (+bent)
        groups = connected_components_end_to_end(set(all_seg_ids), e2e_map)
        if not groups:
            results[(sid, -1)] = TaprootResult(sid, -1, -1, [])
            continue

        # Per-segment quantities
        widths_all = np.array([max(seg_w(seg_map[s]["meta"]), 0.0) for s in all_seg_ids], float)
        lengths_all = np.array([seg_len(seg_map[s]["points"]) for s in all_seg_ids], float)

        # Group stats
        group_stats = []
        for gid, gset in enumerate(groups):
            g_list = list(gset)
            g_widths  = np.array([max(seg_w(seg_map[s]["meta"]), 0.0) for s in g_list], float)
            g_lengths = np.array([seg_len(seg_map[s]["points"]) for s in g_list], float)
            g_weights = g_widths * np.maximum(g_lengths, 1e-6)
            g_centers = np.array([_segment_centroid(seg_map[s]["points"]) for s in g_list], float)

            # representative segment = argmax width*length
            rep_idx = int(np.argmax(g_weights)) if len(g_weights) else 0
            rep_seg = g_list[rep_idx]
            rep_pts = np.asarray(seg_map[rep_seg]["points"], float)

            # orient rep polyline to go DOWN (increase t = -y)
            if rep_pts.shape[0] >= 2:
                t = _proj_v(rep_pts)
                if t[-1] < t[0]:
                    rep_pts = rep_pts[::-1]

            # heading ~ end-to-end vector of rep polyline
            heading = np.zeros(2, float)
            if rep_pts.shape[0] >= 2:
                hv = rep_pts[-1] - rep_pts[0]
                n = np.linalg.norm(hv)
                if n > 0:
                    heading = hv / n

            t_vals = _proj_v(g_centers) if len(g_centers) else np.array([0.0])
            t_min = float(np.min(t_vals)) if len(t_vals) else 0.0
            t_max = float(np.max(t_vals)) if len(t_vals) else 0.0

            tot_len = float(np.sum(g_lengths))
            w_avg   = float(np.sum(g_widths * g_lengths) / max(tot_len, 1e-9)) if tot_len > 0 else 0.0

            group_stats.append({
                "gid": gid,
                "seg_ids": g_list,
                "rep_seg": rep_seg,
                "rep_pts": rep_pts,
                "width": w_avg,
                "length": tot_len,
                "heading": heading,
                "t_min": t_min,
                "t_max": t_max,
                "centroid": g_centers.mean(axis=0) if len(g_centers) else np.array([np.nan, np.nan]),
                "angle_to_vertical": _angle_deg(heading, v),
            })

        n = len(group_stats)
        if n == 0:
            results[(sid, -1)] = TaprootResult(sid, -1, -1, [])
            continue

        # Build natural adjacency via branch_links (across groups)
        seg_to_gid = {s: gid for gid, gs in enumerate(groups) for s in gs}
        natural_adj = set()
        for (a, b) in branch_links:
            ga, gb = seg_to_gid[a], seg_to_gid[b]
            if ga != gb:
                u, w = (ga, gb) if ga < gb else (gb, ga)
                natural_adj.add((u, w))

        # Order groups by vertical coordinate (t_mid)
        t_mid = [0.5 * (g["t_min"] + g["t_max"]) for g in group_stats]
        order = sorted(range(n), key=lambda i: t_mid[i])

        # Helper: best inline bridge endpoints & metrics consistent with downward direction
        def bridge_metrics(i: int, j: int):
            Pi = np.asarray(group_stats[i]["rep_pts"], float)
            Pj = np.asarray(group_stats[j]["rep_pts"], float)
            if Pi.shape[0] == 0 or Pj.shape[0] == 0:
                return None
            ends_i = [Pi[0], Pi[-1]]
            ends_j = [Pj[0], Pj[-1]]
            best = None
            for a in ends_i:
                for b in ends_j:
                    # forward only: j should be deeper (t increases)
                    if _proj_v(b) + params.seq_slack_px < _proj_v(a):
                        continue
                    vec = b - a
                    gdist = float(np.linalg.norm(vec))
                    if gdist == 0:
                        ang_vert = 0.0
                        lat = 0.0
                    else:
                        ang_vert = _angle_deg(vec, v)
                        lat_vec = vec - (np.dot(vec, v) * v)  # lateral
                        lat = float(np.linalg.norm(lat_vec))
                    # Heading compatibility
                    ang_i = _angle_deg(group_stats[i]["heading"], vec) if gdist > 0 else 0.0
                    ang_j = _angle_deg(vec, group_stats[j]["heading"]) if gdist > 0 else 0.0
                    # store the best (smallest euclid gap) that passes angular caps later
                    if (best is None) or (gdist < best["euclid_gap"]):
                        best = {"a": a, "b": b, "vec": vec, "euclid_gap": gdist,
                                "lat": lat, "ang_vert": ang_vert, "ang_i": ang_i, "ang_j": ang_j}
            return best

        # Build directed edges i -> j
        edges_from = [[] for _ in range(n)]
        for ia, i in enumerate(order):
            gi = group_stats[i]
            for jb in range(ia + 1, n):
                j = order[jb]
                gj = group_stats[j]

                # vertical sequence checks
                overlap = min(gi["t_max"], gj["t_max"]) - max(gi["t_min"], gj["t_min"])
                if overlap > params.max_overlap_px:
                    continue
                if gi["t_max"] > gj["t_min"] + params.seq_slack_px:
                    continue

                # try to connect naturally or via inline bridge
                natural = (min(i, j), max(i, j)) in natural_adj
                bm = bridge_metrics(i, j)

                if natural:
                    # Natural adjacency allowed; compute edge penalty from heading change
                    head_diff = _angle_deg(gi["heading"], gj["heading"])
                    edges_from[i].append({
                        "j": j, "is_bridge": False,
                        "euclid_gap": 0.0, "lat": 0.0,
                        "head_diff": head_diff
                    })
                elif bm is not None:
                    # Bridge must satisfy angle & distance caps
                    if (bm["euclid_gap"] <= params.bridge_max_gap_px and
                        bm["lat"] <= params.bridge_max_lateral_px and
                        bm["ang_vert"] <= params.bridge_axis_angle_max_deg and
                        bm["ang_i"] <= params.colinear_tol_deg and
                        bm["ang_j"] <= params.colinear_tol_deg and
                        _angle_deg(gi["heading"], gj["heading"]) <= params.heading_diff_max_deg):
                        edges_from[i].append({
                            "j": j, "is_bridge": True,
                            "euclid_gap": float(bm["euclid_gap"]),
                            "lat": float(bm["lat"]),
                            "head_diff": float(_angle_deg(gi["heading"], gj["heading"])),
                        })

        # Node & edge scores
        node_score = np.zeros(n, float)
        for k in range(n):
            g = group_stats[k]
            reward = (g["width"] ** params.alpha_width) * g["length"]
            pen_vertical = params.w_vertical * g["angle_to_vertical"]
            node_score[k] = params.w_W * reward - pen_vertical

        def edge_penalty(e) -> float:
            pen = 0.0
            if e["is_bridge"]:
                pen += params.bridge_cost_per_px * e["euclid_gap"]
                pen += params.bridge_lateral_cost_per_px * e["lat"]
            # smoothness penalty
            excess = max(0.0, e["head_diff"] - params.heading_smooth_tol_deg)
            pen += params.w_smooth * excess
            return pen

        # DP for best chain
        best = node_score.copy()
        back = np.full(n, -1, int)

        for i in order:
            for e in edges_from[i]:
                j = e["j"]
                cand = best[i] + node_score[j] - edge_penalty(e)
                if cand > best[j]:
                    best[j] = cand
                    back[j] = i

        # pick terminal
        term = int(np.argmax(best))
        chain = []
        cur = term
        while cur != -1:
            chain.append(cur)
            cur = back[cur]
        chain.reverse()

        if not chain:
            results[(sid, -1)] = TaprootResult(sid, -1, -1, [])
            continue

        # Build polyline + stats
        obs_len, fab_len = 0.0, 0.0
        pts_list = []
        for idx, k in enumerate(chain):
            gp = np.asarray(group_stats[k]["rep_pts"], float)
            if gp.shape[0] >= 2:
                if pts_list:
                    a = pts_list[-1][-1]
                    b = gp[0]
                    vec = b - a
                    gap = float(np.linalg.norm(vec))
                    if gap > 1e-6:
                        # Allow a straight stitch only if within bridge caps
                        lat_vec = vec - (np.dot(vec, v) * v)
                        lat = float(np.linalg.norm(lat_vec))
                        ang_vert = _angle_deg(vec, v)
                        if (gap <= params.bridge_max_gap_px and
                            lat <= params.bridge_max_lateral_px and
                            ang_vert <= params.bridge_axis_angle_max_deg):
                            pts_list.append(np.vstack([a, b]))
                            fab_len += gap
                        else:
                            # stop stitching if the next group is too far / off-axis
                            break
                pts_list.append(gp)
                obs_len += float(np.sum(np.linalg.norm(np.diff(gp, axis=0), axis=1)))

        poly = np.vstack(pts_list) if pts_list else np.zeros((0, 2), float)
        if poly.shape[0] < 2:
            results[(sid, -1)] = TaprootResult(sid, -1, -1, [])
            continue

        depth_vert = float(np.ptp(_proj_v(poly)))
        pct_fab = fab_len / max(obs_len + fab_len, 1e-9)

        # Depth reward
        score = float(best[term]) + params.w_D * depth_vert

        candidate = {
            "path_groups": chain,
            "score": score,
            "score_parts": {
                "W_node_sum": float(np.sum([(group_stats[k]["width"] ** params.alpha_width) * group_stats[k]["length"] for k in chain])),
                "vertical_pen_sum": float(np.sum([params.w_vertical * group_stats[k]["angle_to_vertical"] for k in chain])),
                "depth_vert": depth_vert,
                "fabricated_px": fab_len,
            },
            "mask_seg_ids": list({s for k in chain for s in groups[k]}),
            "polyline": poly.tolist(),
            "observed_length_px": obs_len,
            "fabricated_length_px": fab_len,
            "pct_fabricated": pct_fab,
            "depth_along_axis_px": depth_vert,
            "start_gid": chain[0],
            "end_gid": chain[-1],
        }

        # optionally filter by fabrication; we keep one candidate regardless (fallback)
        if pct_fab > params.max_pct_fabricated:
            # keep but mark low score (so you can still visualize/debug)
            candidate["score"] -= 1e6

        results[(sid, -1)] = TaprootResult(sample_id=sid, system_index=-1, best_idx=0, candidates=[candidate])

    return results

# ---- Visualization for vertical taproot (1 taproot per image) --------------

def plot_global_taproot(
    json_file: str | Path,
    sample_id: Union[str, List[str]],
    global_results: Dict[Tuple[str, int], TaprootResult],
    *,
    taproot_alpha: float = 0.9,
    linewidth_bg: float = 0.6,
    linewidth_poly: float = 4.0,
    n_seg_min: int = 0,
    show_legend: bool = False,
    save_dir: Optional[str] = None,
    invert_y: Optional[bool] = None,
) -> None:
    """One figure per sample; background = full sample; overlay = single (sid,-1) candidate."""
    import os
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    data = json.loads(Path(json_file).read_text(encoding="utf-8"))
    inv = _VIZ.invert_y_default if invert_y is None else invert_y
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    sids = [sample_id] if isinstance(sample_id, str) else list(sample_id)
    for sid in sids:
        try:
            img = next(img for img in data["images"] if img["info"]["experiment"] == sid)
        except StopIteration:
            continue

        segment_map = {str(seg["name"]): seg for root in img["roots"] for seg in root["segments"]}

        # Background (whole sample)
        bg_ids = list(segment_map.keys())
        segs, widths = _collect_segment_drawables(segment_map, bg_ids, invert_y=inv)

        fig, ax = plt.subplots(figsize=figsize or _VIZ.figsize, dpi=dpi or _VIZ.figure_dpi)

        if segs:
            lc_bg = LineCollection(segs, colors=[_VIZ.base_gray]*len(segs),
                                   linewidths=widths, alpha=_VIZ.base_alpha, zorder=1)
            ax.add_collection(lc_bg)

        # Candidate
        key = (sid, -1)
        if key in global_results and global_results[key].candidates:
            cand = global_results[key].candidates[0]
            P = np.asarray(cand.get("polyline", []), dtype=float)
            if P.ndim == 2 and P.shape[0] >= 2:
                if inv:
                    P = P.copy(); P[:, 1] = -P[:, 1]
                ax.plot(P[:, 0], P[:, 1], color=_VIZ.emph_color, linewidth=linewidth_poly,
                        alpha=taproot_alpha, zorder=3)

        _auto_limits(ax, segs if segs else [])
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(sid)

        if save_dir:
            out = os.path.join(save_dir, f"{sid}__global_vertical_taproot.png")
            plt.savefig(out_path, dpi=_VIZ.save_dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


# ---- Side-by-side comparison helper ----------------------------------------
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional, Union
import os

def compare_taproot_versions(
    json_file: str | Path,
    sample_id: Union[str, List[str]],
    *,
    baseline_params: TaprootParams = TaprootParams(),
    vertical_params: ImageVerticalParams = ImageVerticalParams(),      # ← use ImageVerticalParams
    system_index: Optional[int] = None,
    topk: int = 3,
    n_seg_min: int = 0,
    taproot_alpha: float = 0.8,
    show_legend: bool = True,
    invert_y: Optional[bool] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, Dict[Tuple[str, int], TaprootResult]]:
    """
    Run both the baseline and vertical-prior taproot finders and visualize them.

    - sample_id: str or list[str]; runs per-sample.
    - If save_dir is given, creates it (if needed) and writes:
        <sid>__baseline__taproot.png
        <sid>__vertical__taproot.png
      Otherwise figures are shown.
    - Returns a dict with keys {"baseline", "vertical"} mapping to their result dicts.
    """
    # strict normalization to list[str]
    if isinstance(sample_id, str):
        sids = [sample_id]
    elif isinstance(sample_id, list) and all(isinstance(s, str) for s in sample_id):
        sids = sample_id
    else:
        raise TypeError("sample_id must be str or list[str].")

    # Run both finders once (they handle multiple samples internally)
    baseline_res = find_taproot_from_json(
        json_file=json_file,
        sample_id=sids,
        system_id=None,
        params=baseline_params,
    )
    vertical_res = find_taproot_from_json_vertical(
        json_file=json_file,
        sample_id=sids,
        system_id=None,
        params=vertical_params,
    )

    # Ensure save_dir exists if provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Visualize per-sample, per-variant
    for sid in sids:
        # Baseline
        plot_taproot_candidates(
            json_file=json_file,
            sample_id=sid,
            taproot_results=baseline_res,
            system_index=system_index,
            topk=topk,
            n_seg_min=n_seg_min,
            show_legend=show_legend,
            title=f"{sid} — baseline",
            invert_y=invert_y,
            taproot_alpha=taproot_alpha,
            save_path=(None if not save_dir else os.path.join(save_dir, f"{sid}__baseline__taproot.png")),
        )

        # Vertical-prior
        plot_taproot_candidates(
            json_file=json_file,
            sample_id=sid,
            taproot_results=vertical_res,
            system_index=system_index,
            topk=topk,
            n_seg_min=n_seg_min,
            show_legend=show_legend,
            title=f"{sid} — vertical prior",
            invert_y=invert_y,
            taproot_alpha=taproot_alpha,
            save_path=(None if not save_dir else os.path.join(save_dir, f"{sid}__vertical__taproot.png")),
        )

    return {"baseline": baseline_res, "vertical": vertical_res}
# ---- end comparison helper --------------------------------------------------


# --- Minimal GT tools: annotate, map-to-segments, evaluate -------------------
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json, os, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Uses seg_w/seg_len already in your taproot.py imports
# from .geometry import segment_length as seg_len, segment_width as seg_w

@dataclass
class EvalConfig:
    eps_px: float = 10.0     # corridor radius for mapping clicks → segments
    seg_hit_frac: float = 0.6  # fraction of a segment's vertices within eps to count as GT
    crown_frac: float = 0.2    # top % of length considered “crown” for alignment/smoothness

# ---------- drawing helpers (no dependency on viz.py) ----------
def _collect_segment_drawables(segment_map: Dict[str, Any], seg_ids: List[str], invert_y: bool = True):
    segs, lws = [], []
    for sid in seg_ids:
        seg = segment_map.get(str(sid))
        if not seg:
            continue
        P = np.asarray(seg["points"], float)
        if P.shape[0] < 2:
            continue
        if invert_y:  # your images use y-down, flip for plotting
            P = P.copy(); P[:, 1] = -P[:, 1]
        segs.extend(np.stack([P[:-1], P[1:]], axis=1))
        w = max(seg_w(seg["meta"]), 0.5)
        lws.extend([w] * (P.shape[0]-1))
    return segs, lws

def _load_sample(json_file: str | os.PathLike, sample_id: str):
    data = json.loads(Path(json_file).read_text(encoding="utf-8"))
    img = next(img for img in data["images"] if img["info"]["experiment"] == sample_id)
    segmap = {str(seg["name"]): seg for root in img["roots"] for seg in root["segments"]}
    return img, segmap

# ---------- geometry: distance point ↔ polyline ----------
def _dist_point_polyline(pt: np.ndarray, poly: np.ndarray) -> float:
    """Min distance from point to a polyline (piecewise linear)."""
    if poly.shape[0] < 2:
        return np.inf
    dmin = np.inf
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        t = 0.0 if (ab == 0).all() else float(np.clip(np.dot(pt - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1))
        proj = a + t * ab
        d = float(np.linalg.norm(pt - proj))
        if d < dmin: dmin = d
    return dmin

def _poly_length(P: np.ndarray) -> float:
    return 0.0 if P.shape[0] < 2 else float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))

# ---------- map clicked GT polyline → GT segment IDs ----------
def segments_near_polyline(segment_map: Dict[str, Any],
                           polyline: np.ndarray,
                           eps_px: float = 10.0,
                           seg_hit_frac: float = 0.6) -> List[str]:
    """Return seg_ids whose vertices lie near the polyline by at least seg_hit_frac."""
    out = []
    for sid, seg in segment_map.items():
        P = np.asarray(seg["points"], float)
        if P.shape[0] == 0:
            continue
        dists = np.array([_dist_point_polyline(p, polyline) for p in P], float)
        hit_frac = float(np.mean(dists <= eps_px))
        if hit_frac >= seg_hit_frac:
            out.append(str(sid))
    return out

# ---------- interactive GT collection ----------
def annotate_taproot_polyline_tk(
    json_file: str | os.PathLike,
    sample_id: str,
    save_to: str = "taproot_gt.json",
    *,
    invert_y: bool = True,       # flip Y only for display; unflip when saving
    eps_px: float = 10.0,        # corridor radius for mapping clicks → segments
    seg_hit_frac: float = 0.6,   # fraction of segment vertices within eps to count it
    show_clicks: bool = True,    # show click markers while collecting
) -> dict:
    """
    Tk-only annotator: draws the whole sample with true segment widths, lets you click
    crown→tip, and saves a sidecar JSON with {'sample_id','polyline','seg_ids'}.

    - Requires a GUI backend with Tk: run `%matplotlib tk` before calling.
    - Left-click to add points; press Enter (or middle-click) to finish.
    """
    import json
    from pathlib import Path
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    # --- guard: ensure we're on Tk so ginput blocks reliably
    backend = matplotlib.get_backend().lower()
    if "tk" not in backend:
        raise RuntimeError("This function requires the Tk backend. Run `%matplotlib tk` first.")

    # --- load sample & build background drawables (uses your helpers)
    img, segmap = _load_sample(json_file, sample_id)
    seg_ids_all = list(segmap.keys())
    if not seg_ids_all:
        raise ValueError(f"No segments found for sample '{sample_id}'.")

    segs, lws = _collect_segment_drawables(segmap, seg_ids_all, invert_y=invert_y)
    if not segs:
        raise ValueError(f"Could not build drawables for '{sample_id}' (empty polylines).")

    # --- single, named figure; draw background
    fig = plt.figure(num=f"taproot-annotator-{sample_id}", figsize=(10, 10), dpi=150)
    fig.clf()
    ax = fig.add_subplot(111)
    lc = LineCollection(segs, colors="#BDBDBD", linewidths=lws, alpha=0.7, zorder=1)
    ax.add_collection(lc)

    arr = np.array(segs).reshape(-1, 2)
    ax.set_xlim(arr[:, 0].min() - 5, arr[:, 0].max() + 5)
    ax.set_ylim(arr[:, 1].min() - 5, arr[:, 1].max() + 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{sample_id} — click crown→tip • Enter/middle-click to finish")

    # Ensure the window is visible before collecting clicks (non-blocking show)
    fig.canvas.draw()
    try:
        fig.show()
    except Exception:
        pass

    # --- blocking click collection on THIS figure (Tk handles this well)
    clicks = plt.ginput(n=-1, timeout=0, show_clicks=show_clicks)  # blocks until Enter/middle-click
    plt.close(fig)

    if len(clicks) < 2:
        raise ValueError("No polyline annotated (need at least 2 clicks).")

    # --- convert back to data coords (undo y-flip) + map to segments
    P_plot = np.asarray(clicks, float)
    P_data = P_plot.copy()
    if invert_y:
        P_data[:, 1] = -P_data[:, 1]

    gt_seg_ids = segments_near_polyline(segmap, P_data, eps_px=eps_px, seg_hit_frac=seg_hit_frac)

    # --- write/update sidecar JSON
    sidecar_path = Path(save_to)
    sidecar = {"entries": []}
    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if "entries" not in sidecar:
                sidecar = {"entries": []}
        except Exception:
            sidecar = {"entries": []}

    sidecar["entries"] = [e for e in sidecar["entries"] if e.get("sample_id") != sample_id]
    sidecar["entries"].append({
        "sample_id": sample_id,
        "polyline": P_data.tolist(),
        "seg_ids": gt_seg_ids,
        "meta": {"eps_px": eps_px, "seg_hit_frac": seg_hit_frac},
    })
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    return {"sample_id": sample_id, "polyline": P_data, "seg_ids": gt_seg_ids}

# ---------- load GT & evaluate a prediction ----------
def load_gt_sidecar(sidecar_path: str | os.PathLike) -> Dict[str, Dict[str, Any]]:
    """Return dict[sample_id] -> {"polyline": np.ndarray, "seg_ids": [..]}."""
    D = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    out = {}
    for e in D.get("entries", []):
        sid = e.get("sample_id")
        if not sid: continue
        pl = np.asarray(e.get("polyline", []), float)
        segs = [str(s) for s in e.get("seg_ids", [])]
        out[sid] = {"polyline": pl, "seg_ids": segs}
    return out

def eval_prediction_against_gt(json_file: str | os.PathLike,
                               sample_id: str,
                               pred_candidate: Dict[str, Any],
                               gt: Dict[str, Any],
                               cfg: EvalConfig = EvalConfig()) -> Dict[str, float]:
    """
    Compare your predicted candidate (one dict) vs GT for `sample_id`.
    Returns a dict of simple, objective metrics.
    """
    # Curve metrics: symmetric avg distance (P↔G) with corridor
    P = np.asarray(pred_candidate.get("polyline", []), float)
    G = np.asarray(gt["polyline"], float)
    def _avg_min_dist(A, B):
        if len(A) == 0 or len(B) == 0: return np.inf
        d = np.array([_dist_point_polyline(a, B) for a in A], float)
        return float(d.mean())
    # densify polyline for fair curve distance
    def _densify(Q, step=5.0):
        if len(Q) < 2: return Q
        segs = np.diff(Q, axis=0)
        lens = np.linalg.norm(segs, axis=1)
        nsteps = np.maximum(1, (lens/step).astype(int))
        out = [Q[0]]
        for i, n in enumerate(nsteps):
            for t in range(1, int(n)+1):
                out.append(Q[i] + (t/(n+1))*segs[i])
            out.append(Q[i+1])
        return np.vstack(out)
    Pd = _densify(P); Gd = _densify(G)
    avg_P_to_G = _avg_min_dist(Pd, G)
    avg_G_to_P = _avg_min_dist(Gd, P)

    # Segment set metrics (IoU / Precision / Recall)
    img, segmap = _load_sample(json_file, sample_id)
    pred_set = set(str(s) for s in pred_candidate.get("mask_seg_ids", []))
    gt_set = set(gt.get("seg_ids", []))
    inter = len(pred_set & gt_set); uni = len(pred_set | gt_set)
    iou = inter / uni if uni > 0 else 0.0
    prec = inter / len(pred_set) if pred_set else 0.0
    rec = inter / len(gt_set) if gt_set else 0.0

    # Fabrication & depth (already provided by your candidate)
    pct_fab = float(pred_candidate.get("pct_fabricated", 0.0))
    depth_px = float(pred_candidate.get("depth_along_axis_px", 0.0))

    # Simple vertical alignment near crown: mean |angle to vertical| on top 20% of length
    v = np.array([0.0, -1.0], float)
    def _angle(u, w):
        nu, nw = np.linalg.norm(u), np.linalg.norm(w)
        if nu == 0 or nw == 0: return 0.0
        c = float(np.clip(np.dot(u,w)/(nu*nw), -1.0, 1.0))
        return math.degrees(math.acos(c))
    # split P into crown/tail by arc length
    if len(P) >= 2:
        L = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        total = L[-1] if L.size else 0.0
        cutoff = cfg.crown_frac * total
        crown_idx = np.searchsorted(L, cutoff)
        crown = P[:max(2, crown_idx+1)]
        # heading samples along crown
        angs = []
        for a,b in zip(crown[:-1], crown[1:]):
            hv = b - a
            if np.linalg.norm(hv) > 0:
                angs.append(_angle(hv, v))
        vert_angle_crown = float(np.mean(angs)) if angs else 0.0
    else:
        vert_angle_crown = 90.0

    return {
        "avg_dist_P_to_G": avg_P_to_G,
        "avg_dist_G_to_P": avg_G_to_P,
        "iou_segments": iou,
        "precision_segments": prec,
        "recall_segments": rec,
        "pct_fabricated": pct_fab,
        "depth_px": depth_px,
        "vertical_angle_crown_deg": vert_angle_crown,
    }
# ---------------------------------------------------------------------------

# --- put this in taproot.py (or wherever your annotator lives) ---

def annotate_taproot_polyline_tk_viz(
    json_file: str | os.PathLike,
    sample_id: str,
    save_to: str = "taproot_gt.json",
    *,
    invert_y: bool | None = None,     # None → use viz default; else override
    eps_px: float = 10.0,
    seg_hit_frac: float = 0.6,
    show_clicks: bool = True,
    min_clicks: int = 2,
    viz_overrides: dict | None = None,   # optional: e.g., {"base_alpha": 0.75}
    save_png_dir: str | None = None,
    fig_max_px: int = 900,               # longest canvas side in pixels
    dpi: int = 120,                      # pixels-per-inch for the figure
) -> dict | None:
    """
    Tk-only annotator with viz styling + pixel-cap sizing.

    - Longest canvas side = fig_max_px (default 900 px), preserving data aspect ratio.
    - DPI controls physical window size: inches = pixels / dpi (default 120).
    - Line widths come solely from viz (_VIZ.width_scale). No per-call scaling.

    Interaction: left-click add; Backspace undo; Enter/middle-click finish; Esc abort.
    """
    import json, os
    from pathlib import Path
    import numpy as np
    import matplotlib, matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    # viz helpers
    from rootlab.viz import (
        _VIZ,
        _collect_segment_drawables,
        _auto_limits,
        set_viz_defaults,
    )

    # --- backend guard (Tk only) ---
    if "tk" not in matplotlib.get_backend().lower():
        raise RuntimeError("This function requires the Tk backend. Run `%matplotlib tk` first.")

    # --- optional, temporary viz overrides (no width scaling here) ---
    _old_vals = {}
    if viz_overrides:
        # remember only the keys we change so we can restore cleanly
        for k, v in viz_overrides.items():
            if hasattr(_VIZ, k):
                _old_vals[k] = getattr(_VIZ, k)
        set_viz_defaults(**{k: v for k, v in viz_overrides.items() if hasattr(_VIZ, k)})

    try:
        # --- load sample & segment map (your helper) ---
        img, segmap = _load_sample(json_file, sample_id)

        inv = _VIZ.invert_y_default if invert_y is None else bool(invert_y)
        all_seg_ids = list(segmap.keys())
        seg_arrays, widths = _collect_segment_drawables(segmap, all_seg_ids, invert_y=inv)
        if not seg_arrays:
            raise ValueError(f"No drawable segments for sample '{sample_id}'.")

        # --- compute data aspect ratio to size canvas by pixel cap (robust bounds) ---
        xs, ys = [], []
        for s in seg_arrays:
            a = np.asarray(s, float)
            if a.ndim == 2 and a.shape == (2, 2):        # one segment
                xs.extend([a[0, 0], a[1, 0]])
                ys.extend([a[0, 1], a[1, 1]])
            elif a.ndim == 3 and a.shape[-2:] == (2, 2): # block of segments
                xs.extend(a[..., 0].ravel().tolist())
                ys.extend(a[..., 1].ravel().tolist())
            # else: ignore malformed

        if not xs:
            raise ValueError(f"No drawable segments for sample '{sample_id}'.")

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        w_data = max(xmax - xmin, 1e-6)
        h_data = max(ymax - ymin, 1e-6)
        aspect = w_data / h_data

        # pixels with cap on longest side, preserving aspect
        if aspect >= 1.0:
            width_px = int(fig_max_px)
            height_px = max(1, int(round(fig_max_px / aspect)))
        else:
            height_px = int(fig_max_px)
            width_px  = max(1, int(round(fig_max_px * aspect)))

        # inches for Matplotlib/Tk window
        fig_w_in = width_px / float(dpi)
        fig_h_in = height_px / float(dpi)

        # --- figure (ONE), draw background in viz style ---
        fig = plt.figure(num=f"taproot-annotator-{sample_id}", figsize=(fig_w_in, fig_h_in), dpi=dpi)
        fig.clf()
        ax = fig.add_subplot(111)

        lc = LineCollection(
            seg_arrays,
            colors=[_VIZ.base_gray] * len(seg_arrays),
            linewidths=widths,              # from viz; no local scaling
            alpha=_VIZ.base_alpha,
            zorder=1,
        )
        ax.add_collection(lc)
        _auto_limits(ax, seg_arrays)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{sample_id} — Left-click add • Backspace undo • Enter finish • Esc abort")

        # --- interactive overlay (viz highlight color) ---
        pts: list[list[float]] = []
        (line,) = ax.plot([], [], "-o", lw=3, ms=4, color=_VIZ.emph_color, zorder=3)

        fig.canvas.draw()
        try:
            fig.show()
        except Exception:
            pass

        aborted = False

        def on_click(e):
            if e.inaxes is not ax or e.button != 1 or e.xdata is None or e.ydata is None:
                return
            pts.append([e.xdata, e.ydata])
            P = np.asarray(pts, float)
            line.set_data(P[:, 0], P[:, 1])
            fig.canvas.draw_idle()

        def on_key(e):
            nonlocal aborted
            if e.key == "backspace":
                if pts:
                    pts.pop()
                    if pts:
                        P = np.asarray(pts, float)
                        line.set_data(P[:, 0], P[:, 1])
                    else:
                        line.set_data([], [])
                    fig.canvas.draw_idle()
            elif e.key in ("escape",):
                aborted = True
                plt.close(fig)
            elif e.key in ("enter", "return"):
                plt.close(fig)

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)

        # --- blocking click collection (Tk ginput is reliable) ---
        clicks = plt.ginput(n=-1, timeout=0, show_clicks=show_clicks)
        try:
            plt.close(fig)
        except Exception:
            pass

        if aborted:
            return None

        if len(clicks) >= min_clicks:
            pts = clicks
        if len(pts) < min_clicks:
            raise ValueError(f"No polyline annotated (need at least {min_clicks} clicks).")

        # --- to data coords (undo y-flip) ---
        P_plot = np.asarray(pts, float)
        P_data = P_plot.copy()
        if inv:
            P_data[:, 1] = -P_data[:, 1]

        # --- map to segments + save sidecar ---
        gt_seg_ids = segments_near_polyline(segmap, P_data, eps_px=eps_px, seg_hit_frac=seg_hit_frac)

        sidecar_path = Path(save_to)
        sidecar = {"entries": []}
        if sidecar_path.exists():
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if "entries" not in sidecar:
                    sidecar = {"entries": []}
            except Exception:
                sidecar = {"entries": []}
        sidecar["entries"] = [e for e in sidecar["entries"] if e.get("sample_id") != sample_id]
        sidecar["entries"].append({
            "sample_id": sample_id,
            "polyline": P_data.tolist(),
            "seg_ids": gt_seg_ids,
            "meta": {"eps_px": eps_px, "seg_hit_frac": seg_hit_frac,
                     "fig_max_px": fig_max_px, "dpi": dpi},
        })
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        # --- optional PNG snapshot (same sizing) ---
        if save_png_dir:
            os.makedirs(save_png_dir, exist_ok=True)
            fig2 = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
            ax2 = fig2.add_subplot(111)
            lc2 = LineCollection(seg_arrays, colors=[_VIZ.base_gray]*len(seg_arrays),
                                 linewidths=widths, alpha=_VIZ.base_alpha, zorder=1)
            ax2.add_collection(lc2)
            _auto_limits(ax2, seg_arrays)
            ax2.set_aspect("equal"); ax2.axis("off")
            ax2.set_title(f"{sample_id} — GT ({len(gt_seg_ids)} segs)")
            P_save = P_plot.copy()
            ax2.plot(P_save[:,0], P_save[:,1], "-o", lw=3, ms=4, color=_VIZ.emph_color, zorder=3)
            out_png = Path(save_png_dir) / f"{sample_id}__taproot_gt.png"
            fig2.savefig(out_png, dpi=dpi, bbox_inches="tight")
            plt.close(fig2)

        return {"sample_id": sample_id, "polyline": P_data, "seg_ids": gt_seg_ids}

    finally:
        # restore only the viz keys we temporarily changed
        if _old_vals:
            set_viz_defaults(**_old_vals)

# ----------------------------------------------------------------------------

# -------------------- Manually Build Taproot --------------------------------
# Patch: add click-guided taproot builder that maps a hand-drawn polyline
# to the closest set of real segments, compatible with your existing
# TaprootResult / candidate schema.

from typing import Dict, List, Tuple, Optional, Iterable
import math
import json
import numpy as np
import networkx as nx

# Assumes these exist in your codebase (uploaded):
from .geometry import segment_length

# ---- Small helpers ---------------------------------------------------------

def _polyline_to_array(poly: List[List[float]]) -> np.ndarray:
    return np.asarray(poly, dtype=float)


def _polyline_length(poly: Iterable[Tuple[float, float]]) -> float:
    pts = np.asarray(list(poly), dtype=float)
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def _sample_polyline(poly: List[List[float]], step_px: float = 15.0) -> np.ndarray:
    """Return approximately equally spaced samples along the polyline."""
    P = _polyline_to_array(poly)
    if len(P) <= 1:
        return P
    # cumulative arc-length
    segs = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(segs)])
    L = float(s[-1])
    if L == 0:
        return P[:1]
    n = max(2, int(math.ceil(L / max(1e-6, step_px))) + 1)
    targets = np.linspace(0.0, L, n)
    out = []
    j = 0
    for t in targets:
        while j < len(segs) and s[j+1] < t:
            j += 1
        if j == len(segs):
            out.append(P[-1])
        else:
            t0, t1 = s[j], s[j+1]
            alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            out.append(P[j] * (1 - alpha) + P[j+1] * alpha)
    return np.asarray(out)


def _point_to_polyline_dist(pt: np.ndarray, poly: np.ndarray) -> float:
    """Min distance from a point to a polyline (poly as Nx2 array)."""
    if len(poly) == 0:
        return float("inf")
    # segment-wise projection
    diffs = np.diff(poly, axis=0)
    v2 = (diffs ** 2).sum(axis=1)
    v2[v2 == 0] = 1.0
    t = ((pt - poly[:-1]) * diffs).sum(axis=1) / v2
    t = np.clip(t, 0.0, 1.0)
    proj = poly[:-1] + diffs * t[:, None]
    d = np.linalg.norm(proj - pt, axis=1).min()
    return float(d)


def _segment_midpoint(seg: Dict) -> np.ndarray:
    Ps = np.asarray(seg["points"], dtype=float)
    return (Ps[0] + Ps[-1]) * 0.5


def _segment_heading(seg: Dict) -> float:
    Ps = np.asarray(seg["points"], dtype=float)
    if len(Ps) < 2:
        return 0.0
    v = Ps[-1] - Ps[0]
    return float(math.atan2(v[1], v[0]))


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b)
    d = (d + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def _axis_from_polyline(poly: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (origin, unit_direction) as dominant PCA axis of the clicks polyline."""
    P = _polyline_to_array(poly)
    if len(P) == 0:
        return np.zeros(2), np.array([1.0, 0.0])
    C = P.mean(axis=0)
    X = P - C
    if len(P) == 1:
        return C, np.array([1.0, 0.0])
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    dir0 = Vt[0]
    if dir0[0] < 0:  # make direction deterministic-ish
        dir0 = -dir0
    return C, dir0 / np.linalg.norm(dir0)


def _project_scalar(pt: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> float:
    return float(np.dot(pt - origin, axis))


def _concat_polylines(polys: List[List[List[float]]]) -> List[List[float]]:
    out: List[List[float]] = []
    prev = None
    for P in polys:
        if not P:
            continue
        if prev is not None and np.allclose(prev, P[0]):
            out.extend(P[1:])
        elif prev is not None and np.allclose(prev, P[-1]):
            out.extend(P[-2::-1])  # reverse P except duplicate
        else:
            if out and not np.allclose(out[-1], P[0]):
                out.extend(P)
            else:
                out.extend(P)
        prev = out[-1] if out else None
    return out


# =========================
# Taproot Annotator (v2.1)
# =========================

import os, json, math
from pathlib import Path
import numpy as np
import matplotlib, matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

from rootlab.viz import (
    _VIZ,
    _collect_segment_drawables,
    _auto_limits,
    set_viz_defaults,
    data_width_to_points,
)

# If these live elsewhere in your package, keep the imports as-is.
# They were used in your existing annotator:
#   _load_sample(json_file, sample_id)
#   _sample_polyline(points, step_px)
from rootlab.taproot import _load_sample, _sample_polyline


class _AbortAll(Exception):
    """Internal signal to stop remaining samples after user presses Esc."""
    pass


def _annotate_single_sample(
    json_file: str | os.PathLike,
    sample_id: str,
    save_to: str,
    *,
    invert_y: bool | None,
    buffer_px: float,
    overlap_threshold: float,
    show_clicks: bool,
    min_clicks: int,
    viz_overrides: dict | None,
    save_png_dir: str | None,
    fig_max_px: int,
    dpi: int,
    angle_max: int,
) -> dict | None:
    """
    Runs the interactive annotator for a single sample.
    Returns {sample_id: entry} on success; raises _AbortAll on Esc.
    """
    # ---- backend guard (Tk only) ----
    if "tk" not in matplotlib.get_backend().lower():
        raise RuntimeError("This function requires the Tk backend. Run `%matplotlib tk` first.")

    # ---- temporary viz overrides ----
    _old_vals = {}
    if viz_overrides:
        for k, v in viz_overrides.items():
            if hasattr(_VIZ, k):
                _old_vals[k] = getattr(_VIZ, k)
        set_viz_defaults(**{k: v for k, v in viz_overrides.items() if hasattr(_VIZ, k)})

    try:
        # ---- load sample & segment map ----
        img, segmap = _load_sample(json_file, sample_id)
        inv = _VIZ.invert_y_default if invert_y is None else bool(invert_y)
        all_seg_ids = list(segmap.keys())
        seg_arrays, widths = _collect_segment_drawables(segmap, all_seg_ids, invert_y=inv)
        if not seg_arrays:
            raise ValueError(f"No drawable segments for sample '{sample_id}'.")

        # --- compute figure size ---
        xs, ys = [], []
        for s in seg_arrays:
            a = np.asarray(s, float)
            if a.ndim == 2 and a.shape == (2, 2):
                xs.extend([a[0, 0], a[1, 0]]); ys.extend([a[0, 1], a[1, 1]])
            elif a.ndim == 3 and a.shape[-2:] == (2, 2):
                xs.extend(a[..., 0].ravel().tolist()); ys.extend(a[..., 1].ravel().tolist())
        xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
        w_data, h_data = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
        aspect = w_data / h_data
        if aspect >= 1.0:
            width_px, height_px = int(fig_max_px), max(1, int(round(fig_max_px / aspect)))
        else:
            height_px, width_px = int(fig_max_px), max(1, int(round(fig_max_px * aspect)))
        fig_w_in, fig_h_in = width_px / dpi, height_px / dpi

        # ---- figure setup ----
        fig = plt.figure(num=f"taproot-annotator-{sample_id}", figsize=(fig_w_in, fig_h_in), dpi=dpi)
        fig.clf()
        ax = fig.add_subplot(111)

        widths_pts = [data_width_to_points(ax, w_px) for w_px in widths]
        lc = LineCollection(
            seg_arrays,
            colors=[_VIZ.base_gray] * len(seg_arrays),
            linewidths=widths_pts,
            alpha=_VIZ.base_alpha,
            zorder=1,
        )
        ax.add_collection(lc)
        _auto_limits(ax, seg_arrays)

        # ---- calibration overlay ----
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        cal_y = ymin + 0.02 * (ymax - ymin)
        ax.plot([xmin, xmin + buffer_px], [cal_y, cal_y], 'k-', lw=1.5, zorder=10)
        ax.text(xmin, cal_y - 0.015 * (ymax - ymin),
                f"{int(buffer_px)} data px", color='k', va='top', fontsize=8)

        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{sample_id} — Left-click add • Backspace undo • Enter finish • Esc abort all")

        # ---- interactive overlay ----
        lw_pts = data_width_to_points(ax, buffer_px)
        pts: list[list[float]] = []
        (line,) = ax.plot([], [], "-", lw=lw_pts, color=_VIZ.emph_color, alpha=_VIZ.taproot_alpha, zorder=3)
        (mark,) = ax.plot([], [], "o", ms=4, color=_VIZ.emph_color, zorder=4)

        fig.canvas.draw()
        try: fig.show()
        except Exception: pass

        abort_all = False

        def on_click(e):
            if e.inaxes is not ax or e.button != 1 or e.xdata is None or e.ydata is None:
                return
            pts.append([e.xdata, e.ydata])
            P = np.asarray(pts, float)
            line.set_data(P[:, 0], P[:, 1]); mark.set_data(P[:, 0], P[:, 1])
            fig.canvas.draw_idle()

        def on_key(e):
            nonlocal abort_all
            if e.key == "backspace":
                if pts:
                    pts.pop()
                    P = np.asarray(pts, float)
                    line.set_data(P[:, 0], P[:, 1]); mark.set_data(P[:, 0], P[:, 1])
                    fig.canvas.draw_idle()
            elif e.key in ("escape",):
                abort_all = True
                plt.close(fig)
            elif e.key in ("enter", "return"):
                plt.close(fig)

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)

        clicks = plt.ginput(n=-1, timeout=0, show_clicks=show_clicks)
        try: plt.close(fig)
        except Exception: pass

        if abort_all:
            raise _AbortAll()

        # ---- clicks → data coords ----
        if len(clicks) >= min_clicks:
            pts = clicks
        if len(pts) < min_clicks:
            # No annotation for this sample; skip (do not raise)
            return None

        P_data = np.asarray(pts, float)
        if inv:
            P_data[:, 1] = -P_data[:, 1]
        AL = np.asarray(P_data, float)

        # ---- densify AL for geometry ----
        AL_dense = np.asarray(_sample_polyline(AL.tolist(), step_px=2.0), float)
        halfw = float(buffer_px) / 2.0

        # --- KD-tree on AL midpoints for local tangent ---
        AL_midpoints = 0.5 * (AL_dense[:-1] + AL_dense[1:])
        tree = cKDTree(AL_midpoints)

        def _angle_deg(u: np.ndarray, v: np.ndarray) -> float | None:
            """Folded angle (0–90°) between vectors u and v; None if degenerate."""
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu == 0 or nv == 0:
                return None
            c = np.clip(np.dot(u, v) / (nu * nv), -1, 1)
            ang = math.degrees(math.acos(c))
            return min(ang, 180.0 - ang)

        def _dist_points_polyline(pts_arr: np.ndarray, poly: np.ndarray) -> np.ndarray:
            """Vectorized distances from many points to a polyline."""
            diffs = np.diff(poly, axis=0)
            v2 = (diffs ** 2).sum(axis=1)
            v2[v2 == 0] = 1.0
            d = pts_arr[:, None, :] - poly[None, :-1, :]
            t = np.clip((d * diffs[None, :, :]).sum(axis=2) / v2, 0, 1)
            proj = poly[None, :-1, :] + diffs[None, :, :] * t[:, :, None]
            dist2 = ((pts_arr[:, None, :] - proj) ** 2).sum(axis=2)
            return np.sqrt(dist2.min(axis=1))

        def _coverage_frac(segment_points: list[list[float]], step_px: float = 3.0) -> float:
            """Approximate length coverage: fraction of sampled points inside the corridor."""
            S = np.asarray(segment_points, float)
            if S.shape[0] < 2:
                return 0.0
            d = np.linalg.norm(np.diff(S, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(d)])
            L = float(cum[-1])
            if L <= 0:
                return 0.0
            n = max(2, int(round(L / max(1e-6, step_px))) + 1)
            t = np.linspace(0.0, L, n)
            idx = np.searchsorted(cum, t, side="right") - 1
            idx = np.clip(idx, 0, len(d) - 1)
            alpha = (t - cum[idx]) / np.maximum(d[idx], 1e-6)
            P = S[idx] * (1 - alpha)[:, None] + S[idx + 1] * alpha[:, None]
            dist = _dist_points_polyline(P, AL_dense)
            inside = dist <= (halfw + 0.5)
            return inside.mean()

        # ---- segment selection ----
        kept_ids = []
        for sid, seg in segmap.items():
            P = np.asarray(seg["points"], float)
            if len(P) < 2:
                continue

            cov = _coverage_frac(seg["points"])
            dist = _dist_points_polyline(P, AL_dense)
            min_dist = float(dist.min())
            seg_width = abs(seg["meta"].get("width_right", 0) - seg["meta"].get("width_left", 0))

            # Coverage + width-tolerant inclusion
            if cov >= overlap_threshold or min_dist <= (halfw + 0.5 + 0.5 * seg_width):
                # Local tangent angle
                centroid = P.mean(axis=0)
                _, j = tree.query(centroid)
                # j is 0..(len(AL_dense)-2) because midpoints = len-1
                j = int(min(max(j, 0), len(AL_dense) - 2))
                v_al = AL_dense[j + 1] - AL_dense[j]
                v_seg = P[-1] - P[0]
                ang = _angle_deg(v_seg, v_al)
                if (ang is None) or (ang <= angle_max):
                    kept_ids.append(sid)

        # ---- write JSON sidecar (append/replace this sample) ----
        sidecar_path = Path(save_to)
        if sidecar_path.exists():
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                entries = sidecar.get("entries", [])
            except Exception:
                entries = []
        else:
            entries = []

        # drop existing entry for this sample_id
        entries = [e for e in entries if e.get("sample_id") != sample_id]

        new_entry = {
            "sample_id": sample_id,
            "polyline": AL.tolist(),
            "seg_ids": kept_ids,
            "meta": {
                "buffer_px": buffer_px,
                "overlap_threshold": overlap_threshold,
                "fig_max_px": fig_max_px,
                "dpi": dpi,
                "angle_max": angle_max,
            },
        }
        entries.append(new_entry)
        sidecar_path.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")

        # ---- optional PNG snapshot ----
        if save_png_dir:
            os.makedirs(save_png_dir, exist_ok=True)
            fig2 = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
            ax2 = fig2.add_subplot(111)
            lc2 = LineCollection(seg_arrays, colors=[_VIZ.base_gray]*len(seg_arrays),
                                 linewidths=widths_pts, alpha=_VIZ.base_alpha, zorder=1)
            ax2.add_collection(lc2)
            _auto_limits(ax2, seg_arrays)
            ax2.set_aspect("equal"); ax2.axis("off")
            lw_pts2 = data_width_to_points(ax2, buffer_px)
            ax2.plot(AL[:, 0], AL[:, 1], "-", lw=lw_pts2,
                     color=_VIZ.emph_color, alpha=_VIZ.taproot_alpha, zorder=3)
            ax2.set_title(f"{sample_id} — GT (kept {len(kept_ids)} segs)")
            out_png = Path(save_png_dir) / f"{sample_id}__taproot_gt.png"
            fig2.savefig(out_png, dpi=dpi, bbox_inches="tight")
            plt.close(fig2)

        # in-memory form mirrors your prior return shape
        return {sample_id: new_entry}

    finally:
        if _old_vals:
            set_viz_defaults(**_old_vals)


def annotate_taproot(
    json_file: str | os.PathLike,
    sample_id: str | list[str] | None = None,
    save_to: str = "taproot_gt.json",
    *,
    invert_y: bool | None = None,
    buffer_px: float = 20.0,
    overlap_threshold: float = 0.95,
    show_clicks: bool = True,
    min_clicks: int = 2,
    viz_overrides: dict | None = None,
    save_png_dir: str | None = None,
    fig_max_px: int = 900,
    dpi: int = 120,
    angle_max: int = 45,
) -> dict | None:
    """
    Interactive taproot annotator.
    - If `sample_id` is a list/tuple, iterates through them.
    - Press Enter to save current sample & advance.
    - Press Esc to abort current sample AND stop remaining samples.
    - Already-completed samples are preserved on disk.
    """
    # If no sample_id provided, expand to ALL samples in the json
    if sample_id is None:
        # Match package convention from utils.extract_segment_maps(...)
        try:
            from rootlab.io import load_json
            data = load_json(json_file)
        except Exception:
            # fall back to stdlib if needed
            import json as _json
            from pathlib import Path
            data = _json.loads(Path(json_file).read_text(encoding="utf-8"))
    
        # Collect experiments that look like real sample bundles
        sample_id = [
            str(img["info"]["experiment"])
            for img in data.get("images", [])
            if isinstance(img, dict)
            and isinstance(img.get("info"), dict)
            and img["info"].get("experiment") is not None
        ]

    if not sample_id:
        raise ValueError("Could not infer any sample IDs from json_file; please pass sample_id explicitly.")


    
    # multi-sample driver
    if isinstance(sample_id, (list, tuple)):
        results = {}
        for sid in sample_id:
            print(f"\n=== Annotating sample {sid} ===")
            try:
                single_res = _annotate_single_sample(
                    json_file, sid, save_to,
                    invert_y=invert_y,
                    buffer_px=buffer_px,
                    overlap_threshold=overlap_threshold,
                    show_clicks=show_clicks,
                    min_clicks=min_clicks,
                    viz_overrides=viz_overrides,
                    save_png_dir=save_png_dir,
                    fig_max_px=fig_max_px,
                    dpi=dpi,
                    angle_max=angle_max,
                )
            except _AbortAll:
                print("🛑 Aborted by user — stopping remaining samples.")
                break
            if single_res is not None:
                results.update(single_res)
        return results

    # single-sample path
    try:
        return _annotate_single_sample(
            json_file, sample_id, save_to,
            invert_y=invert_y,
            buffer_px=buffer_px,
            overlap_threshold=overlap_threshold,
            show_clicks=show_clicks,
            min_clicks=min_clicks,
            viz_overrides=viz_overrides,
            save_png_dir=save_png_dir,
            fig_max_px=fig_max_px,
            dpi=dpi,
            angle_max=angle_max,
        )
    except _AbortAll:
        # preserve previous behavior for single sample: no output on Esc
        return None