"""Depth-first search tree, linking continuous segments into coherent branches. Uses helpers from geometry, graph, and utils."""
from __future__ import annotations
from typing import Iterable, Mapping, Any, Dict, List, Set
import numpy as np

from .geometry import segment_length as compute_segment_length, segment_width as compute_segment_width
from .graph import (
    classify_links,
    build_end_to_end_map,
    merge_end_links,
    connected_components_end_to_end,
    build_branch_adjacency,
)
from .utils import compute_biological_order, flip_parent_if_thicker

def build_dfs_tree(
    system_segments: Iterable[str],
    links: Iterable[Mapping[str, Any]],
    segment_map: Mapping[str, Mapping[str, Any]],
    *,
    apply_thickness_flip: bool = False,
    flip_threshold: float = 0.2,
    start_strategy: str = "thickest",   # 'thickest' | 'nth_thickest' | 'weighted'
    n_rank: int = 0,                    # used only for 'nth_thickest'
    show_top_candidates: bool = False,
    top_n_display: int = 10,
    visualize_candidates: bool = False,
    candidate_count: int = 5,
):
    system_segments = set(system_segments)

    # 1) classify links
    end_to_end_links, branch_links = classify_links(links, system_segments, segment_map)

    # 2) end_to_end connectivity
    e2e_map = build_end_to_end_map(end_to_end_links)

    # 3) pre-groups (connected components under e2e)
    pre_groups = connected_components_end_to_end(system_segments, e2e_map)

    # 4) compute metrics per group
    group_metrics: list[dict] = []
    for group in pre_groups:
        total_len = 0.0
        weighted_sum = 0.0
        for seg_id in group:
            length = compute_segment_length(segment_map[seg_id]["points"])
            width = max(compute_segment_width(segment_map[seg_id]["meta"]), 0.0)
            weighted_sum += width * length
            total_len += length
        group_width = (weighted_sum / total_len) if total_len > 0 else 0.0
        score = (group_width ** 2) * total_len
        group_metrics.append({
            "group": group,
            "width": group_width,
            "length": total_len,
            "score": score,
        })

    # 5) rank candidates
    if start_strategy in ("thickest", "nth_thickest"):
        group_metrics.sort(key=lambda x: x["width"], reverse=True)
    elif start_strategy == "weighted":
        group_metrics.sort(key=lambda x: x["score"], reverse=True)
    else:
        raise ValueError("start_strategy must be 'thickest', 'nth_thickest', or 'weighted'.")

    if not group_metrics:
        return []

    if start_strategy == "nth_thickest":
        if not (0 <= n_rank < len(group_metrics)):
            raise IndexError(f"n_rank {n_rank} is out of range for {len(group_metrics)} groups.")

    if show_top_candidates:
        k = min(top_n_display, len(group_metrics))
        print(f"\nTop {k} candidates for '{start_strategy}':")
        for i, gm in enumerate(group_metrics[:k], start=1):
            print(f"{i}. Width={gm['width']:.3f}, Length={gm['length']:.3f}, "
                  f"Score={gm['score']:.3f}, Segments={len(gm['group'])}")

    if visualize_candidates:
        # Import here to avoid hard dependency when not used
        from .viz import visualize_candidate_groups
        visualize_candidate_groups(system_segments, segment_map, group_metrics, candidate_count)

    # 6) choose start group
    start_idx = n_rank if start_strategy == "nth_thickest" else 0
    start_group = group_metrics[start_idx]["group"]
    if not start_group:
        return []

    # 7) DFS outward across branch connections
    branch_map = build_branch_adjacency(branch_links)
    visited: Set[str] = set()
    groups: List[set[str]] = [merge_end_links(next(iter(start_group)), e2e_map, visited)]
    parent_child_map: Dict[int, List[int]] = {}

    def dfs_build_all(current_id: int):
        current_group = groups[current_id]
        parent_child_map[current_id] = []
        for seg in current_group:
            for neighbor in branch_map.get(seg, []):
                if neighbor not in visited:
                    child_group = merge_end_links(neighbor, e2e_map, visited)
                    if child_group:
                        child_id = len(groups)
                        groups.append(child_group)
                        parent_child_map[current_id].append(child_id)
                        dfs_build_all(child_id)

    dfs_build_all(0)

    # 8) per-branch metrics
    group_length: Dict[int, float] = {}
    group_width: Dict[int, float] = {}
    for gid, seg_ids in enumerate(groups):
        total_len = 0.0
        weighted_sum = 0.0
        for seg_id in seg_ids:
            length = compute_segment_length(segment_map[seg_id]["points"])
            width = max(compute_segment_width(segment_map[seg_id]["meta"]), 0.0)
            weighted_sum += width * length
            total_len += length
        group_length[gid] = total_len
        group_width[gid] = (weighted_sum / total_len) if total_len > 0 else 0.0

    # 9) biological order, optional flip
    order_before = compute_biological_order(parent_child_map)
    if apply_thickness_flip:
        # Lazy import to keep utils light; pass any structure you already use
        try:
            import pandas as pd  # optional
            branch_table = pd.DataFrame({
                "branch_id": list(range(len(groups))),
                "branch_width": [group_width[g] for g in range(len(groups))]
            })
        except Exception:
            branch_table = {"branch_id": list(range(len(groups))),
                            "branch_width": [group_width[g] for g in range(len(groups))]}
        flipped_map = flip_parent_if_thicker(branch_table, parent_child_map, threshold=flip_threshold)
        order_after = compute_biological_order(flipped_map)
    else:
        flipped_map = parent_child_map
        order_after = order_before

    # 10) return tidy records
    parent_lookup = {c: p for p, children in parent_child_map.items() for c in children}
    out = []
    for gid, seg_ids in enumerate(groups):
        out.append({
            "branch_id": gid,
            "dfs_order": gid + 1,
            "branch_length": group_length[gid],
            "branch_width": group_width[gid],
            "branch_parent": parent_lookup.get(gid),
            "branch_child": parent_child_map.get(gid, []),
            "n_child": len(parent_child_map.get(gid, [])),
            "n_seg": len(seg_ids),
            "seg_ids": ",".join(seg_ids),
            "order_before_flip": order_before.get(gid),
            "order_after_flip": order_after.get(gid),
        })
    return out


# -----------------------
# Evaluate methods of choosing starting point
# -----------------------

def choose_starting_branch(json_file: str,
                           sample_id: str,
                           system_id: str,
                           start_strategy: str = "thickest",   # 'thickest' | 'nth_thickest' | 'weighted'
                           visualize_candidates: bool = True,
                           candidate_count: int = 5) -> dict:
    """
    Select (and optionally visualize) the starting branch for DFS tree building.

    Returns a dict like:
      {"group": set[str], "width": float, "length": float, "score": float}
    """
    import json
    import numpy as np
    try:
        import networkx as nx
    except Exception as e:
        raise RuntimeError("choose_starting_branch requires networkx installed") from e

    # --- Load JSON and extract sample
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    sample = next(img for img in data["images"] if img["info"]["experiment"] == sample_id)

    # --- Segment map and links for this sample
    segment_map: Mapping[str, Mapping[str, Any]] = {
        seg["name"]: seg for root in sample["roots"] for seg in root["segments"]
    }
    all_links: List[Mapping[str, Any]] = [link for root in sample["roots"] for link in root["links"]]

    # --- Find connected components (systems) via NetworkX
    G = nx.Graph()
    G.add_nodes_from(segment_map.keys())
    G.add_edges_from([(l["seg1"], l["seg2"]) for l in all_links])
    components = sorted(nx.connected_components(G), key=len, reverse=True)

    # Parse system_id like "C14C_1" → index 0
    sys_index = int(system_id.split("_")[-1]) - 1
    system_segments = set(components[sys_index])

    # --- Classify links and compute end_to_end groups
    e2e_links, _branch_links = classify_links(all_links, system_segments, segment_map)
    e2e_map = build_end_to_end_map(e2e_links)
    pre_groups = connected_components_end_to_end(system_segments, e2e_map)

    # --- Compute metrics for each group
    group_metrics: List[dict] = []
    for group in pre_groups:
        total_len = 0.0
        weighted_sum = 0.0
        for seg_id in group:
            length = compute_segment_length(segment_map[seg_id]["points"])
            width = max(compute_segment_width(segment_map[seg_id]["meta"]), 0.0)
            weighted_sum += width * length
            total_len += length
        group_width = (weighted_sum / total_len) if total_len > 0 else 0.0
        score = (group_width ** 2) * total_len
        group_metrics.append({"group": group, "width": group_width, "length": total_len, "score": score})

    # --- Sort groups according to strategy
    if start_strategy in ("thickest", "nth_thickest"):
        group_metrics.sort(key=lambda x: x["width"], reverse=True)
    elif start_strategy == "weighted":
        group_metrics.sort(key=lambda x: x["score"], reverse=True)
    else:
        raise ValueError("Invalid start_strategy: use 'thickest', 'nth_thickest', or 'weighted'.")

    # --- Visualize (optional)
    if visualize_candidates:
        from .viz import visualize_candidate_groups  # imported only if needed
        visualize_candidate_groups(system_segments, segment_map, group_metrics, candidate_count)

    # Pick the top candidate for the chosen strategy
    selected = group_metrics[0]
    print(
        f"\nSelected Group for '{start_strategy}': "
        f"Width={selected['width']:.3f}, Length={selected['length']:.3f}, "
        f"Score={selected['score']:.3f}, Segments={len(selected['group'])}"
    )
    return selected

# --- Single function to wrap up all others and process the json directly ---
# --- Step 1: Load JSON file
# --- Step 2: Extract Sample IDs
# --- Step 3: Use extract_root_systems to obtain connected sets of segments, and segment map
# --- Step 4: For each system of connected segments, build the DFS tree
# --- Step 5: 
import pandas as pd
from .io import load_json
from .utils import extract_root_systems

def process_json(
    json_file: str,
    sample_ids: list[str] | None = None,
    *,
    apply_thickness_flip: bool = True,
    flip_threshold: float = 0.2,
    start_strategy: str = "thickest",
    n_rank: int = 0,
    px_per_cm: float = 312.0,
    min_children: int | None = None,
    min_length: float | None = None,
    max_order: int | None = None,
    flipped_order: bool = True,
    return_diagnostics: bool = False
):
    """
    Process a full JSON file to build DFS trees for all systems in specified samples.

    Returns
    -------
    system_summary_df : pd.DataFrame
        Simple per-system branch summary (from summarize_branches).
    branch_table_df : pd.DataFrame
        Long-form table of per-branch attributes.
    segment_maps : dict[str, dict]
        Mapping of sample_id -> segment_map.
    diag_df : pd.DataFrame, optional
        Diagnostic summary if return_diagnostics=True.
    """
    from .utils import summarize_branches

    data = load_json(json_file)
    all_records = []
    segment_maps = {}

    # Loop through all images or specified subset
    imgs = data["images"]
    if sample_ids is not None:
        imgs = [img for img in imgs if img["info"]["experiment"] in sample_ids]

    for img in imgs:
        sample_id = img["info"]["experiment"]
        
        system_bundle = extract_root_systems(data, sample_id)[sample_id]
        segment_map = system_bundle["segment_map"]
        links = system_bundle["links"]
        components = system_bundle["components"]
        segment_maps[sample_id] = segment_map

        for system_index, system_segments in enumerate(components):
            branch_records = build_dfs_tree(
                system_segments=set(system_segments),
                links=links,
                segment_map=segment_map,
                apply_thickness_flip=apply_thickness_flip,
                flip_threshold=flip_threshold,
                start_strategy=start_strategy,
                n_rank=n_rank,
            )
            # Add sample/system info
            for rec in branch_records:
                rec["sample_id"] = sample_id
                rec["system_id"] = system_index
            all_records.extend(branch_records)

    branch_df = pd.DataFrame(all_records)

    # Simple per-system summary (keep diagnostics optional)
    summary, _, *diag = summarize_branches(
        branch_df,
        px_per_cm=px_per_cm,
        min_children=min_children,
        min_length=min_length,
        max_order=max_order,
        flipped_order=flipped_order,
        return_diagnostics=return_diagnostics
    )

    if return_diagnostics:
        diag_df = diag[0]
        return summary, branch_df, segment_maps, diag_df
    else:
        return summary, branch_df, segment_maps
