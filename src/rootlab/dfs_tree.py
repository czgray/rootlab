# Copyright 2025 Christian Gray
#
# File: dfs_tree.py
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

"""Depth-first search tree, linking continuous segments into coherent branches. Uses helpers from geometry, graph, and utils."""
from __future__ import annotations
from typing import Iterable, Mapping, Any, Dict, List, Set
import numpy as np
import pandas as pd
import json

from .geometry import segment_length, segment_width
from .graph import (
    classify_links,
    build_end_to_end_map,
    merge_end_links,
    connected_components_end_to_end,
    build_branch_adjacency,
)    
from .utils import compute_biological_order, flip_parent_if_thicker, extract_root_systems, summarize_branches
from .io import load_json
    

                
def build_dfs_tree(
    system_segments: Iterable[str],
    links: Iterable[Mapping[str, Any]],
    segment_map: Mapping[str, Mapping[str, Any]],
    *,
    apply_thickness_flip: bool = False,
    flip_threshold: float = 0.2,
    start_strategy: str = "weighted",   # 'thickest' | 'nth_thickest' | 'weighted' | 'manual'
    n_rank: int = 0,
    show_top_candidates: bool = False,
    top_n_display: int = 10,
    visualize_candidates: bool = False,
    candidate_count: int = 5,
    taproot_map: dict[str, dict] | None = None,
    sample_id: str | None = None,       # required only for 'manual'
):
    """
    Build a DFS-style hierarchy of branches from connected segments.

    New behavior (start_strategy='manual'):
        - Requires taproot_map[sample_id] with 'seg_ids' list.
        - All annotated seg_ids are merged into one root branch.
        - Systems without a manual taproot entry should be skipped upstream.

    Returns a list of per-branch dicts as before.
    """
    system_segments = set(system_segments)

    # --- classify links
    end_to_end_links, branch_links = classify_links(links, system_segments, segment_map)
    e2e_map = build_end_to_end_map(end_to_end_links)
    pre_groups = connected_components_end_to_end(system_segments, e2e_map)
    
    # --- manual override ----------------------------------------------------
    if start_strategy == "manual":
        if taproot_map is None:
            raise ValueError("start_strategy='manual' requires a provided taproot_map.")
        if sample_id is None:
            raise ValueError("start_strategy='manual' requires sample_id to identify taproot.")
        if sample_id not in taproot_map:
            raise ValueError(f"No manual taproot found for sample '{sample_id}' in taproot_map.")

        manual_ids = set(str(s) for s in taproot_map[sample_id].get("seg_ids", []))
        if not manual_ids:
            raise ValueError(f"Manual taproot for sample '{sample_id}' has no seg_ids.")

        # restrict to valid ones within this system
        valid_manual = [sid for sid in manual_ids if sid in system_segments]
        if not valid_manual:
            raise ValueError(f"None of the manual seg_ids for '{sample_id}' are in this system.")

        # expand manual seg_ids to include end_to_end links 
        expanded_manual = set()
        for sid in valid_manual:
            expanded_manual |= merge_end_links(sid, e2e_map, set())

        # force-merge into one root group (expanded taproot)
        start_group = expanded_manual
        print(f"🌿 Taproot expanded: {len(valid_manual)} → {len(start_group)} segments for {sample_id}")


        # build branch adjacency (for DFS)
        branch_map = build_branch_adjacency(branch_links)
        visited: Set[str] = set(start_group)
        groups: List[set[str]] = [start_group]
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

        # --- Build tree for singletons (are inherently filtered out because they have no linkage)
        for seg in system_segments:
            if seg not in visited:
                root = merge_end_links(seg, e2e_map, visited)
                if not root:
                    continue
                new_id = len(groups)
                groups.append(root)
                parent_child_map[new_id] = []   # new root, no parent
                dfs_build_all(new_id)

    else:
        # --- existing ranking logic for non-manual modes -------------------
        group_metrics: list[dict] = []
        for group in pre_groups:
            total_len = 0.0
            weighted_sum = 0.0
            for seg_id in group:
                length = segment_length(segment_map[seg_id]["points"])
                width = max(segment_width(segment_map[seg_id]["meta"]), 0.0)
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

        if start_strategy in ("thickest", "nth_thickest"):
            group_metrics.sort(key=lambda x: x["width"], reverse=True)
        elif start_strategy == "weighted":
            group_metrics.sort(key=lambda x: x["score"], reverse=True)
        else:
            raise ValueError(
                "start_strategy must be one of 'thickest', 'nth_thickest', "
                "'weighted', or 'manual'."
            )

        if not group_metrics:
            return []

        if start_strategy == "nth_thickest":
            if not (0 <= n_rank < len(group_metrics)):
                raise IndexError(f"n_rank {n_rank} is out of range for {len(group_metrics)} groups.")

        if show_top_candidates:
            k = min(top_n_display, len(group_metrics))
            print(f"\nTop {k} candidates for '{start_strategy}':")
            for i, gm in enumerate(group_metrics[:k], start=1):
                print(
                    f"{i}. Width={gm['width']:.3f}, Length={gm['length']:.3f}, "
                    f"Score={gm['score']:.3f}, Segments={len(gm['group'])}"
                )

        if visualize_candidates:
            from .viz import visualize_candidate_groups
            visualize_candidate_groups(system_segments, segment_map, group_metrics, candidate_count)

        start_idx = n_rank if start_strategy == "nth_thickest" else 0
        start_group = group_metrics[start_idx]["group"]

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

        # --- Build tree for singletons (are inherently filtered out because they have no linkage)
        for seg in system_segments:
            if seg not in visited:
                root = merge_end_links(seg, e2e_map, visited)
                if not root:
                    continue
                new_id = len(groups)
                groups.append(root)
                parent_child_map[new_id] = []   # new root, no parent
                dfs_build_all(new_id)

    # --- identical downstream metrics & ordering ----------------------------
    group_length: Dict[int, float] = {}
    group_width: Dict[int, float] = {}
    group_volume: Dict[int, float] = {}  
    for gid, seg_ids in enumerate(groups):
        total_len = 0.0
        weighted_sum = 0.0
        total_vol = 0.0
        for seg_id in seg_ids:
            length = segment_length(segment_map[seg_id]["points"])
            width = max(segment_width(segment_map[seg_id]["meta"]), 0.0)
            weighted_sum += width * length
            total_len += length
            # --- compute segment-level volume and accumulate
            total_vol += np.pi * (width / 2.0) ** 2 * length
            
        group_length[gid] = total_len
        group_width[gid] = (weighted_sum / total_len) if total_len > 0 else 0.0
        group_volume[gid] = total_vol

    order_before = compute_biological_order(parent_child_map)
    if apply_thickness_flip:
        try:
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

    parent_lookup = {c: p for p, children in parent_child_map.items() for c in children}
    out = []
    for gid, seg_ids in enumerate(groups):
        out.append({
            "branch_id": gid,
            "dfs_order": gid + 1,
            "branch_length": group_length[gid],
            "branch_width": group_width[gid],
            "branch_volume_px3": group_volume[gid],
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
            length = segment_length(segment_map[seg_id]["points"])
            width = max(segment_width(segment_map[seg_id]["meta"]), 0.0)
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


def process_json(
    json_file: str,
    sample_ids: list[str] | None = None,
    *,
    apply_thickness_flip: bool = True,
    flip_threshold: float = 0.2,
    start_strategy: str = "weighted",
    taproot_map: dict[str, dict] | None = None,
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
    
    data = load_json(json_file)
    all_records = []
    segment_maps = {}
    
    # Loop through all images or specified subset
    imgs = data["images"]
    if sample_ids is not None:
        imgs = [img for img in imgs if img["info"]["experiment"] in sample_ids]
        
    for img in imgs:
        sample_id = img["info"]["experiment"]
        
        #-- Diagnostic tool
        if start_strategy == "manual" and (
            taproot_map is None or sample_id not in taproot_map
        ):
            print(f"⚠️ Skipping {sample_id}: no manual taproot entry.")
            continue

        
        system_bundle = extract_root_systems(data, sample_id)[sample_id]
        segment_map = system_bundle["segment_map"]
        links = system_bundle["links"]
        components = system_bundle["components"]
        segment_maps[sample_id] = segment_map

        # print(f"   Components before merge: {len(system_bundle['components'])}")
        # print(f"🎯 Beginning processing for sample {sample_id}")
        # print(f"   segment_map has {len(segment_map)} segments")
    
        # --- Merge systems if taproot spans multiple components (manual start only) ---
        if start_strategy == "manual" and taproot_map is not None and sample_id in taproot_map:
            taproot_ids = set(str(s) for s in taproot_map[sample_id].get("seg_ids", []))
            overlap_indices = [
                i for i, comp in enumerate(components)
                if len(set(comp) & taproot_ids) > 0
            ]
            if overlap_indices:
                merged_system = set().union(*[components[i] for i in overlap_indices])
                components = [merged_system] + [
                    comp for i, comp in enumerate(components) if i not in overlap_indices
                ]
                # print(f"✅ Merged {len(overlap_indices)} systems into 1 for sample {sample_id}")
                # print(f"   Taproot segments: {len(taproot_ids)}  |  "
                #       f"System[0] size: {len(components[0])}")
    
        # --- Loop over each connected component (system) ---
        for system_index, system_segments in enumerate(components):

            # --- Fallback: only the first system (merged) uses manual ---
            if start_strategy == "manual" and system_index > 0:
                local_start_strategy = "weighted"
            else:
                local_start_strategy = start_strategy

            # Build dictionary for build_dfs_tree
            kwargs = dict(
                system_segments=set(system_segments),
                links=links,
                segment_map=segment_map,
                apply_thickness_flip=apply_thickness_flip,
                flip_threshold=flip_threshold,
                start_strategy=local_start_strategy,
                n_rank=n_rank,
            )
    
            if start_strategy == "manual":
                if taproot_map is None:
                    raise ValueError("start_strategy='manual' requires a taproot_map.")
                kwargs.update(dict(taproot_map=taproot_map, sample_id=sample_id))
                # print(f" Passing {len(kwargs['system_segments'])} segments into build_dfs_tree for {sample_id}")
    
            branch_records = build_dfs_tree(**kwargs)
    
            # Add sample/system info
            for rec in branch_records:
                rec["sample_id"] = sample_id
                rec["system_id"] = system_index

            # print(f"🔎 {sample_id}: build_dfs_tree returned {len(branch_records)} branches")
            all_records.extend(branch_records)

    
    # Combine everything, AFTER END of image loop.
    branch_df = pd.DataFrame(all_records)

    # print(f"✅ Finished processing for sample {sample_id}\n")
    print("Columns in branch_df:", list(branch_df.columns))
    print("Branch_df length:", len(branch_df))

    
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

    # Return dataframe, w/ or w/o diagnostics
    if return_diagnostics:
        diag_df = diag[0]
        return summary, branch_df, segment_maps, diag_df
    else:
        return summary, branch_df, segment_maps



        # for system_index, system_segments in enumerate(components):
        #     branch_records = build_dfs_tree(
        #         system_segments=set(system_segments),
        #         links=links,
        #         segment_map=segment_map,
        #         apply_thickness_flip=apply_thickness_flip,
        #         flip_threshold=flip_threshold,
        #         start_strategy=start_strategy,
        #         n_rank=n_rank,
        #     )