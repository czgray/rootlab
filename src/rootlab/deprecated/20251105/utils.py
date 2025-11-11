"""Miscellaneous helpers """
from __future__ import annotations
from typing import Dict, List, Set, Iterable, Mapping, Any, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx
import json
from pathlib import Path

def flatten(list_of_lists):
    return [x for sub in list_of_lists for x in sub]

# -----------------------
# Biological Order & Flipping parent-child relationship based on thickness
# -----------------------
def compute_biological_order(parent_child_map):
    """Compute biological order from parent-child hierarchy."""
    all_nodes = set(parent_child_map.keys()) | {c for children in parent_child_map.values() for c in children}
    order_map = {node: None for node in all_nodes}
    while any(v is None for v in order_map.values()):
        for node in order_map:
            if order_map[node] is not None:
                continue
            children = parent_child_map.get(node, [])
            if not children:
                order_map[node] = 1
            else:
                child_orders = [order_map.get(ch) for ch in children]
                if all(co is not None for co in child_orders):
                    order_map[node] = max(child_orders) + 1
    return order_map

def flip_parent_if_thicker(branch_table_df, parent_child_map, threshold=0.2):
    """Flip parent-child if child is thicker than parent by threshold."""
    width_lookup = branch_table_df.set_index("branch_id")["branch_width"].to_dict()
    adjusted_map = {p: list(children) for p, children in parent_child_map.items()}
    for parent, children in list(parent_child_map.items()):
        for child in list(children):
            parent_width = width_lookup.get(parent, 0)
            child_width = width_lookup.get(child, 0)
            if child_width > parent_width * (1 + threshold):
                adjusted_map[parent].remove(child)
                adjusted_map.setdefault(child, []).append(parent)
    return adjusted_map


# -----------------------
# Summarize branches
# -----------------------
def summarize_branches(
    branch_df,
    px_per_cm: float = 312,
    sample_ids: Optional[Iterable[str]] = None,
    by_system: bool = False,
    by_order: bool = False,
    min_children: Optional[int] = None,
    min_length: Optional[float] = None,
    max_order: Optional[int] = None,
    flipped_order: bool = True,
    return_diagnostics: bool = False,
):
    """
    Summarize branch statistics from a process_json-like DataFrame.

    Expects columns (at minimum):
      ['branch_id','branch_length','branch_width','n_child','seg_ids',
       'sample_id','system_id','order_before_flip','order_after_flip']

    Returns: (summary_df, branching_ratio_df[, diag_df])
    """
    try:
        import pandas as pd
        import numpy as np
    except Exception as e:
        raise RuntimeError("summarize_branches requires pandas and numpy") from e

    df = branch_df.copy()
    order_col = 'order_after_flip' if flipped_order else 'order_before_flip'
    df['order'] = df[order_col]

    if sample_ids is not None:
        df = df[df['sample_id'].isin(sample_ids)]

    # unit conversion
    df['length_cm'] = df['branch_length'] / px_per_cm
    df['width_cm']  = df['branch_width']  / px_per_cm
    df['w_x_l']     = df['width_cm'] * df['length_cm']
    df["volume_cm3"] = np.pi * (df["width_cm"] / 2.0) ** 2 * df["length_cm"]

    if return_diagnostics:
        diag_before = (
            df.groupby(['sample_id','system_id'], dropna=False)
              .agg(n_rows=('branch_id','size'),
                   n_orders=('order', lambda s: s.dropna().nunique()))
              .reset_index()
              .rename(columns={'n_rows':'rows_before','n_orders':'orders_before'})
        )

    if min_children is not None:
        df = df[df['n_child'] >= min_children]
    if min_length is not None:
        df = df[df['length_cm'] >= min_length]

    if max_order is not None:
        mask = df['order'].notna()
        df.loc[mask, 'order'] = df.loc[mask, 'order'].astype(int).clip(upper=max_order)

    if by_order:
        df = df[df['order'].notna()]

    group_cols = ['sample_id']
    if by_system:
        group_cols.append('system_id')
    if by_order:
        group_cols.append('order')

    g = df.groupby(group_cols, dropna=False)

    total_length = g['length_cm'].sum().rename('total_length_cm')
    num = g['w_x_l'].sum()
    den = g['length_cm'].sum()
    avg_width = (num / den).rename('avg_width_cm')
    branching_intensity = (g['n_child'].sum() / den).rename('branching_intensity')
    total_vol_cm3_branch = g['volume_cm3_branch'].sum().rename('total_vol_cm3_branch')

    
    if by_order:
        length_by_order = g['length_cm'].sum()
        base_cols = [c for c in group_cols if c != 'order']
        total_by_group = length_by_order.groupby(level=base_cols).transform('sum')
        length_prop = (length_by_order / total_by_group).rename('length_proportion')
        summary_df = pd.concat(
            [total_length, avg_width, branching_intensity, length_prop, total_vol_cm3_branch],axis=1).reset_index()

    else:
        summary_df = pd.concat(
            [total_length, avg_width, branching_intensity, total_vol_cm3_branch],axis=1).reset_index()

    if by_order:
        order_counts = g.size().rename('count').reset_index()
        base_cols = [c for c in group_cols if c != 'order']
        br_rows = []
        for keys, sub in order_counts.groupby(base_cols, dropna=False):
            sub = sub.sort_values('order')
            orders = sub['order'].tolist()
            counts = sub['count'].tolist()
            for i in range(len(orders) - 1):
                n, n1 = counts[i], counts[i+1]
                ratio = n / n1 if n1 != 0 else np.nan
                keyrow = dict(zip(base_cols, keys if isinstance(keys, tuple) else (keys,)))
                br_rows.append({
                    **keyrow,
                    'order_n': orders[i], 'count_n': n,
                    'order_n1': orders[i+1], 'count_n1': n1,
                    'branching_ratio': ratio
                })
        branching_ratio_df = pd.DataFrame(br_rows)
    else:
        branching_ratio_df = pd.DataFrame()

    if return_diagnostics:
        diag_after = (
            df.groupby(['sample_id','system_id'], dropna=False)
              .agg(n_rows=('branch_id','size'),
                   n_orders=('order', 'nunique'))
              .reset_index()
              .rename(columns={'n_rows':'rows_after','n_orders':'orders_after'})
        )
        diag_df = (diag_before.merge(diag_after, on=['sample_id','system_id'], how='left')
                             .fillna({'rows_after':0,'orders_after':0}))
        return summary_df, branching_ratio_df, diag_df

    return summary_df, branching_ratio_df

# --- Build system of connected segments (primarily helper for build_DFS_tree)
# Also useful
import networkx as nx
from typing import Tuple, Dict, Any, List, Set

# --- in utils.py ---
from typing import Tuple, Dict, Any, List, Set, Union
import networkx as nx

from typing import Dict, List, Tuple, Set, Any, Union
import networkx as nx

def extract_root_systems(
    json_data: dict,
    sample_id: Union[str, List[str], Tuple[str, ...], Set[str]],
) -> Dict[str, Dict[str, Any]]:
    """
    Extract segment, link, and connected-system data for one or more samples.

    Always returns:
        {
          "<sample_id>": {
             "segment_map": dict[str, dict],     # seg_id -> {"points": Nx2, "meta": {...}}
             "links": list[dict],                # raw link objects
             "components": list[set[str]],       # connected components of seg_ids
          },
          ...
        }

    Notes:
    - Works for a single sample_id or a collection; single sample returns a 1-key dict.
    - segment_map keys (sample_id, seg_id) are strings.
    """
    # Normalize input to a list of str sample IDs
    if isinstance(sample_id, (list, tuple, set)):
        sids = [str(s) for s in sample_id]
    else:
        sids = [str(sample_id)]

    results: Dict[str, Dict[str, Any]] = {}

    for sid in sids:
        sample = next(
            (img for img in json_data["images"] if img["info"].get("experiment") == sid),
            None
        )
        if sample is None:
            raise ValueError(f"Sample '{sid}' not found in JSON.")

        # Flatten segments & links
        segment_map = {
            str(seg["name"]): seg
            for root in sample["roots"]
            for seg in root["segments"]
        }
        links = [link for root in sample["roots"] for link in root["links"]]

        # Connectivity → systems
        G = nx.Graph()
        G.add_nodes_from(segment_map.keys())
        G.add_edges_from((str(l["seg1"]), str(l["seg2"])) for l in links)
        components = sorted(nx.connected_components(G), key=len, reverse=True)

        results[sid] = {
            "segment_map": segment_map,
            "links": links,
            "components": components,
        }

    return results


from typing import Any, Dict, Iterable, Optional, Union

def extract_segment_maps(
    json_data: dict,
    sample_id: Optional[Union[str, Iterable[str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Return a canonical {sample_id -> segment_map} mapping.

    - sample_id=None  -> extract ALL samples from json_data["images"].
    - sample_id=str   -> extract just that sample.
    - sample_id=iter  -> extract that list/tuple/set of samples.

    The segment_map for each sample is a dict[str, dict] with:
      { seg_id: { "points": Nx2 list, "meta": {...} } }.
    """
    # Build the list of sample IDs to request from extract_root_systems
    if sample_id is None:
        sids = [
            str(img["info"]["experiment"])
            for img in json_data.get("images", [])
            if isinstance(img, dict)
            and isinstance(img.get("info"), dict)
            and img["info"].get("experiment") is not None
        ]
    elif isinstance(sample_id, (list, tuple, set)):
        sids = [str(s) for s in sample_id]
    else:
        sids = [str(sample_id)]

    # Always returns the rich bundle {sid: {"segment_map", "links", "components"}}
    bundle = extract_root_systems(json_data, sids)

    # Project just the segment maps
    return {sid: bundle[sid]["segment_map"] for sid in bundle}

def save_taproots(results, path):
    """
    Save taproot annotation results (from annotate_taproot_polyline_tk_viz_v2)
    to disk in a consistent JSON format.

    Parameters
    ----------
    results : dict
        Keyed dictionary of annotations, e.g.
        {
            "C7C": {"sample_id": "C7C", "polyline": [...], "seg_ids": [...], "meta": {...}},
            "M6C": {...}
        }

    path : str or Path
        Output path for the JSON file, e.g. "Data/Morphology/taproot_gt.json"
    """

    # ensure JSON-friendly format
    entries = {"entries": list(results.values())}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    print(f"✅ Saved {len(results)} taproot annotations to {path}")


def load_taproots(path: str | Path) -> dict[str, dict]:
    """Load taproot_gt.json (list-of-dicts format) into keyed dict."""
    data = json.load(open(path, "r", encoding="utf-8"))
    entries = data.get("entries", [])
    return {e["sample_id"]: e for e in entries}


def merge_taproots(existing_path, new_results):
    """
    Merge new taproot annotations (keyed dict) into an existing JSON file.

    - existing_path: str or Path to 'taproot_gt.json'
    - new_results: dict[sample_id -> annotation dict]
    """
    import json
    from pathlib import Path

    path = Path(existing_path)

    # --- Load existing entries ---
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            entries = existing.get("entries", [])
        except Exception:
            entries = []
    else:
        entries = []

    # --- Convert to keyed dict for easy merging ---
    merged = {e["sample_id"]: e for e in entries}

    # --- Update with new results (overwrites duplicates) ---
    for sid, entry in new_results.items():
        merged[sid] = entry

    # --- Save back as list of entries ---
    out = {"entries": list(merged.values())}
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"✅ Merged {len(new_results)} new annotations into {path.name}. "
          f"Now contains {len(merged)} total samples.")


def merge_taproots_multi(input_dir_or_list, output_path: str | None = None):
    """
    Merge multiple taproot annotation sources (dicts or JSON files) into one canonical structure.

    Parameters
    ----------
    input_dir_or_list : str | Path | list
        - If a directory, scans for all '*.json' files and merges them.
        - If a list, may contain dict objects (already-loaded taproot data)
          and/or paths to JSON files.
    output_path : str | Path | None, optional
        If provided, saves the merged result to this file in {"entries": [...]} format.
        If None, the function just returns the merged dictionary.

    Returns
    -------
    dict
        Dictionary keyed by sample_id with taproot entries.
    """
    import json
    from pathlib import Path
    from collections.abc import Mapping

    # --- Normalize inputs ---
    inputs = []
    if isinstance(input_dir_or_list, (str, Path)):
        p = Path(input_dir_or_list)
        if p.is_dir():
            inputs = sorted(p.glob("*.json"))
        elif p.is_file():
            inputs = [p]
        else:
            raise FileNotFoundError(f"No file or directory found at {p}")
    elif isinstance(input_dir_or_list, list):
        inputs = input_dir_or_list
    else:
        raise TypeError("input_dir_or_list must be a path or list")

    merged: dict[str, dict] = {}

    # --- Load and unify ---
    for src in inputs:
        if isinstance(src, Mapping):  # already-loaded dict
            data = src
        else:
            p = Path(src)
            if not p.exists():
                print(f"⚠️  Skipping missing file: {p}")
                continue
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Handle both formats: {"entries": [...]} or flat dict {sample_id: {...}}
        if isinstance(data, dict) and "entries" in data:
            entries = data["entries"]
        elif isinstance(data, dict):
            entries = list(data.values())
        elif isinstance(data, list):
            entries = data
        else:
            print(f"⚠️  Unrecognized format in {src}; skipping.")
            continue

        for e in entries:
            sid = e.get("sample_id")
            if sid is None:
                continue
            merged[sid] = e  # overwrite duplicates

    # --- Save only if output_path specified ---
    if output_path is not None:
        out_path = Path(output_path)
        out_data = {"entries": list(merged.values())}
        out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"✅ Merged {len(merged)} total samples and saved to {out_path.name}")
    else:
        print(f"✅ Merged {len(merged)} total samples (not saved)")

    return merged

