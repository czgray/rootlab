# Copyright 2025 Christian Gray
#
# File: diagnostics.py
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

import numpy as np
import pandas as pd
from typing import Mapping, Any, Optional, Iterable

def _normalize_seg_ids_column(df: pd.DataFrame, col: str = "seg_ids") -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        out[col] = [[] for _ in range(len(out))]
        return out
    normed = []
    for v in out[col].tolist():
        if isinstance(v, list):
            normed.append([str(x).strip() for x in v])
        elif isinstance(v, tuple):
            normed.append([str(x).strip() for x in v])
        elif isinstance(v, np.ndarray):
            normed.append([str(x).strip() for x in v.tolist()])
        elif isinstance(v, str):
            normed.append([t.strip() for t in v.split(",") if t.strip()])
        elif pd.isna(v):
            normed.append([])
        else:
            normed.append([str(v).strip()])
    out[col] = normed
    return out

def _resolve_order_col(mode: str = "bio", use_flipped_order: bool = True, order_col: Optional[str] = None) -> str:
    if order_col:
        return order_col
    if mode.lower() == "dfs":
        return "dfs_order"
    return "order_after_flip" if use_flipped_order else "order_before_flip"

def diagnose_plot_by_order_gaps(
    *,
    branch_table_df: pd.DataFrame,
    segment_maps: Mapping[str, Mapping[str, Any]],
    sample_id: str,
    system_id: Optional[str | int] = None,
    mode: str = "bio",
    use_flipped_order: bool = True,
    order_col: Optional[str] = None,
    background_from: str = "segment_map",  # "segment_map" or "branch_df_all"
) -> dict:
    """Return ids the order-plot would color vs those present in the background."""
    sid = str(sample_id)
    if sid not in segment_maps:
        raise KeyError(f"Sample '{sid}' not in segment_maps.")
    segmap = segment_maps[sid]

    # background universe (what highlight shows)
    if background_from == "segment_map":
        background_ids = set(map(str, segmap.keys()))
    elif background_from == "branch_df_all":
        df_all = _normalize_seg_ids_column(branch_table_df[branch_table_df["sample_id"] == sid], "seg_ids")
        background_ids = set(s for lst in df_all["seg_ids"] for s in lst)
    else:
        raise ValueError("background_from must be 'segment_map' or 'branch_df_all'")

    # what order-plot actually uses
    df = _normalize_seg_ids_column(branch_table_df[branch_table_df["sample_id"] == sid], "seg_ids")
    if system_id is not None and "system_id" in df.columns:
        df = df[df["system_id"] == system_id]

    ord_col = _resolve_order_col(mode, use_flipped_order, order_col)
    rows = []
    for _, r in df.iterrows():
        ids = r.get("seg_ids", [])
        val = r.get(ord_col, np.nan)
        for s in ids:
            rows.append((str(s), val))
    used = pd.DataFrame(rows, columns=["seg_id", "order_val"]) if rows else pd.DataFrame(columns=["seg_id","order_val"])
    plotted_ids = set(used["seg_id"].tolist())

    missing_ids = sorted(background_ids - plotted_ids)

    return {
        "sample_id": sid,
        "system_id": system_id,
        "order_col": ord_col,
        "background_ids": background_ids,
        "plotted_ids": plotted_ids,
        "missing_ids": missing_ids,   # ← feed this to step (2)
    }

import numpy as np
import pandas as pd
import networkx as nx
from typing import Mapping, Any, Optional, Iterable, Dict, List

def _seg_length(pts) -> float:
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        return 0.0
    d = np.diff(pts, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))

def diagnose_missing_segments_for_ids(
    *,
    sample_id: str,
    segment_maps: Mapping[str, Mapping[str, Dict[str, Any]]],
    missing_ids: Iterable[str],
    links: Optional[List[Dict[str, Any]]] = None,
    min_length_flag: Optional[float] = None,  # set if your builder uses a length cutoff; else None
) -> pd.DataFrame:
    """
    For the given missing segment IDs, report geometry + graph diagnostics:
    length, width, degree, component size, and simple flags.
    """
    sid = str(sample_id)
    segmap = segment_maps[sid]
    mids = [str(x) for x in missing_ids]

    # Build graph if links given
    comp_size = {}
    degree = {}
    if links is not None:
        G = nx.Graph()
        G.add_nodes_from(segmap.keys())
        for l in links:
            a, b = str(l["seg1"]), str(l["seg2"])
            if a in segmap and b in segmap:
                G.add_edge(a, b)
        for comp in nx.connected_components(G):
            size = len(comp)
            for n in comp:
                comp_size[n] = size
        degree = dict(G.degree())
    else:
        comp_size = {s: np.nan for s in segmap.keys()}
        degree   = {s: np.nan for s in segmap.keys()}

    rows = []
    for s in mids:
        seg = segmap.get(s, {})
        pts = seg.get("points", [])
        length = _seg_length(pts)
        meta = seg.get("meta", {}) if isinstance(seg.get("meta"), dict) else {}
        width = max(float(meta.get("width_right", 0.0)) - float(meta.get("width_left", 0.0)), 0.0)

        rows.append({
            "seg_id": s,
            "len_px": length,
            "width_units": width,
            "graph_degree": degree.get(s, np.nan),
            "component_size": comp_size.get(s, np.nan),
            "flag_tiny_len": (min_length_flag is not None and length < min_length_flag),
            "flag_isolated": (degree.get(s, 0) == 0) if not np.isnan(degree.get(s, np.nan)) else np.nan,
        })

    return pd.DataFrame(rows).sort_values(["flag_isolated", "flag_tiny_len", "len_px"], ascending=[False, True, True])

def summarize_missing_reasons(diag_df: pd.DataFrame):
    total = len(diag_df)
    if total == 0:
        print("No missing segments. ✅")
        return
    iso = int((diag_df["graph_degree"] == 0).sum(skipna=True))
    tiny = int(diag_df.get("flag_tiny_len", pd.Series(False, index=diag_df.index)).sum())
    small_comp = int((diag_df["component_size"].fillna(0) <= 2).sum())
    zero_w = int((diag_df["width_units"] <= 0).sum())
    print(f"Total missing: {total}")
    print(f"  • isolated (degree=0):        {iso} ({iso/total:.1%})")
    print(f"  • below min_length (flag):    {tiny} ({tiny/total:.1%})")
    print(f"  • tiny components (<=2):      {small_comp} ({small_comp/total:.1%})")
    print(f"  • zero/neg width:             {zero_w} ({zero_w/total:.1%})")
