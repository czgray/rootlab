"""Visualization helpers (plot*-style API)

Changes in this version:
- Renamed public entry points to use `plot_*` (not `visualize_*`).
- Added a shared `_render_core(...)` for identical draw/axes/IO.
- Decoupled on-screen DPI from save DPI (screen-friendly `figure_dpi`, high `save_dpi`).
- Kept backward-compat aliases for old names, but prefer the new `plot_*` API.
- Legend label for taproot candidates now supports per-system labels (S{sys_idx} #j).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Mapping, Any, Optional
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from typing import Optional

# -----------------------------------------------------------------------------
# Module-level defaults
# -----------------------------------------------------------------------------

@dataclass
class _VizConfig:
    # Layout & DPI
    figsize: tuple[float, float] = (9, 7)
    figure_dpi: int = 110     # DPI for interactive figures (plt.show)
    save_dpi: int = 450       # DPI for saved outputs (savefig)

    # Stroke/width policy
    min_linewidth: float = 1.0
    width_scale: float = 0.25  # global multiplier applied to (width_right - width_left)
    width_max: float = 7.0    # clamp ceiling after scaling

    # Orientation
    invert_y_default: bool = True

    # Colors / palettes
    base_gray: str = "#BDBDBD"
    base_alpha: float = 0.7
    cmap_order: str = "viridis"
    cmap_system: str = "tab10"
    emph_color: str = "#7B1818"

    # Taproot overlay
    taproot_alpha: float = 0.85
    linewidth_poly: float = 3.0  # candidate polyline stroke (not a true segment)

_VIZ = _VizConfig()


def set_viz_defaults(**kwargs):
    """Update visualization defaults module-wide.

    Example:
        set_viz_defaults(width_scale=6.0, figsize=(9, 9))
    """
    for k, v in kwargs.items():
        if hasattr(_VIZ, k):
            setattr(_VIZ, k, v)
        else:
            warnings.warn(f"set_viz_defaults: unknown option '{k}' ignored.")


def data_width_to_points(ax, data_width: float) -> float:
    """
    Convert a width in data coordinates (e.g., pixels in the source image)
    to a linewidth in screen points for Matplotlib plotting.

    This ensures that if you draw a line with lw=data_width_to_points(ax, X),
    the visible stroke width on screen corresponds to X data units
    in the current axes transform — i.e., pixel-accurate display.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the line will be drawn.
    data_width : float
        Desired width in data units (e.g., pixels).

    Returns
    -------
    float
        Linewidth in points suitable for Matplotlib `plot()` or `LineCollection`.
    """
    import numpy as np
    fig = ax.figure
    trans = ax.transData.transform
    # Transform 0→data_width segment to display coordinates
    x0, y0 = trans((0, 0))
    x1, y1 = trans((data_width, 0))
    display_dist_px = np.hypot(x1 - x0, y1 - y0)
    # Convert display pixels → points (1 pt = 1/72 inch)
    return display_dist_px * 72.0 / fig.dpi


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

_REQUIRED_BASE = {"sample_id", "system_id", "seg_ids"}

def _normalize_seg_ids_column(df, col: str = "seg_ids"):
    """Ensure df[col] is list[str] for every row; returns a copy with normalized column."""
    if col not in df.columns:
        return df
    df = df.copy()
    normed = []
    for val in df[col].tolist():
        if isinstance(val, str):
            parts = [s.strip() for s in val.split(",") if s.strip()]
            normed.append(parts)
        elif isinstance(val, (list, tuple, np.ndarray)):
            normed.append([str(x) for x in val])
        elif val is None or (isinstance(val, float) and np.isnan(val)):
            normed.append([])
        else:
            normed.append([str(val)])
    df[col] = normed
    return df


def _order_column(use_flipped_order: bool = True, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    return "order_after_flip" if use_flipped_order else "order_before_flip"


def _width_from_meta(meta: Mapping[str, Any]) -> float:
    wl = float(meta.get("width_left", 0) or 0)
    wr = float(meta.get("width_right", 0) or 0)
    w = max(wr - wl, 0.0)
    lw = w * _VIZ.width_scale
    # clamp final linewidth
    return max(_VIZ.min_linewidth, min(lw, _VIZ.width_max))


def _get_cmap(name: str, N: Optional[int] = None):
    return cm.get_cmap(name, N) if N else cm.get_cmap(name)


def _iter_samples(sample_id, segment_maps):
    """Yield sample ids to process based on None/str/list input."""
    if sample_id is None:
        for sid in segment_maps.keys():
            yield sid
    elif isinstance(sample_id, (list, tuple, np.ndarray)):
        for sid in sample_id:
            yield sid
    else:
        yield sample_id


def _points_from_segment(seg: Mapping[str, Any], invert_y: bool) -> Optional[np.ndarray]:
    pts = np.asarray(seg.get("points", []), dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        return None
    if invert_y:
        pts = pts.copy()
        pts[:, 1] = -pts[:, 1]
    return pts


def _collect_segment_drawables(segmap: Mapping[str, Any],
                               seg_ids: Iterable[str],
                               invert_y: bool) -> tuple[list[np.ndarray], list[float]]:
    seg_arrays: list[np.ndarray] = []
    widths: list[float] = []
    for sid in seg_ids:
        seg = segmap.get(str(sid))
        if not seg:
            continue
        pts = _points_from_segment(seg, invert_y)
        if pts is None:
            continue
        seg_arrays.append(pts)
        widths.append(_width_from_meta(seg.get("meta", {})))
    return seg_arrays, widths


def _auto_limits(ax, segments: list[np.ndarray]):
    if not segments:
        return
    all_xy = np.concatenate(segments, axis=0)
    xmin, ymin = np.min(all_xy, axis=0)
    xmax, ymax = np.max(all_xy, axis=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def _seg_attr_map(branch_df, attr_col: str) -> dict[str, Any]:
    """Map each segment id to an attribute value from its owning branch."""
    m: dict[str, Any] = {}
    for _, row in branch_df.iterrows():
        seg_ids = row["seg_ids"]
        val = row.get(attr_col)
        for sid in seg_ids:
            m[str(sid)] = val
    return m


def _colors_from_attr_values(values: list[Any], cmap_name: str,
                             discrete_keys: Optional[list[Any]] = None) -> list:
    """Return list of RGBA colors. If `discrete_keys` provided, use indexed palette; else continuous."""
    if discrete_keys is not None:
        keys = list(discrete_keys)
        cmap = _get_cmap(cmap_name, max(1, len(keys)))
        idx_map = {k: i for i, k in enumerate(keys)}
        return [cmap(idx_map.get(v, 0)) for v in values]
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return [_get_cmap(cmap_name)(0.5) for _ in values]
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if vmax - vmin <= 1e-12:
        normed = np.full_like(arr, 0.5)
    else:
        normed = (arr - vmin) / (vmax - vmin)
    cmap = _get_cmap(cmap_name)
    return [cmap(float(t)) for t in normed]


# -----------------------------------------------------------------------------
# Rendering core (shared final draw/axes/IO plumbing)
# -----------------------------------------------------------------------------

def _render_core(
    *,
    segments: list[np.ndarray],
    widths: list[float],
    colors: list,
    title: str,
    figsize: Optional[tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save_path: Optional[str] = None,
    base_segments: Optional[list[np.ndarray]] = None,
    base_widths: Optional[list[float]] = None,
    base_alpha: Optional[float] = None,
):
    """Draw a set of segments with per-segment widths/colors and handle IO.

    Optionally draws a gray base layer (e.g., for background systems).
    """
    # Create with screen-friendly DPI
    fig, ax = plt.subplots(figsize=figsize or _VIZ.figsize, dpi=dpi or _VIZ.figure_dpi)

    # Optional gray background layer
    if base_segments:
        lc_bg = LineCollection(
            base_segments,
            colors=[_VIZ.base_gray] * len(base_segments),
            linewidths=base_widths,
            alpha=_VIZ.base_alpha if base_alpha is None else base_alpha,
            zorder=1,
        )
        ax.add_collection(lc_bg)

    lc = LineCollection(segments, colors=colors, linewidths=widths, zorder=2)
    ax.add_collection(lc)

    all_for_limits = (base_segments or []) + segments
    _auto_limits(ax, all_for_limits)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=_VIZ.save_dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# -----------------------------------------------------------------------------
# Public API — plot by order (bio/dfs) and by system, plus convenience wrapper
# -----------------------------------------------------------------------------

def plot_sample_by_order(
    branch_table_df,
    segment_maps: Mapping[str, Mapping[str, Any]],
    sample_id: Optional[Iterable[str] | str] = None,
    system_id: Optional[str | int] = None,
    *,
    mode: str = "bio",                  # 'bio' or 'dfs'
    use_flipped_order: bool = True,
    order_col: Optional[str] = None,     # overrides use_flipped_order when provided
    figsize: Optional[tuple[float, float]] = None,
    dpi: Optional[int] = None,
    invert_y: Optional[bool] = None,
    title: Optional[str] = None,
    json_name: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Plot sample(s) colored by order (biological or DFS)."""
    if mode.lower() == "dfs":
        resolved_order_col = "dfs_order"
    else:
        resolved_order_col = _order_column(use_flipped_order, order_col)

    required = set(_REQUIRED_BASE)
    required.add(resolved_order_col)
    _validate_required_columns(branch_table_df, required)

    branch_table_df = _normalize_seg_ids_column(branch_table_df, col="seg_ids")

    for sid in _iter_samples(sample_id, segment_maps):
        segmap = segment_maps.get(sid)
        if not segmap:
            warnings.warn(f"plot_sample_by_order: sample '{sid}' not found in segment_maps; skipped.")
            continue

        df = branch_table_df[branch_table_df["sample_id"] == sid].copy()
        if system_id is not None:
            df = df[df["system_id"] == system_id]
        if df.empty:
            warnings.warn(f"plot_sample_by_order: no branches for sample '{sid}' (system_id={system_id}).")
            continue

        seg_to_order = _seg_attr_map(df, resolved_order_col)
        all_seg_ids = list(seg_to_order.keys())

        inv = _VIZ.invert_y_default if invert_y is None else invert_y
        segments, widths = _collect_segment_drawables(segmap, all_seg_ids, invert_y=inv)

        order_vals = [seg_to_order.get(str_id) for str_id in all_seg_ids]
        colors = _colors_from_attr_values(order_vals, _VIZ.cmap_order)

        # Resolve output path (directory-aware)
        out_path = save_path
        if save_path and os.path.isdir(save_path):
            fname = f"{json_name + '__' if json_name else ''}{sid}__system-{system_id if system_id is not None else 'all'}__mode-order-{mode}.png"
            out_path = os.path.join(save_path, fname)

        _render_core(
            segments=segments,
            widths=widths,
            colors=colors,
            title=(title if title is not None else str(sid)),
            figsize=figsize,
            dpi=dpi,
            save_path=out_path,
        )


def plot_sample_by_system(
    branch_table_df,
    segment_maps: Mapping[str, Mapping[str, Any]],
    sample_id: Optional[Iterable[str] | str] = None,
    system_id: Optional[str | int] = None,
    *,
    figsize: Optional[tuple[float, float]] = None,
    dpi: Optional[int] = None,
    invert_y: Optional[bool] = None,
    title: Optional[str] = None,
    json_name: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Plot sample(s) colored by system (tab10)."""
    required = set(_REQUIRED_BASE)
    _validate_required_columns(branch_table_df, required)

    branch_table_df = _normalize_seg_ids_column(branch_table_df, col="seg_ids")

    for sid in _iter_samples(sample_id, segment_maps):
        segmap = segment_maps.get(sid)
        if not segmap:
            warnings.warn(f"plot_sample_by_system: sample '{sid}' not found in segment_maps; skipped.")
            continue

        df = branch_table_df[branch_table_df["sample_id"] == sid].copy()
        if df.empty:
            warnings.warn(f"plot_sample_by_system: no branches for sample '{sid}'.")
            continue
        if system_id is not None:
            df = df[df["system_id"] == system_id]
            if df.empty:
                warnings.warn(f"plot_sample_by_system: no branches for sample '{sid}' with system_id={system_id}.")
                continue

        seg_to_sys = _seg_attr_map(df, "system_id")
        all_seg_ids = list(seg_to_sys.keys())
        unique_systems = sorted(set(seg_to_sys.values()), key=lambda x: str(x))

        inv = _VIZ.invert_y_default if invert_y is None else invert_y
        segments, widths = _collect_segment_drawables(segmap, all_seg_ids, invert_y=inv)
        sys_vals = [seg_to_sys.get(str_id) for str_id in all_seg_ids]
        colors = _colors_from_attr_values(sys_vals, _VIZ.cmap_system, discrete_keys=unique_systems)

        out_path = save_path
        if save_path and os.path.isdir(save_path):
            fname = f"{json_name + '__' if json_name else ''}{sid}__system-{system_id if system_id is not None else 'all'}__mode-system.png"
            out_path = os.path.join(save_path, fname)

        _render_core(
            segments=segments,
            widths=widths,
            colors=colors,
            title=(title if title is not None else str(sid)),
            figsize=figsize,
            dpi=dpi,
            save_path=out_path,
        )


def plot_system(
    branch_table_df,
    segment_maps: Mapping[str, Mapping[str, Any]],
    *,
    sample_id: str,
    system_id: str | int,
    color_mode: str = "order",   # 'order' or 'system'
    **kwargs,
):
    if color_mode == "system":
        return plot_sample_by_system(branch_table_df, segment_maps, sample_id=sample_id, system_id=system_id, **kwargs)
    return plot_sample_by_order(branch_table_df, segment_maps, sample_id=sample_id, system_id=system_id, **kwargs)


# -----------------------------------------------------------------------------
# Highlighting — generic function to overlay arbitrary segment IDs
# -----------------------------------------------------------------------------

def plot_highlight_segments(
    branch_table_df: Optional[Any] = None,          # ← now optional
    segment_maps: Mapping[str, Mapping[str, Any]] = None,
    sample_id: str | Iterable[str] = None,
    highlight_seg_ids: Iterable[str | int] = (),
    system_id: Optional[str | int] = None,
    *,
    highlight_color: Optional[Any] = None,
    per_id_colors: Optional[Iterable[Any]] = None,
    figsize: Optional[tuple[float, float]] = None,
    dpi: Optional[int] = None,
    invert_y: Optional[bool] = None,
    title: Optional[str] = None,
    json_name: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot and highlight specific segment IDs on top of a gray background.

    - `sample_id` may be a single ID or an iterable of IDs. One figure per sample.
    - If `system_id` is provided, the gray background is restricted to that system.
    - `highlight_seg_ids` are overlaid with true per-segment widths. Missing IDs warn but do not stop.
    - Colors: if `per_id_colors` is provided (same length as found highlights), use it; otherwise
      use `highlight_color` for all (default None ⇒ cycle through tab10 per highlighted segment).
    - Uses shared helpers for consistency and performance.
    - `branch_table_df` is OPTIONAL. It is only required if you pass `system_id`
      (to restrict the gray background to that system). If `system_id` is given
      while `branch_table_df` is None, this function raises a ValueError.
    """
    
    # Normalize highlight IDs to strings once
    raw_ids = list(highlight_seg_ids)
    requested_ids = [str(x) for x in raw_ids]

    # If a system filter is requested, we need the branch table
    if system_id is not None and branch_table_df is None:
        raise ValueError(
            "plot_highlight_segments: system_id was provided but branch_table_df=None. "
            "Provide branch_table_df (or omit system_id)."
        )

    # When restricting background to a system, we need the seg_ids from the table
    if system_id is not None:
        _validate_required_columns(branch_table_df, _REQUIRED_BASE)

    for sid in _iter_samples(sample_id, segment_maps):
        segmap = segment_maps.get(sid)
        if not segmap:
            warnings.warn(f"plot_highlight_segments: sample '{sid}' not found in segment_maps; skipped.")
            continue

        # Background selection
        if system_id is None or branch_table_df is None:
            # Simple mode: whole-sample background (works without branch_table_df)
            background_seg_ids = list(segmap.keys())
        else:
            # System-filtered background (requires branch_table_df)
            df = branch_table_df[
                (branch_table_df["sample_id"] == sid) & (branch_table_df["system_id"] == system_id)
            ]
            if df.empty:
                warnings.warn(
                    f"plot_highlight_segments: no branches for sample '{sid}' with system_id={system_id}; "
                    "background will be empty."
                )
                background_seg_ids = []
            else:
                df = _normalize_seg_ids_column(df, col="seg_ids")
                acc = []
                for lst in df["seg_ids"].tolist():
                    acc.extend(lst)
                background_seg_ids = list(dict.fromkeys(str(x) for x in acc))

        # Which of the requested highlight IDs exist in this sample?
        missing = [h for h in requested_ids if h not in segmap]
        present_ids = [h for h in requested_ids if h in segmap]
        if missing:
            preview = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
            warnings.warn(
                f"plot_highlight_segments: {len(missing)}/{len(requested_ids)} highlight id(s) not found in sample '{sid}': {preview}"
            )

        inv = _VIZ.invert_y_default if invert_y is None else invert_y

        # Collect drawables
        bg_segments, bg_widths = _collect_segment_drawables(segmap, background_seg_ids, invert_y=inv)
        hi_segments, hi_widths = _collect_segment_drawables(segmap, present_ids, invert_y=inv)

        # Build highlight colors
        if per_id_colors is not None:
            colors = list(per_id_colors)
            if len(colors) != len(hi_segments):
                warnings.warn(
                    f"plot_highlight_segments: per_id_colors length ({len(colors)}) does not match number of found highlights ({len(hi_segments)}); truncating/padding as needed."
                )
            # pad or truncate to match
            if len(colors) < len(hi_segments):
                colors = colors + [colors[-1]] * (len(hi_segments) - len(colors)) if colors else ["C0"] * len(hi_segments)
            else:
                colors = colors[: len(hi_segments)]
        elif highlight_color is not None:
            colors = [highlight_color] * len(hi_segments)
        else:
            # cycle tab10 to distinguish when multiple highlights are given
            cmap = _get_cmap("tab10", max(1, len(hi_segments)))
            colors = [cmap(i) for i in range(len(hi_segments))]

        # Resolve output path (directory-aware)
        out_path = save_path
        if save_path and os.path.isdir(save_path):
            sys_tag = system_id if system_id is not None else 'all'
            fname = f"{json_name + '__' if json_name else ''}{sid}__system-{sys_tag}__mode-highlight.png"
            out_path = os.path.join(save_path, fname)

        # Render via shared core (background in gray, overlays colored)
        _render_core(
            segments=hi_segments,
            widths=hi_widths,
            colors=colors,
            title=(title if title is not None else str(sid)),
            figsize=figsize,
            dpi=dpi,
            save_path=out_path,
            base_segments=bg_segments,
            base_widths=bg_widths,
            base_alpha=_VIZ.base_alpha,
        )

# -----------------------------------------------------------------------------
# Taproot candidates (renamed to plot_taproot_candidates)
# -----------------------------------------------------------------------------

def plot_taproot_candidates(
    json_file: str | os.PathLike,
    sample_id: str | Iterable[str] | None,
    taproot_results: Mapping[tuple[str, int], Any],
    system_index: Optional[int] = None,   # None => whole-sample per selected SID(s)
    *,
    topk: int = 1,
    n_seg_min: int = 0,
    show_legend: bool = True,
    title: Optional[str] = None,
    invert_y: Optional[bool] = None,
    json_name: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    dpi: Optional[int] = None,
    taproot_alpha: Optional[float] = None,
) -> None:
    """Plot taproot candidates with gray background and per-segment widths.

    sample_id may be a str, an iterable of str (one figure per sample), or None (all samples in JSON).
    Legend labels use "S{sys_idx} #j" in whole-sample mode; in single-system mode they use that system_index.
    """
    import json
    import networkx as nx

    taproot_alpha = _VIZ.taproot_alpha if taproot_alpha is None else float(taproot_alpha)
    inv = _VIZ.invert_y_default if invert_y is None else invert_y

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if sample_id is None:
        sample_ids = [img["info"]["experiment"] for img in data.get("images", [])]
    elif isinstance(sample_id, (list, tuple, set, np.ndarray)):
        sample_ids = list(sample_id)
    else:
        sample_ids = [str(sample_id)]

    def _resolve_out_path(base_path: Optional[str], sid: str, sys_tag: str) -> Optional[str]:
        if not base_path:
            return None
        if os.path.isdir(base_path):
            fname = f"{json_name + '__' if json_name else ''}{sid}__system-{sys_tag}__mode-taproot.png"
            return os.path.join(base_path, fname)
        root, ext = os.path.splitext(base_path)
        if len(sample_ids) > 1:
            return f"{root}__{sid}__system-{sys_tag}__mode-taproot{ext or '.png'}"
        return base_path

    for sid in sample_ids:
        try:
            img = next(img for img in data["images"] if img["info"]["experiment"] == sid)
        except StopIteration:
            warnings.warn(f"plot_taproot_candidates: sample '{sid}' not found in JSON; skipped.")
            continue

        segment_map = {str(seg["name"]): seg for root in img["roots"] for seg in root["segments"]}
        links = [lk for root in img["roots"] for lk in root.get("links", [])]

        G = nx.Graph()
        G.add_nodes_from(segment_map.keys())
        G.add_edges_from((str(lk["seg1"]), str(lk["seg2"])) for lk in links)
        systems = sorted(nx.connected_components(G), key=len, reverse=True)

        if system_index is None:
            background_segments = list(segment_map.keys())
        else:
            if not (0 <= int(system_index) < len(systems)):
                warnings.warn(
                    f"plot_taproot_candidates: system_index={system_index} out of range for sample '{sid}'. Skipping."
                )
                continue
            background_segments = list(systems[int(system_index)])

        # Background (gray, true widths, batched)
        bg_segments, bg_widths = _collect_segment_drawables(segment_map, background_segments, invert_y=inv)

        # Collect candidates for this sample
        candidates = []
        if system_index is None:
            for sys_idx, comp in enumerate(systems):
                if n_seg_min > 0 and len(comp) < n_seg_min:
                    continue
                key = (sid, sys_idx)
                if key not in taproot_results:
                    continue
                cands = list(taproot_results[key].candidates[: max(0, int(topk))])
                for j, c in enumerate(cands, start=1):
                    d = dict(c)
                    d["legend_label"] = f"S{sys_idx} #{j}"
                    candidates.append(d)
        else:
            key = (sid, int(system_index))
            if key in taproot_results:
                cands = list(taproot_results[key].candidates[: max(0, int(topk))])
                for j, c in enumerate(cands, start=1):
                    d = dict(c)
                    d["legend_label"] = f"S{int(system_index)} #{j}"
                    candidates.append(d)

        # Render per-sample figure
        fig, ax = plt.subplots(figsize=figsize or _VIZ.figsize, dpi=dpi or _VIZ.figure_dpi)

        if bg_segments:
            lc_bg = LineCollection(bg_segments, colors=[_VIZ.base_gray]*len(bg_segments), linewidths=bg_widths,
                                   alpha=_VIZ.base_alpha, zorder=1)
            ax.add_collection(lc_bg)

        cmap = _get_cmap("tab10", max(1, len(candidates)))
        handles, labels = [], []
        for i, cand in enumerate(candidates):
            color = cmap(i)
            mask_ids = [str(s) for s in cand.get("mask_seg_ids", [])]
            segs, lws = _collect_segment_drawables(segment_map, mask_ids, invert_y=inv)
            if segs:
                lc = LineCollection(segs, colors=[color for _ in segs], linewidths=lws, alpha=_VIZ.taproot_alpha, zorder=2)
                ax.add_collection(lc)
            P = np.asarray(cand.get("polyline", []), dtype=float)
            if P.ndim == 2 and P.shape[0] >= 2:
                if inv:
                    P = P.copy(); P[:, 1] = -P[:, 1]
                ax.plot(P[:, 0], P[:, 1], color=color, linewidth=_VIZ.linewidth_poly, alpha=_VIZ.taproot_alpha, zorder=3)
            handles.append(plt.Line2D([0], [0], color=color, lw=_VIZ.linewidth_poly))
            labels.append(cand.get("legend_label", f"#{i+1}"))

        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=8)

        _auto_limits(ax, bg_segments if bg_segments else [])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title if title is not None else str(sid))

        sys_tag = "all" if system_index is None else str(system_index)
        out_path = None
        if save_path:
            if os.path.isdir(save_path):
                fname = f"{json_name + '__' if json_name else ''}{sid}__system-{sys_tag}__mode-taproot.png"
                out_path = os.path.join(save_path, fname)
            else:
                root, ext = os.path.splitext(save_path)
                if len(sample_ids) > 1:
                    out_path = f"{root}__{sid}__system-{sys_tag}__mode-taproot{ext or '.png'}"
                else:
                    out_path = save_path
        if out_path:
            plt.savefig(out_path, dpi=_VIZ.save_dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


# -----------------------------------------------------------------------------
# Backward-compatibility aliases (prefer the new plot_* names)
# -----------------------------------------------------------------------------

def plot_root_system(
    branch_table_df,
    segment_maps,
    sample_id,
    system_id: Optional[str | int] = None,
    mode: str = "biological_order",     # 'biological_order' or 'dfs_order'
    use_flipped_order: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    m = mode.lower()
    if m in {"biological_order", "bio", "biological"}:
        return plot_sample_by_order(
            branch_table_df,
            segment_maps,
            sample_id=sample_id,
            system_id=system_id,
            mode="bio",
            use_flipped_order=use_flipped_order,
            title=title,
            save_path=save_path,
            **kwargs,
        )
    elif m in {"dfs_order", "dfs"}:
        return plot_sample_by_order(
            branch_table_df,
            segment_maps,
            sample_id=sample_id,
            system_id=system_id,
            mode="dfs",
            title=title,
            save_path=save_path,
            **kwargs,
        )
    else:
        raise ValueError("mode must be 'biological_order' or 'dfs_order'")


# Old names kept as thin wrappers (deprecated)

def visualize_root_system(*args, **kwargs):
    warnings.warn("visualize_root_system is deprecated; use plot_root_system instead.", DeprecationWarning)
    return plot_root_system(*args, **kwargs)


def visualize_taproot_candidates(*args, **kwargs):
    warnings.warn("visualize_taproot_candidates is deprecated; use plot_taproot_candidates instead.", DeprecationWarning)
    return plot_taproot_candidates(*args, **kwargs)


# -----------------------------------------------------------------------------
# Pixel-accurate export (left as-is for later refinement)
# -----------------------------------------------------------------------------

def export_pixel_accurate_image(branch_table_df, segment_maps, sample_id,
                                system_id=None, dpi=600, pixels_per_width_unit=1,
                                color_mode="bw", use_flipped_order=True,
                                save_path=None, invert_y=True):
    """Export or display a pixel-accurate visualization (kept for later work)."""
    if sample_id not in segment_maps:
        raise ValueError(f"Sample {sample_id} not found in segment_maps.")

    segment_map = segment_maps[sample_id]

    df_sample = branch_table_df[branch_table_df["sample_id"] == sample_id]
    if system_id:
        df_sample = df_sample[df_sample["system_id"] == system_id]

    if df_sample.empty:
        raise ValueError(f"No branches found for sample {sample_id} with system {system_id}.")

    df_sample = _normalize_seg_ids_column(df_sample, col="seg_ids")
    segment_names = set()
    for seg_list in df_sample["seg_ids"]:
        segment_names.update(seg_list)

    if not system_id:
        segment_names = set(segment_map.keys())

    order_col = _order_column(use_flipped_order)
    bio_orders = df_sample.set_index("branch_id")[order_col].to_dict()
    max_order = max(bio_orders.values()) if bio_orders else 1

    cmap = cm.viridis
    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=dpi)

    all_x, all_y = [], []
    for seg_name in segment_names:
        seg = segment_map.get(str(seg_name))
        if seg is None:
            continue
        points = np.array(seg.get("points", []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            continue
        all_x.extend(points[:, 0])
        all_y.extend(points[:, 1])
        meta = seg.get("meta", {})
        width_left = meta.get("width_left", 0)
        width_right = meta.get("width_right", 0)
        width = max(width_right - width_left, 0.1)
        if color_mode == "bio_order":
            branch_row = df_sample[df_sample["seg_ids"].apply(lambda L: str(seg_name) in L)]
            branch_id = branch_row["branch_id"].iloc[0] if not branch_row.empty else None
            order_val = bio_orders.get(branch_id, 0)
            color = cmap(order_val / max_order) if max_order > 0 else "black"
        else:
            color = "black"
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=width * pixels_per_width_unit)

    if all_x and all_y:
        ax.set_xlim(min(all_x), max(all_x))
        ax.set_ylim(max(all_y), min(all_y) if invert_y else max(all_y))

    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()
