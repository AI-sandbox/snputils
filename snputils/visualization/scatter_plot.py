from __future__ import annotations

import colorsys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Callable, Mapping, Optional, Union

from adjustText import adjust_text

from ._figure_export import default_savefig_kwargs, scatter_rasterized_for_path

PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
}


def _normalize_label_key(label: str) -> list[str]:
    s = str(label).strip()
    return [
        s,
        s.replace(" ", "_"),
        s.replace("_", " "),
    ]


def _resolve_label_color(
    label: str,
    idx: int,
    label_colors: Optional[Mapping[str, str]],
    cmap_fn: Callable[[int], np.ndarray],
) -> Union[str, np.ndarray]:
    if label_colors is not None:
        for key in _normalize_label_key(label):
            if key in label_colors:
                return label_colors[key]
    return cmap_fn(int(idx))


def _generate_distinct_colors(n: int) -> list:
    """Generate *n* visually distinct RGBA colours.

    Uses tab10 / tab20 for small palettes, then evenly-spaced HSV hues with
    alternating saturation / value so neighbouring colours stay distinguishable.
    """
    if n <= 10:
        base = cm.get_cmap("tab10", 10)
        return [base(i) for i in range(n)]
    if n <= 20:
        base = cm.get_cmap("tab20", 20)
        return [base(i) for i in range(n)]
    colors: list = []
    for i in range(n):
        hue = i / n
        sat = 0.55 + 0.35 * ((i % 3) / 2.0)
        val = 0.50 + 0.40 * (((i + 1) % 3) / 2.0)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append((r, g, b, 1.0))
    return colors


def scatter(
    dimredobj: np.ndarray,
    labels_file: str,
    abbreviation_inside_dots: bool = True,
    arrows_for_titles: bool = False,
    dots: bool = True,
    legend: bool = True,
    color_palette=None,
    show: bool = True,
    save_path: Optional[str] = None,
    *,
    label_mode: Optional[str] = None,
    style: str = "default",
    figsize: Optional[tuple[float, float]] = None,
    label_colors: Optional[Mapping[str, str]] = None,
    legend_outside: Optional[bool] = None,
    despine: Optional[bool] = None,
    axis_xlabel: Optional[str] = None,
    axis_ylabel: Optional[str] = None,
    point_size: Optional[float] = None,
    centroid_size: Optional[float] = None,
    point_alpha: Optional[float] = None,
    savefig_kwargs: Optional[dict] = None,
    equal_aspect: Optional[bool] = None,
) -> None:
    """
    Plot a scatter with group centroids and optional label styling.

    Args:
        dimredobj:
            Object produced by a dimensionality-reduction step, e.g.
            :class:`~snputils.processing.maasmds.maasMDS`,
            :class:`~snputils.processing.mdpca.mdPCA`, or
            :class:`~snputils.processing.pca.PCA`. Must expose ``X_new_`` (``(n, 2)`` embedding) and
            ``samples_`` (identifiers aligned with embedding rows).
        labels_file (str):
            TSV with columns ``indID`` and ``label``.
        abbreviation_inside_dots (bool):
            If True, show a short acronym inside each centroid marker.
        arrows_for_titles (bool):
            If True, draw arrows from text labels to centroids.
        dots (bool):
            If True, draw scatter points; if False, print coordinates and use text markers instead.
        legend (bool):
            If True, include a legend for group labels.
        color_palette (optional):
            Colormap or indexable color list; default palette is chosen automatically if None.
        show (bool, optional):
            If True, call ``plt.show()``; otherwise close the figure after saving. Default True.
        save_path (str, optional):
            If set, save the figure to this path (``plt.savefig``). Prefer ``.pdf`` or ``.svg`` for
            publication: dense scatter is rasterized at ``dpi`` (default 300) while axes and text stay
            vector. Bitmap formats (``.png``, ...) also default to that ``dpi``. Override via
            ``savefig_kwargs``.
        label_mode (str, optional):
            Overrides ``abbreviation_inside_dots``, ``arrows_for_titles``, and ``legend``.
            ``"legend"`` — legend plus abbreviations inside centroids.
            ``"acronym"`` — abbreviations inside centroids only.
            ``"arrow"`` — labels near centroids with ``adjustText`` arrows; best for many groups.
            ``None`` keeps the individual boolean flags.
        style (str):
            ``"default"`` — legacy appearance. ``"publication"`` — typography, despine, room for an outside legend,
            slightly larger markers, MDS-oriented axis labels.
        figsize (tuple, optional):
            Figure size in inches; chosen from ``style`` when None.
        label_colors (Mapping, optional):
            Map group labels (as in the TSV) to matplotlib color strings; unlisted labels use the palette.
        legend_outside (bool, optional):
            If True, place the legend outside the axes. Default True when ``style=="publication"``.
        despine (bool, optional):
            Hide top and right spines. Default True when ``style=="publication"``.
        axis_xlabel, axis_ylabel (str, optional):
            Axis labels; defaults depend on ``style``.
        point_size, centroid_size, point_alpha (float, optional):
            Override scatter sizes and point alpha.
        savefig_kwargs (dict, optional):
            Extra keyword arguments for ``plt.savefig`` when ``save_path`` is set.
        equal_aspect (bool, optional):
            If True, equal data aspect (typical for MDS/PCA). Default True when ``style="publication"``.

    Returns:
        None
    """
    if style not in ("default", "publication"):
        raise ValueError(f"style must be 'default' or 'publication', got {style!r}")

    if label_mode is not None:
        _valid = ("legend", "acronym", "arrow")
        if label_mode not in _valid:
            raise ValueError(f"label_mode must be one of {_valid}, got {label_mode!r}")
        if label_mode == "legend":
            legend, abbreviation_inside_dots, arrows_for_titles = True, True, False
        elif label_mode == "acronym":
            legend, abbreviation_inside_dots, arrows_for_titles = False, True, False
        elif label_mode == "arrow":
            legend, abbreviation_inside_dots, arrows_for_titles = False, False, True

    pub = style == "publication"
    if legend_outside is None:
        legend_outside = pub
    if despine is None:
        despine = pub
    if equal_aspect is None:
        equal_aspect = pub

    if figsize is None:
        if arrows_for_titles:
            figsize = (16.0, 14.0)
        elif pub and legend_outside:
            figsize = (12.0, 8.0)
        else:
            figsize = (10.0, 8.0)

    if axis_xlabel is None:
        axis_xlabel = "MDS 1" if pub else "Component 1"
    if axis_ylabel is None:
        axis_ylabel = "MDS 2" if pub else "Component 2"

    if point_size is None:
        point_size = 42.0 if pub else 30.0
    if centroid_size is None:
        centroid_size = 220.0 if pub else 300.0
    if point_alpha is None:
        point_alpha = 0.72 if pub else 0.6

    rc = PUBLICATION_RC if pub else {}
    savefig_kwargs = dict(savefig_kwargs or {})
    if pub and save_path and "bbox_inches" not in savefig_kwargs:
        savefig_kwargs["bbox_inches"] = "tight"
    if pub and save_path and "pad_inches" not in savefig_kwargs:
        savefig_kwargs["pad_inches"] = 0.08
    if save_path:
        for k, v in default_savefig_kwargs(str(save_path)).items():
            savefig_kwargs.setdefault(k, v)

    # Load labels from TSV
    labels_df = pd.read_csv(labels_file, sep="\t")

    # Ensure 'indID' is treated as a string
    labels_df["indID"] = labels_df["indID"].astype(str)

    # Filter labels based on the indIDs in dimredobj
    sample_ids = dimredobj.samples_
    filtered_labels_df = labels_df[labels_df["indID"].isin(sample_ids)]

    # Define unique colors for each group label
    unique_labels = filtered_labels_df["label"].unique()
    n_labels = len(unique_labels)

    if color_palette is not None:
        _cmap = color_palette

        def cmap_fn(i: int):
            return _cmap(int(i))
    else:
        _auto_colors = _generate_distinct_colors(n_labels)

        def cmap_fn(i: int):
            return _auto_colors[int(i) % len(_auto_colors)]

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize)

        centroids = {}
        all_scatter_x: list[float] = []
        all_scatter_y: list[float] = []

        for i, label in enumerate(unique_labels):
            sample_ids_for_label = filtered_labels_df[filtered_labels_df["label"] == label]["indID"]

            points = dimredobj.X_new_[np.isin(dimredobj.samples_, sample_ids_for_label)]

            c = _resolve_label_color(label, i, label_colors, cmap_fn)

            all_scatter_x.extend(points[:, 0].tolist())
            all_scatter_y.extend(points[:, 1].tolist())

            if dots:
                ec = "0.15" if pub else None
                lw = 0.25 if pub else 0.0
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    s=point_size,
                    color=c,
                    alpha=point_alpha,
                    label=label,
                    edgecolors=ec if ec and lw else "none",
                    linewidths=lw if lw else 0,
                    rasterized=scatter_rasterized_for_path(save_path),
                )
            else:
                for point in points:
                    print(point[0], point[1])
                    ax.text(
                        point[0],
                        point[1],
                        label[:2].upper(),
                        ha="center",
                        va="center",
                        color=c,
                        fontsize=8,
                        weight="bold",
                    )

            centroid = points.mean(axis=0)
            centroids[label] = centroid

            ax.scatter(
                *centroid,
                color=c,
                s=centroid_size,
                edgecolors="none",
                linewidths=0,
                zorder=5,
            )

            if abbreviation_inside_dots:
                ax.text(
                    centroid[0],
                    centroid[1],
                    label[:2].upper(),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8 if not pub else 9,
                    weight="bold",
                    zorder=6,
                )

        texts = []
        if arrows_for_titles:
            for label, centroid in centroids.items():
                idx = unique_labels.tolist().index(label)
                c_arrow = _resolve_label_color(label, idx, label_colors, cmap_fn)
                texts.append(
                    ax.text(
                        centroid[0],
                        centroid[1],
                        label,
                        color=c_arrow,
                        fontsize=9 if pub else 10,
                        weight="bold",
                        zorder=7,
                    )
                )

            if texts:
                target_x = [centroids[lbl][0] for lbl in centroids]
                target_y = [centroids[lbl][1] for lbl in centroids]
                adjust_text(
                    texts,
                    all_scatter_x,
                    all_scatter_y,
                    ax=ax,
                    force_text=(0.4, 0.6),
                    force_static=(0.3, 0.4),
                    force_pull=(0.005, 0.01),
                    force_explode=(0.2, 0.8),
                    expand=(1.2, 1.4),
                    max_move=(80, 80),
                    explode_radius="auto",
                    ensure_inside_axes=False,
                    prevent_crossings=True,
                    min_arrow_len=1,
                    iter_lim=3000,
                    target_x=target_x,
                    target_y=target_y,
                    arrowprops=dict(
                        arrowstyle="->",
                        color="gray",
                        alpha=0.8,
                        lw=1.0,
                        mutation_scale=12,
                        shrinkA=2,
                        shrinkB=2,
                    ),
                )

        ax.set_xlabel(axis_xlabel)
        ax.set_ylabel(axis_ylabel)

        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")

        if despine:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if legend:
            legend_kw: dict = {
                "frameon": True,
                "fancybox": False,
            }
            if pub:
                legend_kw.update(
                    {
                        "framealpha": 0.96,
                        "edgecolor": "#bfbfbf",
                        "fontsize": 9,
                    }
                )
            if legend_outside:
                legend_kw.update(
                    {
                        "loc": "upper left",
                        "bbox_to_anchor": (1.01, 1.0),
                        "borderaxespad": 0.0,
                    }
                )
                ax.legend(**legend_kw)
                fig.subplots_adjust(right=0.74)
            else:
                legend_kw.setdefault("loc", "upper right")
                ax.legend(**legend_kw)

        if save_path:
            fig.savefig(save_path, **savefig_kwargs)

        if show:
            plt.show()
        else:
            plt.close(fig)
