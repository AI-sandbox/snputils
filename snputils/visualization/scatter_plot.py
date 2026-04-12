import colorsys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Callable, Mapping, Optional, Union

from adjustText import adjust_text

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
    Plot a scatter plot with centroids for each group, with options for labeling and display styles.

    Args:
        dimredobj (np.ndarray):
            Reduced dimensionality data; expected to have `(n_haplotypes, 2)` shape.
        labels_file (str):
            Path to a TSV file with columns 'indID' and 'label', providing labels for coloring and annotating points.
        abbreviation_inside_dots (bool):
            If True, displays abbreviated labels (first 3 characters) inside the centroid markers.
        arrows_for_titles (bool):
            If True, adds arrows pointing to centroids with group labels displayed near the centroids.
        legend (bool):
            If True, includes a legend indicating each group label.
        color_palette (optional):
            Color map or list of colors to use for unique labels. Defaults to 'tab10' if None.
        show (bool, optional):
            Whether to display the plot. Defaults to False.
        save_path (str, optional):
            Path to save the plot image. If None, the plot is not saved.
        label_mode (str, optional):
            Overrides ``abbreviation_inside_dots``, ``arrows_for_titles``, and ``legend``.
            ``"legend"`` — show a legend with abbreviations inside centroids.
            ``"acronym"`` — abbreviations inside centroids, no legend.
            ``"arrow"`` — population labels placed near centroids with non-overlapping arrows
            (uses ``adjustText``), no legend, no abbreviations. Best for many populations.
            Default ``None`` uses the individual boolean flags.
        style (str):
            ``"default"`` — current behaviour. ``"publication"`` — typography, despine, room for legend,
            slightly larger markers, axis labels suitable for figures.
        figsize (tuple, optional):
            Figure size in inches. If None, chosen from ``style``.
        label_colors (Mapping, optional):
            Map from group label (as in the TSV) to a matplotlib color string. Labels not present in the map
            fall back to ``color_palette`` (or tab10). Useful for study-specific palettes; snputils does not
            ship fixed maps for particular cohorts.
        legend_outside (bool, optional):
            If True, legend is placed outside the axes on the right (avoids covering points).
            Default: True when ``style=="publication"``, else False.
        despine (bool, optional):
            Hide top and right spines. Default: True when ``style=="publication"``, else False.
        axis_xlabel, axis_ylabel (str, optional):
            Axis labels. Defaults depend on ``style``.
        point_size, centroid_size, point_alpha (float, optional):
            Override scatter sizes and point transparency.
        savefig_kwargs (dict, optional):
            Extra keyword arguments passed to ``plt.savefig`` when ``save_path`` is set.
        equal_aspect (bool, optional):
            If True, set equal data aspect ratio (typical for MDS / PCA). Default: True for
            ``style="publication"``, else False.

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
                    rasterized=bool(save_path and str(save_path).lower().endswith((".png", ".pdf"))),
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
                    expand_points=(3.0, 3.0),
                    expand_text=(1.2, 2.0),
                    expand_objects=(3.0, 3.0),
                    max_move=(80, 80),
                    explode_radius="auto",
                    ensure_inside_axes=False,
                    prevent_crossings=True,
                    min_arrow_len=1,
                    time_lim=5,
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
