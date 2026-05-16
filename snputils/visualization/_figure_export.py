"""Shared Matplotlib export defaults for PDF, SVG, PNG, and related formats."""

from __future__ import annotations

import pathlib
from typing import Any, Optional, Union

DEFAULT_PUBLICATION_DPI = 300

# Dense scatter layers: raster-embed in vector formats / match DPI for bitmap formats.
RASTERIZE_SCATTER_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".pdf", ".svg", ".eps", ".ps"}
)

# When the user does not pass ``dpi`` in ``savefig`` kwargs, use this resolution for
# rasterized content (and for wholly bitmap formats).
DEFAULT_DPI_SUFFIXES: frozenset[str] = frozenset(
    {".pdf", ".svg", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".eps", ".ps"}
)


def scatter_rasterized_for_path(save_path: Optional[Union[str, pathlib.Path]]) -> bool:
    """Return True when point clouds should be raster-embedded (PDF/SVG/PNG/...)."""
    if save_path is None:
        return False
    return pathlib.Path(save_path).suffix.lower() in RASTERIZE_SCATTER_SUFFIXES


def style_association_axes(
    ax=None,
    *,
    hide_bottom: bool = False,
    y_floor: Optional[float] = 0,
    x_floor: Optional[float] = None,
) -> None:
    """Publication-style axes for Manhattan and QQ plots.

    Hides top/right spines by default. Optionally hides the bottom spine (x-axis
  line only; tick marks and labels remain) and pins limits to zero without margin
  padding so points sit flush on the baseline.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if hide_bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", bottom=True)

    if y_floor is not None:
        _ymin, ymax = ax.get_ylim()
        if ymax <= y_floor:
            ymax = y_floor + 1.0
        ax.set_ylim(y_floor, ymax)
        ax.margins(y=0)

    if x_floor is not None:
        _xmin, xmax = ax.get_xlim()
        if xmax <= x_floor:
            xmax = x_floor + 1.0
        ax.set_xlim(x_floor, xmax)
        ax.margins(x=0)


def default_savefig_kwargs(save_path: Optional[Union[str, pathlib.Path]]) -> dict[str, Any]:
    """Keyword arguments merged into ``plt.savefig`` / ``fig.savefig`` (``dpi`` when applicable)."""
    if save_path is None:
        return {}
    suf = pathlib.Path(save_path).suffix.lower()
    if suf in DEFAULT_DPI_SUFFIXES:
        return {"dpi": DEFAULT_PUBLICATION_DPI}
    return {}
