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


def default_savefig_kwargs(save_path: Optional[Union[str, pathlib.Path]]) -> dict[str, Any]:
    """Keyword arguments merged into ``plt.savefig`` / ``fig.savefig`` (``dpi`` when applicable)."""
    if save_path is None:
        return {}
    suf = pathlib.Path(save_path).suffix.lower()
    if suf in DEFAULT_DPI_SUFFIXES:
        return {"dpi": DEFAULT_PUBLICATION_DPI}
    return {}
