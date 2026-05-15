"""Tests for figure export helpers (PDF/SVG/PNG raster defaults)."""

from pathlib import Path

import pytest

from snputils.visualization._figure_export import (
    default_savefig_kwargs,
    scatter_rasterized_for_path,
)


@pytest.mark.parametrize(
    "path, expect_raster",
    [
        ("a.pdf", True),
        ("x.SVG", True),
        (Path("plot.svg"), True),
        ("out.png", True),
        ("pure.svgz", False),  # not in our list
        ("noext", False),
    ],
)
def test_scatter_rasterized_for_path(path, expect_raster):
    assert scatter_rasterized_for_path(path) is expect_raster


def test_default_savefig_kwargs_svg_pdf_png():
    assert default_savefig_kwargs("f.svg") == {"dpi": 300}
    assert default_savefig_kwargs(Path("a.pdf")) == {"dpi": 300}
