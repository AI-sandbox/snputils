import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from snputils.ancestry.genobj.local import LocalAncestryObject
from snputils.visualization.constants import CHROM_SIZES, snputils_palette
from snputils.visualization.lai import plot_lai


def test_plot_lai_separates_chromosome_blocks_on_x_axis():
    lai = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.int8,
    )
    chromosomes = np.array(["1", "1", "2", "2", "3", "3"], dtype=object)
    laiobj = LocalAncestryObject(
        haplotypes=["sample0.0", "sample0.1", "sample1.0", "sample1.1"],
        lai=lai,
        samples=["sample0", "sample1"],
        ancestry_map={"0": "AFR", "1": "EUR"},
        chromosomes=chromosomes,
    )

    plot_lai(
        laiobj,
        colors={"AFR": "#4C78A8", "EUR": "#F58518"},
        sort=False,
        figsize=(8, 3),
        legend=False,
        scale=1,
    )

    ax = plt.gca()
    xticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert xticklabels == ["chr1", "chr2", "chr3"]
    assert len(ax.lines) == 2
    plt.close(ax.figure)


def test_plot_lai_scales_chromosome_widths_from_constants():
    lai = np.array(
        [
            [0, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=np.int8,
    )
    chromosomes = np.array(["1", "2", "21"], dtype=object)
    laiobj = LocalAncestryObject(
        haplotypes=["sample0.0", "sample0.1"],
        lai=lai,
        samples=["sample0"],
        ancestry_map={"0": "AFR", "1": "EUR"},
        chromosomes=chromosomes,
    )

    plot_lai(
        laiobj,
        colors={"AFR": "#4C78A8", "EUR": "#F58518"},
        sort=False,
        figsize=(6, 2),
        legend=False,
        scale=1,
    )

    ax = plt.gca()
    assert len(ax.lines) == 2
    first_boundary = float(ax.lines[0].get_xdata()[0])
    second_boundary = float(ax.lines[1].get_xdata()[0])
    first_width = first_boundary
    second_width = second_boundary - first_boundary

    expected_ratio = CHROM_SIZES["hg38"]["1"] / CHROM_SIZES["hg38"]["2"]
    actual_ratio = first_width / second_width
    assert np.isclose(actual_ratio, expected_ratio, rtol=1e-6)
    plt.close(ax.figure)


def test_plot_lai_uses_snputils_palette_when_colors_are_omitted():
    laiobj = LocalAncestryObject(
        haplotypes=["sample0.0", "sample0.1"],
        lai=np.array([[0, 1], [1, 0]], dtype=np.int8),
        samples=["sample0"],
        ancestry_map={"0": "AFR", "1": "EUR"},
    )

    plot_lai(
        laiobj,
        sort=False,
        figsize=(4, 2),
        legend=True,
        scale=1,
    )

    ax = plt.gca()
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["AFR", "EUR"]
    assert [patch.get_facecolor() for patch in legend.get_patches()] == [
        mcolors.to_rgba(color) for color in snputils_palette[:2]
    ]
    plt.close(ax.figure)


def test_plot_lai_is_available_from_viz_namespace():
    from snputils import viz

    assert viz.plot_lai is plot_lai
